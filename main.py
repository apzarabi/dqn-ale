import os

import numpy as np
import tensorflow as tf
from tqdm import tqdm

from atari_environment import AtariEnvironment
from config import cfg
from dqn import DQN
from experience_replay import CircularBuffer, ExperienceReplay
from utils import add_simple_summary
from generative_model.multigraph_res2_vqvae_5frames import VQVAE


def create_generative_models(sess):
    ret = []
    for i in range(cfg.num_minimal_actions):
        scope = "gen_{}".format(i)
        with tf.variable_scope(scope):
            vae = VQVAE(data_directory="../../atari_ae/data/res-2-vq-vae/{}/".format(cfg.game_name),
                        model_name="vq-vae-action-{}".format(i), graph=tf.get_default_graph())
            iterator, iterable = vae.load_tfrecords(batch_size=vae.LOG_DATA_SIZE, repeat=False)
            tf.logging.info("VAE MODEL PATH {}".format(vae.MODEL_PATH))
            if tf.train.latest_checkpoint(vae.MODEL_PATH):
                tf.logging.info("restoring network for action: {}".format(i))
                vae.restore(iterable, scope=scope, sess=sess)
                tf.logging.info("restored from {} step {}".format(tf.train.latest_checkpoint(vae.MODEL_PATH),
                                                                  vae.global_step.eval(session=vae.sess)))
            else:
                raise RuntimeError("No checkpoints to restore from")
            vae.compute_per_pixel_distribution(iterator, iterable, debug_mode=cfg.debug_mode)
        ret.append(vae)
    return ret


def restore_or_initialize_weights(sess, dqn):
    restore_dir_provided = False
    if cfg.restore_dir is not None and cfg.restore_dir != "":
        restore_dir = os.path.join(cfg.restore_dir, "model/")
        restore_dir_provided = True
    else:
        restore_dir = cfg.save_dir
    latest_ckpt = tf.train.latest_checkpoint(restore_dir)
    tf.logging.info("restore_dir, latest_ckpt {} {}".format(restore_dir, latest_ckpt))

    if latest_ckpt is None:
        if cfg.evaluation_mode or restore_dir_provided:
            raise RuntimeError("Latest Checkpoint shouldn't be none in evaluation mode or when restore_dir is provided")
        tf.logging.info(" Initializing weights")
        sess.run(tf.global_variables_initializer())
    else:
        tf.logging.info(" Restoring weights from checkpoint %s" % latest_ckpt)
        dqn.saver.restore(sess, latest_ckpt)


def main(_):
    # Reproducability
    tf.reset_default_graph()
    np.random.seed(cfg.random_seed)
    tf.set_random_seed(cfg.random_seed)

    # Logging
    summary_writer = tf.summary.FileWriter(cfg.log_dir)

    if not tf.gfile.Exists(cfg.save_dir):
        tf.gfile.MakeDirs(cfg.save_dir)

    # TODO handel this
    episode_results_path = os.path.join(cfg.save_dir, "episodeResults.csv")
    episode_results = tf.gfile.GFile(episode_results_path, "w")
    episode_results.write("episode,reward,steps\n")

    # Setup ALE and DQN graph
    obs_shape = (84, 84, 1)
    input_height, input_width, _ = obs_shape

    dqn = DQN(input_height, input_width, cfg.num_actions)

    # Global step
    global_step = tf.train.get_or_create_global_step()
    increment_step = tf.assign_add(global_step, 1)

    # Save all variables
    vars_to_save = tf.get_collection(
        tf.GraphKeys.TRAINABLE_VARIABLES, scope=cfg.q_scope
    )
    saver = tf.train.Saver(var_list=vars_to_save)

    # Handle loading specific variables
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    restore_or_initialize_weights(sess, dqn)
    sess.run(dqn.copy_to_target)

    # ##### Restoring AEs ########
    vaes = create_generative_models(sess)
    # ############################

    # Initialize ALE
    postprocess_frame = lambda frame: sess.run(
        dqn.process_frame, feed_dict={dqn.image: frame}
    )
    env = AtariEnvironment(obs_shape, postprocess_frame)

    # Replay buffer
    replay_buffer = ExperienceReplay(cfg.replay_buffer_size, obs_shape)

    # Perform random policy to get some training data
    with tqdm(
        total=cfg.seed_frames, disable=cfg.disable_progress or cfg.evaluate
    ) as pbar:
        seed_steps = 0
        while seed_steps * cfg.frame_skip < cfg.seed_frames and not cfg.evaluate:
            action = np.random.randint(cfg.num_actions)
            reward, next_state, terminal = env.act(action)
            seed_steps += 1

            replay_buffer.append(
                next_state[:, :, -1, np.newaxis], action, reward, terminal
            )

            if terminal:
                pbar.update(env.episode_frames)
                env.reset(inc_episode_count=False)

    if cfg.evaluate:
        assert cfg.max_episode_count > 0
    else:
        assert len(replay_buffer) >= cfg.seed_frames // cfg.frame_skip

    # Main training loop
    steps = tf.train.global_step(sess, global_step)
    env.reset(inc_episode_count=False)
    terminal = False

    total = cfg.max_episode_count if cfg.evaluate else cfg.num_frames
    with tqdm(total=total, disable=cfg.disable_progress) as pbar:
        # Loop while we haven't observed our max frame number
        # If we are at our max frame number we will finish the current episode
        while (
            not (
                # We must be evaluating or observed the last frame
                # As well as be terminal
                # As well as seen the maximum episode number
                (steps * cfg.frame_skip > cfg.num_frames or cfg.evaluate)
                and terminal
                and env.episode_count >= cfg.max_episode_count
            )
        ):
            # Epsilon greedy policy with epsilon annealing
            if not cfg.evaluate and steps * cfg.frame_skip < cfg.eps_anneal_over:
                # Only compute epsilon step while we're still annealing epsilon
                epsilon = cfg.eps_initial - steps * (
                    (cfg.eps_initial - cfg.eps_final) / cfg.eps_anneal_over
                )
            else:
                epsilon = cfg.eps_final

            # Epsilon greedy policy
            if np.random.uniform() < epsilon:
                action = np.random.randint(0, cfg.num_actions)
            else:
                action = sess.run(dqn.action, feed_dict={dqn.S: [env.state]})

            # Perform environment step
            steps = sess.run(increment_step)
            reward, next_state, terminal = env.act(action)

            if not cfg.evaluate:
                replay_buffer.append(
                    next_state[:, :, -1, np.newaxis], action, reward, terminal
                )

                # Sample and do gradient updates
                if steps % cfg.learning_freq == 0:
                    placeholders = [
                        dqn.S,
                        dqn.actions,
                        dqn.rewards,
                        dqn.S_p,
                        dqn.terminals,
                    ]
                    batch = replay_buffer.sample(cfg.batch_size)
                    train_op = [dqn.train]
                    if steps % cfg.log_summary_every:
                        train_op.append(dqn.summary)
                    result = sess.run(
                        train_op,
                        feed_dict=dict(zip(placeholders, batch)),
                    )
                    if len(result) > 1:
                        summary_writer.add_summary(result[-1], global_step=steps)
                if steps % cfg.target_update_every == 0:
                    sess.run([dqn.copy_to_target])
                if steps % cfg.model_chkpt_every == 0:
                    saver.save(
                        sess, "%s/model_epoch_%04d" % (cfg.log_dir, steps)
                    )

            if terminal:
                episode_results.write(
                    "%d,%d,%d\n"
                    % (env.episode_count, env.episode_reward, env.episode_frames)
                )
                episode_results.flush()
                # Log episode summaries to Tensorboard
                add_simple_summary(summary_writer, "episode/reward", env.episode_reward, env.episode_count)
                add_simple_summary(summary_writer, "episode/frames", env.episode_frames, env.episode_count)

                pbar.update(env.episode_frames if not cfg.evaluate else 1)
                env.reset()

    episode_results.close()
    tf.logging.info(
        "Finished %d %s"
        % (
            cfg.max_episode_count if cfg.evaluate else cfg.num_frames,
            "episodes" if cfg.evaluate else "frames",
        )
    )


if __name__ == "__main__":
    tf.app.run()
