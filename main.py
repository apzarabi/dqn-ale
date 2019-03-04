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
        with tf.variable_scope(scope):
            vae.compute_per_pixel_distribution(iterator, iterable, debug_mode=cfg.debug_mode)
        ret.append(vae)
    return ret


# Bad practice! using these as global variables so not to instantiate them every time.
imagined_batch_curr_obs = np.zeros((cfg.batch_size, 84, 84, 4))
imagined_batch_actions = np.zeros(cfg.batch_size)
imagined_batch_rewards = np.zeros(cfg.batch_size)
imagined_batch_next_obs = np.zeros((cfg.batch_size, 84, 84, 4))
imagined_batch_terminals = np.zeros(cfg.batch_size)

FREEWAY_UP = [2, 6, 7, 10, 14, 15]
FREEWAY_DOWN = [5, 8, 9, 13, 16, 17]
FREEWAY_NOOP = [0, 1, 3, 4, 11, 12]
FREEWAY_ACTION_MAP = [FREEWAY_NOOP, FREEWAY_UP, FREEWAY_DOWN]


def imagined_batch(vaes, er_batch_actions):
    # ##### Concating of generated experience to actual ones
    global imagined_batch_curr_obs, imagined_batch_actions, imagined_batch_rewards, imagined_batch_next_obs, \
        imagined_batch_terminals

    action_counts = []
    for _ in range(cfg.num_minimal_actions):
        action_counts.append([])
    _sum_action_count = 0
    unique, count = np.unique(er_batch_actions, return_counts=True)
    er_action_counts = dict(zip(unique, count))
    for i in range(cfg.num_minimal_actions):
        for action_idx in FREEWAY_ACTION_MAP[i]:
            if action_idx in er_batch_actions:
                action_counts[i].extend([action_idx]*er_action_counts[action_idx])
                _sum_action_count += er_action_counts[action_idx]

    assert len(er_batch_actions) == cfg.batch_size, "len(er_batch_actions)={}, batch_size={}\n" \
                                                    "er_batch_actions= ".format(len(er_batch_actions),
                                                                                cfg.batch_size, er_batch_actions)
    assert _sum_action_count == cfg.batch_size, "Sum action count == {} , batch size == {}, er_batch_actions" \
                                                "={}".format(_sum_action_count, cfg.batch_size,
                                                             er_batch_actions)

    # Make sure that non of the actions are 0. If that happens, CUDA is gonna raise an error
    max_action_idx = 0
    for i in range(cfg.num_minimal_actions):
        if len(action_counts[i]) > len(action_counts[max_action_idx]):
            max_action_idx = i
    for i in range(cfg.num_minimal_actions):
        if len(action_counts[i]) == 0:
            action_counts[i].append(FREEWAY_ACTION_MAP[i][0])
            action_counts[max_action_idx].pop()

    idx = 0
    for i in range(cfg.num_minimal_actions):
        out, summary = vaes[i].sample(len(action_counts[i]), compute_summaries=False)
        out = out[:len(action_counts[i])]
        imagined_batch_curr_obs[idx: idx+len(action_counts[i]), :, :, :] = out[:, :, :, :4]
        imagined_batch_next_obs[idx: idx+len(action_counts[i]), :, :, :] = out[:, :, :, 1:]
        imagined_batch_actions[idx: idx+len(action_counts[i])] = action_counts[i]
        idx += len(action_counts[i])

    assert idx == cfg.batch_size, "Idx {} {}".format(idx, er_batch_actions)
    assert len(imagined_batch_actions) == cfg.batch_size, "Imagined batch action== {} , batch size == {}, " \
                                                            "er_batch_actions={}, imagined batch " \
                                                            "action={}".format(len(imagined_batch_actions),
                                                                               cfg.batch_size,
                                                                               er_batch_actions,
                                                                               imagined_batch_actions)

    return imagined_batch_curr_obs, imagined_batch_actions, imagined_batch_rewards, imagined_batch_next_obs,\
        imagined_batch_terminals,


def restore_or_initialize_weights(sess, dqn, saver):
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
        saver.restore(sess, latest_ckpt)


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
    vars_to_save = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="agent/q")
    vars_to_save.append(global_step)
    saver = tf.train.Saver(var_list=vars_to_save)

    # Handle loading specific variables
    sess_config = tf.ConfigProto()
    sess_config.gpu_options.allow_growth = True
    sess = tf.Session(config=sess_config)

    restore_or_initialize_weights(sess, dqn, saver)
    sess.run(dqn.copy_to_target)

    # ##### Restoring AEs ########
    vaes = create_generative_models(sess)
    image_summaries = []
    image_summaries_ph = tf.placeholder(tf.float32, shape=(4, 84, 84, 4), name="image_summaries_placeholder")
    for i in range(4):
        for j in range(4):
            image_summaries.append(
                tf.summary.image("VAE_OUT_{}_{}".format(i, j),
                                 tf.reshape(image_summaries_ph[i, :, :, j], (1, 84, 84, 1)))
            )
    # ############################

    summary_writer.add_graph(tf.get_default_graph())
    summary_writer.add_graph(vaes[0].graph)
    summary_writer.add_graph(vaes[1].graph)
    summary_writer.add_graph(vaes[2].graph)

    summary_writer.flush()

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
                    if steps % (cfg.learning_freq * cfg.model_freq) == 0:
                        experience_batch = batch
                        batch = imagined_batch(vaes, batch[1])
                        if steps / (cfg.learning_freq*cfg.model_freq) < 10:
                            placeholders.append(image_summaries_ph)
                            batch = list(batch)
                            batch.append(batch[0][np.random.randint(0, 32, size=4), :, :, :])
                            train_op.extend(image_summaries)
                    if steps % cfg.log_summary_every:
                        train_op.append(dqn.summary)
                    result = sess.run(
                        train_op,
                        feed_dict=dict(zip(placeholders, batch)),
                    )
                    if len(result) > 1:
                        for i in range(1, len(result)):
                            summary_writer.add_summary(result[i], global_step=steps)
                if steps % cfg.target_update_every == 0:
                    sess.run([dqn.copy_to_target])
                if steps % cfg.model_chkpt_every == 0:
                    saver.save(
                        sess, "%s/model_epoch_%04d" % (cfg.save_dir, steps)
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
