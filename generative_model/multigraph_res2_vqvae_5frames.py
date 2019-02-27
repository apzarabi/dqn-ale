import time
import sys
import numpy as np
import tensorflow as tf
import imageio

import generative_model.config as config
from generative_model.autoencoder import AbstractAutoEncoder
# from common.profiler import profile


class VQVAE(AbstractAutoEncoder):
    def __init__(self, log_path="../data/vq-vae-5/log/", model_path="../data/vq-vae-5/models/", data_directory=None,
                 model_name="vq-vae-5", action_number=0, graph=None):
        """
        :param action_number shoud be in [0, 1, 2] for freeway
        """
        if data_directory:
            log_path = data_directory + "log/" + model_name + "/"
            model_path = data_directory + "models/" + model_name + "/"
        super().__init__(log_path=log_path, model_path=model_path)
        self.MODEL_NAME = model_name
        self.ACTION_NUMBER = action_number
        self.INPUT_PATH = config.INPUT_PATH.format(self.ACTION_NUMBER)
        self.graph = graph

        self.RGB = False    # Run this model only on grayscale images
        self.five_frames = True
        self.num_codes = self.EMBEDDING_SIZE

        self.latent_height = 21
        self.latent_width = 21
        self.latent_size = self.latent_height*self.latent_width  # Since the input to the decoder is 10*10

        self.num_actions = 3   # for freeway

        self.code_size = 32
        self.num_samples_per_log = 5

        # init network variables
        self.inp = None
        self.enc_out = None
        self.latent_space = None
        self.vq_idx = None
        self.vq_out = None
        self.dec_input = None
        self.decoded = None
        self.global_step = None
        self.action = None
        self.reward = None
        self.terminal = None
        self.vq_recon_loss = None
        self.action_prediction_loss = None

        self.per_pixel_count = None

    def res_block(self, inp, sizes, name, activation=tf.nn.relu):
        """

        :param inp:
        :param sizes:
        :param name:
        :type activation: Optional
        :return:
        """
        with tf.variable_scope("res_block_{}".format(name)):
            net = tf.layers.conv2d(inp, filters=32, kernel_size=sizes[0], padding="same", strides=1, activation=tf.nn.relu,
                                   name="c1")
            net = tf.layers.conv2d(net, filters=32, kernel_size=sizes[1], padding="same", strides=1, activation=activation,
                                   name="c2")
            return inp + net

    def _encoder(self):
        """
        :param inp: should be a tensor with shape (-1, 84, 84, 5) or (-1, 210, 160, 3*4)
        :return:
        """
        with tf.variable_scope("encoder"):
            # In this case, input is a tuple of frames, action, reward, terminal
            inp = self.inp[0]
            for i in range(5):
                tf.summary.image('Input_Image_{}'.format(i),
                                 tf.reshape(inp[0, :, :, i], (1, self.HEIGHT, self.WIDTH, 1)),
                                 collections=['per_write'])
            r1 = tf.reshape(inp, (-1, self.HEIGHT, self.WIDTH, 5), name="r1")
            c1 = tf.layers.conv2d(r1, filters=32, kernel_size=6, padding="same",
                                  strides=2, activation=tf.nn.relu, name="c1")
            c2 = tf.layers.conv2d(c1, filters=32, kernel_size=6, padding="same",
                                  strides=2, activation=tf.nn.relu, name="c2")
            c3 = self.res_block(c2, [6, 1], "c3")
            c4 = self.res_block(c3, [6, 1], "c4", activation=None)

            rs = tf.reshape(c4, (-1, self.latent_size, self.code_size))
        self.enc_out = rs

    def _vector_quantize(self):
        with tf.variable_scope("Vector_Quantize"):
            # # VQ     #######
            distances = tf.norm(
                tf.expand_dims(self.enc_out, 2) -
                tf.reshape(self.latent_space, [1, 1, self.num_codes, self.code_size]),
                axis=3)
            assignments = tf.argmin(distances, 2)
            one_hot_assignments = tf.one_hot(assignments, depth=self.num_codes)
            one_hot_assignments = tf.stop_gradient(one_hot_assignments)
            print("one hot shape", one_hot_assignments.shape)
            nearest_codebook_entries = tf.reduce_sum(
                tf.expand_dims(one_hot_assignments, -1) *
                tf.reshape(self.latent_space, [1, 1, self.num_codes, self.code_size]),
                axis=2)
            vq_out, vq_idx = nearest_codebook_entries, assignments

            # based on Sergey Ioffe's trick
            dec_input = self.enc_out + tf.stop_gradient(vq_out - self.enc_out)

        self.vq_idx = vq_idx
        self.vq_out = vq_out
        self.dec_input = dec_input

    def _build_decoder_network(self, input, reuse=False):
        with tf.variable_scope("decoder", reuse=reuse):
            dec_input_reshaped = tf.reshape(input, (-1, self.latent_height, self.latent_width, self.code_size))
            dc1 = self.res_block(dec_input_reshaped, [6, 6], "dc1")
            dc2 = self.res_block(dc1, [6, 6], "dc2")
            dc3 = tf.layers.conv2d_transpose(dc2, filters=32, kernel_size=6, padding="same",
                                             strides=2, activation=tf.nn.relu, name="dc1")
            decoded = tf.layers.conv2d_transpose(dc3, filters=5, kernel_size=6, padding="same",
                                                 strides=2, activation=tf.nn.sigmoid, name="dc2")
        return decoded

    def _decoder(self):
            # for i in range(5):
            #     tf.summary.image('A_Recon_Image_{}'.format(i),
            #                      tf.reshape(decoded[0, :, :, i], (1, self.HEIGHT, self.WIDTH, 1)),
            #                      collections=['per_write'])
            # for j in range(self.num_samples_per_log):
            #     for i in range(5):
            #         tf.summary.image('Imagined_Image_{}_{}'.format(j, i),
            #                          tf.reshape(decoded[j, :, :, i], (1, self.HEIGHT, self.WIDTH, 1)),
            #                          collections=['decoder_output'])

            # For using in DQN's code
            # for i in range(5):
            #     self.decoder_out_summary = tf.summary.image('Imagined_action_{}_{}'.format(self.ACTION_NUMBER, i),
            #                                                 tf.reshape(decoded[0, :, :, i],
            #                                                            (1, self.HEIGHT, self.WIDTH, 1)))
        self.decoded = self._build_decoder_network(self.dec_input)

    def _vqvae_recon_loss(self):
        with tf.variable_scope("loss"):
            if self.five_frames:
                inp = self.inp[0]
            else:
                inp = self.inp
            r_loss = tf.losses.mean_squared_error(inp, self.decoded, reduction=tf.losses.Reduction.SUM)
            dict_loss = tf.losses.mean_squared_error(tf.stop_gradient(self.enc_out), self.vq_out,
                                                     reduction=tf.losses.Reduction.SUM)
            commitment_loss = tf.losses.mean_squared_error(tf.stop_gradient(self.vq_out), self.enc_out,
                                                           reduction=tf.losses.Reduction.SUM)
            tf.summary.scalar("Reconstruction_Loss", r_loss, collections=['per_log'])
            tf.summary.scalar("Dictionary_Loss", dict_loss, collections=['per_log'])

            if self.RGB:
                loss = r_loss + 3.0 * dict_loss + 10.0 * commitment_loss
            else:
                loss = r_loss + 2.0 * dict_loss + 0.25 * commitment_loss
            tf.summary.scalar("Loss", loss, collections=['per_log'])
        self.vq_recon_loss = loss

    def _make_train_op(self, global_step, loss):
        # Gradient clipping
        optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate)
        gvs = optimizer.compute_gradients(loss)
        # Remove variables with no gradients
        gvs = [item for item in gvs if item[0] is not None]
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        train_op = optimizer.apply_gradients(capped_gvs, global_step=global_step)
        return train_op

    def _build_network(self, inp):
        # tf.reset_default_graph()
        with tf.variable_scope("input"):
            self.latent_space = tf.get_variable("latent_space", (self.num_codes, self.code_size), dtype=tf.float32,
                                                initializer=tf.random_normal_initializer(stddev=0.08))
        self.inp = inp
        self._encoder()
        self._vector_quantize()
        self._decoder()
        self.global_step = tf.train.get_or_create_global_step()

        self._vqvae_recon_loss()

        self.vq_recon_train_op = self._make_train_op(self.global_step, self.vq_recon_loss)

    def train(self, train_data_file, for_action=False):
        print("reading data ...")
        iterator, iterable = self.load_tfrecords(batch_size=self.BATCH_SIZE)
        large_batch_iterator, large_batch_iterable = self.load_tfrecords(batch_size=self.LOG_DATA_SIZE)
        looping_iterator, looping_iterable = self.load_tfrecords(batch_size=self.LOG_DATA_SIZE, repeat=True)
        print("reading data: done")

        if tf.train.latest_checkpoint(self.MODEL_PATH):
            print("restoring network")
            self.restore(iterable)
            print("restored from", tf.train.latest_checkpoint(self.MODEL_PATH))
            print("step:", self.global_step.eval(session=self.sess))

        else:
            print("building network")
            self._build_network(iterable)
            self.sess = tf.Session(config=self.sess_config)
            # self.sess.run(tf.global_variables_initializer())
            self.sess.run(tf.variables_initializer(self.graph.get_collection("variables")))
            print("done building")
        num_samples = 100 * 1000 - 8

        enc_out_placeholder, idx_placeholder, latent_space_placeholder = self._histogram_summaries()

        per_log_summaries = tf.summary.merge_all(key='per_log')
        per_write_summaries = tf.summary.merge_all(key='per_write')
        manual_summaries = tf.summary.merge_all(key='manual')
        decoder_output_summary = tf.summary.merge_all(key='decoder_output')
        train_writer = tf.summary.FileWriter(self.LOG_PATH)

        batch_num = 0
        total_ping = time.time()
        self.sess.run(looping_iterator.initializer)
        for i in range(0, self.num_epochs):
            print("epoch begins!")
            self.sess.run(iterator.initializer)
            while True:
                try:
                    if batch_num % self.PRINT_TIME == 0:
                        ping = time.time()
                    else:
                        _ = self.sess.run(self.vq_recon_train_op)

                    if batch_num % self.PRINT_TIME == 0:
                        print("#{:>5} batch took {:>6.6} {:>6.6}".format(batch_num, time.time() - ping,
                                                                         time.time() - total_ping))

                    if batch_num % self.LOG_WRITE_INTERVAL == 0:
                        self._write_per_log_summaries(enc_out_placeholder, idx_placeholder,  latent_space_placeholder,
                                                      looping_iterable, manual_summaries, per_log_summaries, total_ping,
                                                      train_writer)

                    batch_num += 1
                except tf.errors.OutOfRangeError:  # end of epoch
                    if i % self.SAVE_IMAGES_SUMMARY == 0:
                        self._write_image_summaries(decoder_output_summary, large_batch_iterable, large_batch_iterator,
                                                    looping_iterable, per_write_summaries, total_ping, train_writer)
                        self.save(self.sess, self.global_step)

                    break

        train_writer.flush()
        self.save(self.sess, self.global_step)

    def _write_image_summaries(self, decoder_output_summary, large_batch_iterable, large_batch_iterator,
                               looping_iterable, per_write_summaries, total_ping, train_writer):
        ping = time.time()
        summary_tr, step = self.sess.run([per_write_summaries, self.global_step],
                                         feed_dict={self.inp: self.sess.run(looping_iterable)})
        train_writer.add_summary(summary_tr, step)
        if self.per_pixel_count is None:
            self._compute_per_pixel_distribution(large_batch_iterator, large_batch_iterable)
        latent_input_vals = self._sample_latent_from_per_pixel_distribution(self.num_samples_per_log)
        summary_tr = self.sess.run(decoder_output_summary,
                                   feed_dict={self.dec_input: latent_input_vals})
        train_writer.add_summary(summary_tr, step)
        train_writer.flush()
        print("\t# imag log took {:>6.6} {:>6.6} {:>5}".format(time.time() - ping,
                                                               time.time() - total_ping, step))

    def _write_per_log_summaries(self, enc_out_placeholder, idx_placeholder, latent_space_placeholder,
                                 looping_iterable, manual_summaries, per_log_summaries, total_ping, train_writer):
        ping = time.time()
        looping_iterable_eval = self.sess.run(looping_iterable)
        latent_space_eval, idx_eval, enc_out_eval, summary_tr, step = self.sess.run(
            [self.latent_space, self.vq_idx, self.enc_out, per_log_summaries, self.global_step],
            feed_dict={self.inp: looping_iterable_eval})
        print("\t\t\tEOE: ", np.max(enc_out_eval), np.min(enc_out_eval), enc_out_eval.shape,
              np.squeeze(enc_out_eval).shape)
        if np.isnan(np.max(enc_out_eval)):
            self.save(self.sess, self.global_step)
            np.save("out_latent_eval", enc_out_eval)
            np.save("out_looping_iter", looping_iterable_eval)
            raise RuntimeError("holy moly")
        train_writer.add_summary(summary_tr, step)
        latent_space_eval = np.squeeze(latent_space_eval)
        summary_tr = self.sess.run(manual_summaries, feed_dict={
            latent_space_placeholder: np.ndarray.flatten(latent_space_eval),
            idx_placeholder: np.ndarray.flatten(idx_eval),
            enc_out_placeholder: np.ndarray.flatten(enc_out_eval)})
        train_writer.add_summary(summary_tr, step)
        train_writer.flush()
        print("\t# data log took {:>6.6} {:>6.6} {:>5}".format(time.time() - ping,
                                                               time.time() - total_ping, step))

    def _histogram_summaries(self):
        with tf.variable_scope("placeholders_for_histogram"):
            idx_placeholder = tf.placeholder(tf.float32, shape=[self.latent_height * self.latent_width * self.LOG_DATA_SIZE],
                                             name="idx_placeholder")
            latent_space_placeholder = tf.placeholder(tf.float32, shape=[self.num_codes * self.code_size],
                                                      name="latent_space_placeholder")
            enc_out_placeholder = tf.placeholder(tf.float32,
                                                 shape=[self.latent_size * self.code_size * self.LOG_DATA_SIZE],
                                                 name="enc_out_placeholder")
        tf.summary.histogram("IDX_hist", idx_placeholder, collections=['manual'])
        tf.summary.histogram("Latent_hist", latent_space_placeholder, collections=['manual'])
        tf.summary.histogram("Enc_out_hist", enc_out_placeholder, collections=['manual'])
        return enc_out_placeholder, idx_placeholder, latent_space_placeholder

    def _sample_latent_from_per_pixel_distribution(self, num_of_requested_samples):
        latent_eval = self.latent_space.eval(session=self.sess)
        idx_inp = np.zeros((num_of_requested_samples, self.latent_size))
        for i in range(self.latent_size):
            idx_inp[:, i] = np.random.choice(self.num_codes,
                                             p=self.per_pixel_count[i] / self.num_samples_in_per_pixel_count,
                                             size=num_of_requested_samples)
        idx_inp = idx_inp.astype(np.int32)

        latent_input_vals = latent_eval[idx_inp]

        return latent_input_vals

    def compute_per_pixel_distribution(self, iterator, iterable, debug_mode=False):
        """
        Save per pixel distribution in self.per_pixel_count
        """
        ping = time.time()
        self.per_pixel_count = np.zeros((self.latent_size, self.num_codes))
        self.num_samples_in_per_pixel_count = 0
        self.sess.run(iterator.initializer)
        i = 0
        while True:
            try:
                idx_eval = self.vq_idx.eval(session=self.sess, feed_dict={self.inp: self.sess.run(iterable)})
                self.num_samples_in_per_pixel_count += idx_eval.shape[0]
                for pos in range(self.latent_size):
                    for sample in range(idx_eval.shape[0]):
                        self.per_pixel_count[pos, idx_eval[sample, pos]] += 1
                i += 1
                if debug_mode and i > 5:
                    break
            except tf.errors.OutOfRangeError:
                break

        # based on
        # https://stackoverflow.com/questions/49713210/how-to-sample-in-tensorflow-by-custom-probability-distribution
        self.idx_inp = tf.Variable(np.zeros((32, self.latent_size)), dtype=tf.int64)
        self.sess.run(tf.initialize_variables([self.idx_inp]))
        true_classes = tf.constant(np.array([np.arange(self.num_codes)]))
        for i in range(self.latent_size):
            self.sess.run(self.idx_inp[:, i].assign(tf.nn.fixed_unigram_candidate_sampler(true_classes=true_classes, num_sampled=32,
                                                                  unigrams=self.per_pixel_count[i].tolist(),
                                                                  num_true=self.num_codes, unique=False,
                                                                  range_max=self.num_codes)[0]))
        print("idx inp shape", self.idx_inp.shape)
        one_hot = tf.one_hot(self.idx_inp, depth=self.num_codes)
        print("idx one hot shape", one_hot.shape)
        self.latent_input_vals = tf.reduce_sum(
                    tf.expand_dims(one_hot, -1) *
                                tf.reshape(self.latent_space, [1, 1, self.num_codes, self.code_size]),
                                            axis=2)

        self.decoded_sampled = self._build_decoder_network(self.latent_input_vals, reuse=True)
        self.num_generated_samples = tf.placeholder(dtype=tf.int32)
        self.decoded_sampled = self.decoded_sampled[:self.num_generated_samples]
        self.decoded_sampled *= 255.0
        print("\t\tComputing per pixel distribution took: {:>6.6}".format(time.time() - ping))


    # @profile
    def sample(self, num_samples, save_images=False, compute_summaries=False, debug_mode=False):
        """
        imaginary out from used latent vectors with per pixel distributions
        """
        if self.per_pixel_count is None:
            if tf.train.latest_checkpoint(self.MODEL_PATH):
                print("reading data")
                iterator, iterable = self.load_tfrecords(batch_size=self.LOG_DATA_SIZE, repeat=False)
                print("restoring network")
                self.restore(iterable)
                print("restored from", tf.train.latest_checkpoint(self.MODEL_PATH))
                print("step:", self.global_step.eval(session=self.sess))
            else:
                raise RuntimeError("No checkpoints to restore from")
            self.compute_per_pixel_distribution(iterator, iterable, debug_mode=debug_mode)

        # latent_input_vals = self._sample_latent_from_per_pixel_distribution(num_samples)

        ret = self.sess.run(self.decoded_sampled, feed_dict={self.num_generated_samples: num_samples})
        return ret, None

        latent_input_vals = self.sess.run(self.latent_input_vals)
        if compute_summaries:
            out, summary_tr = self.sess.run([self.decoded, self.decoder_out_summary],
                                            feed_dict={self.dec_input: latent_input_vals})
        else:
            out = self.decoded.eval(session=self.sess, feed_dict={self.dec_input: latent_input_vals})
            summary_tr = None

        if save_images is True:
            for i in range(out.shape[0]):
                for j in range(5):
                    image = np.squeeze(out[i, :, :, self.channels*j:self.channels*(j+1)]) #* 255
                    # image = image.astype(np.uint8)
                    imageio.imwrite(config.IM_SAVE + self.MODEL_NAME + "_generated_{}_{}.png".format(i, j), image)
        return out[:num_samples], summary_tr


def run_vqvae(model_name, graph):
    with graph.as_default():
        if len(sys.argv) > 1:
            if sys.argv[1] == "sample":
                if len(sys.argv) > 2:
                    vqvae = VQVAE(data_directory=sys.argv[2], model_name=model_name, graph=graph)
                else:
                    vqvae = VQVAE(data_directory="../data/res-2-vq-vae/{}/".format(config.GAME_NAME),
                                  model_name=model_name, graph=graph)
                vqvae.sample(num_samples=2, save_images=True)
            elif sys.argv[1] == "for_action":
                vqvae = VQVAE(data_directory="../data/res-2-vq-vae/{}/".format(config.GAME_NAME), model_name=model_name,
                              graph=graph)
                vqvae.train(train_data_file=config.INPUT_PATH, for_action=True)
            else:
                raise RuntimeError("404")
        else:
            vqvae = VQVAE(data_directory="../data/res-2-vq-vae/{}/".format(config.GAME_NAME), model_name=model_name,
                          graph=graph)
            vqvae.train(train_data_file=config.INPUT_PATH, for_action=False)


if __name__ == "__main__":
    for i in range(3):
        new_graph = tf.Graph()
        run_vqvae("vq-vae-action-{}".format(i), new_graph)
