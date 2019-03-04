import tensorflow as tf
import numpy as np
import os

import generative_model.config as config


class AbstractAutoEncoder:

    def __init__(self, model_path="./models/", log_path="./log/"):
        super().__init__()
        self.MODEL_PATH = model_path
        self.MODEL_SAVE_INTERVAL = config.MODEL_SAVE_INTERVAL       # save the model every # epochs
        self.LOG_PATH = log_path
        self.LOG_WRITE_INTERVAL = config.LOG_WRITE_INTERVAL       # write a log every # epochs for vqvae
        self.PRINT_TIME = config.PRINT_TIME
        self.SAVE_IMAGES_SUMMARY = config.SAVE_IMAGES_SUMMARY
        self.LOG_DATA_SIZE = config.LOG_DATA_SIZE

        self.INPUT_PATH = config.INPUT_PATH

        self.learning_rate = config.LEARNING_RATE
        self.EMBEDDING_SIZE = config.EMBEDDING_SIZE
        self.num_epochs = config.NUM_EPOCHS
        self.BATCH_SIZE = config.BATCH_SIZE
        self.global_step = None

        self.RGB = config.RGB
        self.five_frames = False
        if self.RGB:
            self.HEIGHT = 204
            self.WIDTH = 156
            self.channels = 3
        else:
            self.HEIGHT = 84
            self.WIDTH = 84
            self.channels = 1
        self.IMG_FLAT = self.HEIGHT*self.WIDTH
        self.TRAIN_SIZE = config.TRAIN_SIZE        # TODO pay attention to this

        self.MODEL_NAME = "MODEL"
        self.sess_config = tf.ConfigProto()
        self.sess_config.gpu_options.allow_growth = True

        self.sess = None

    def load_data(self, file_name):
        """
        :return: a numpy array of shape (-1, 84, 84, 4)
        """
        data = np.load(file_name)['screens']

        stacked_input = np.zeros((self.TRAIN_SIZE - 3, self.HEIGHT, self.WIDTH, 4))
        for i in range(stacked_input.shape[0]):
            stacked_input[i, :, :, 0] = data[i]
            stacked_input[i, :, :, 1] = data[i + 1]
            stacked_input[i, :, :, 2] = data[i + 2]
            stacked_input[i, :, :, 3] = data[i + 3]

        np.random.shuffle(stacked_input)
        scaled_input = stacked_input / np.max(stacked_input)

        # do something to shuffle and stuff
        return scaled_input

    def dataset_parser(self, example_proto):
        """
            Parse example_proto

            Returns:
                - frames | np.array([84, 84, 9]) | set of 9 consecutive ALE frames
                - actions | np.array([9]) | actions taken over 9 frames
                - rewards | np.array([9]) | rewards taken over 9 frames
                - terminals | np.array([9]) | terminal status of each frame in frames
        """
        feature_dict = {
            'frames': tf.FixedLenFeature([], tf.string),
            'actions': tf.FixedLenFeature([], tf.string),
            'rewards': tf.FixedLenFeature([], tf.string),
            'terminals': tf.FixedLenFeature([], tf.string)
        }

        parsed_features = tf.parse_single_example(example_proto, feature_dict)

        frames = tf.decode_raw(parsed_features['frames'], tf.uint8)
        # actions = tf.decode_raw(parsed_features['actions'], tf.uint8)
        # rewards = tf.decode_raw(parsed_features['rewards'], tf.uint8)
        # terminals = tf.decode_raw(parsed_features['terminals'], tf.uint8)

        frame_shape = tf.stack([84, 84, 9])
        # array_shape = tf.stack([9])

        frames = tf.reshape(frames, frame_shape)
        # actions = tf.reshape(actions, array_shape)
        # rewards = tf.reshape(rewards, array_shape)
        # terminals = tf.reshape(terminals, array_shape)

        frames = frames[:, :, :4]
        # actions = actions[:4]
        # rewards = rewards[:4]
        # terminals = terminals[:4]

        frames = tf.cast(frames, tf.float32)
        frames /= 255.0

        return frames  # , actions, rewards, terminals

    def dataset_parser_rgb(self, example_proto):
        """
            Parse example_proto

            Returns:
                - frames | np.array([84, 84, 9]) | set of 9 consecutive ALE frames
                - actions | np.array([9]) | actions taken over 9 frames
                - rewards | np.array([9]) | rewards taken over 9 frames
                - terminals | np.array([9]) | terminal status of each frame in frames
        """
        feature_dict = {
            'frames': tf.FixedLenFeature([], tf.string),
            'actions': tf.FixedLenFeature([], tf.string),
            'rewards': tf.FixedLenFeature([], tf.string),
            'terminals': tf.FixedLenFeature([], tf.string)
        }

        parsed_features = tf.parse_single_example(example_proto, feature_dict)

        frames = tf.decode_raw(parsed_features['frames'], tf.uint8)
        # actions = tf.decode_raw(parsed_features['actions'], tf.uint8)
        # rewards = tf.decode_raw(parsed_features['rewards'], tf.uint8)
        # terminals = tf.decode_raw(parsed_features['terminals'], tf.uint8)

        frame_shape = tf.stack([210, 160, 3 * 9])
        # array_shape = tf.stack([9])

        frames = tf.reshape(frames, frame_shape)
        frames = frames[:self.HEIGHT, :self.WIDTH, :3*4]
        # actions = tf.reshape(actions, array_shape)
        # rewards = tf.reshape(rewards, array_shape)
        # terminals = tf.reshape(terminals, array_shape)

        frames = tf.cast(frames, tf.float32)
        frames /= 255.0

        return frames  # , actions, rewards, terminals

    def dataset_parser_5frames(self, example_proto):
        """
            Parse example_proto

            Returns:
                - frames | np.array([84, 84, 9]) | set of 9 consecutive ALE frames
                - actions | np.array([9]) | actions taken over 9 frames
                - rewards | np.array([9]) | rewards taken over 9 frames
                - terminals | np.array([9]) | terminal status of each frame in frames
        """
        feature_dict = {
            'frames': tf.FixedLenFeature([], tf.string),
            'action': tf.FixedLenFeature([], tf.string),
            'reward': tf.FixedLenFeature([], tf.string),
            'terminal': tf.FixedLenFeature([], tf.string)
        }

        parsed_features = tf.parse_single_example(example_proto, feature_dict)

        frames = tf.decode_raw(parsed_features['frames'], tf.uint8)
        action = tf.decode_raw(parsed_features['action'], tf.int32)
        reward = tf.decode_raw(parsed_features['reward'], tf.int32)
        terminal = tf.decode_raw(parsed_features['terminal'], tf.int32)

        frame_shape = tf.stack([84, 84, 5])

        frames = tf.reshape(frames, frame_shape)

        frames = tf.cast(frames, tf.float32)
        frames /= 255.0

        return frames, action, reward, terminal

    def load_tfrecords(self, batch_size, repeat=False):
        dataset = tf.data.TFRecordDataset(self.INPUT_PATH)
        # dataset = dataset.shuffle(buffer_size=config.SHUFFLE_BUFFER_SIZE)
        if repeat:
            dataset = dataset.repeat()
        if self.RGB:
            dataset = dataset.map(self.dataset_parser_rgb, num_parallel_calls=config.NUM_CPUS)
        elif self.five_frames:
            dataset = dataset.map(self.dataset_parser_5frames, num_parallel_calls=config.NUM_CPUS)
        else:
            dataset = dataset.map(self.dataset_parser, num_parallel_calls=config.NUM_CPUS)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(buffer_size=config.PREFETCH_BUFFER_SIZE)
        # Instantiate iterators

        iterator = dataset.make_initializable_iterator()
        iterable = iterator.get_next()
        return iterator, iterable

    def _build_network(self, inp):
        return []

    def train(self, train_data_file):
        pass

    def sample(self, *args, **kwargs):
        pass

    def save(self, sess, global_step):
        saver = tf.train.Saver(save_relative_paths=True)
        tf.gfile.MakeDirs(self.MODEL_PATH)
        _ = saver.save(sess, self.MODEL_PATH, global_step=global_step)

    def restore(self, inp, scope=None, sess=None):
        if scope:
            self.scope = scope
            with tf.variable_scope(scope):
                self._build_network(inp)
        else:
            self._build_network(inp)

        if not sess:
            sess = tf.Session(config=self.sess_config)

        if scope:
            # get rid of scope name in restoring
            var_dict = {}
            for var in tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=scope):
                key = var.name.replace("{}/".format(scope), "")[:-2]  # gets rid of "agent"s as well as :0 at the end
                var_dict[key] = var
            saver = tf.train.Saver(var_list=var_dict)
        else:
            saver = tf.train.Saver()
        _ = saver.restore(sess, tf.train.latest_checkpoint(self.MODEL_PATH))

        self.sess = sess

