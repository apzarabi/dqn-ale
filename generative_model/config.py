import socket

hostname = socket.gethostname()

# ##########  LOCAL SETTINGS   ###########
if not hostname.startswith('eureka'):
    INPUT_DIR = "../../atari_inputs/"
    NUM_EPOCHS = 2
    MODEL_SAVE_INTERVAL = 1
    LOG_WRITE_INTERVAL = 1
    SAVE_IMAGES_SUMMARY = 1
    TRAIN_SIZE = 100
    SHUFFLE_BUFFER_SIZE = 10
    PRINT_TIME = 1
    PREFETCH_BUFFER_SIZE = 100
    NUM_CPUS = 2

# ########## REMOTE SETTINGS  ###########
elif hostname.startswith('eureka'):
    INPUT_DIR = "../../atari_inputs/"
    NUM_EPOCHS = 100
    MODEL_SAVE_INTERVAL = 10
    LOG_WRITE_INTERVAL = 1000
    SAVE_IMAGES_SUMMARY = 2
    TRAIN_SIZE = 50000
    SHUFFLE_BUFFER_SIZE = 1000
    PRINT_TIME = 1000
    PREFETCH_BUFFER_SIZE = 4
    NUM_CPUS = 12


else:
    print("USER DOES NOT EXISTS")
    raise Exception("USER DOES NOT EXISTS")


GAME_TO_ACTION = {
    'freeway': 3,
    'seaquest': 18,
}


# ########## GLOBAL SETTINGS  ###########
LOG_DATA_SIZE = 32
BATCH_SIZE = 4
LEARNING_RATE = 2e-4
EMBEDDING_SIZE = 1024
RGB = False


GAME_NAME = "freeway"
NUM_ACTIONS = GAME_TO_ACTION[GAME_NAME]
# INPUT_FILE_NAME = "100k_5frames_stacked_min_action.tfrecords"
# INPUT_PATH = INPUT_DIR + GAME_NAME + "/" + INPUT_FILE_NAME
# For per-action models
INPUT_FILE_NAME = "100k_5frames_stacked_action_{}.tfrecords"    # for freeway
# INPUT_FILE_NAME = "500k_5frames_stacked_action_{}.tfrecords"  # for seaquest
INPUT_PATH = INPUT_DIR + GAME_NAME + "/" + INPUT_FILE_NAME
IM_SAVE = "images/" + GAME_NAME + "/"
