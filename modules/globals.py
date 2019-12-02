PROJECT_NAME = "Akshar"

# Enable this while developing the programs and model
DEV = False

# Hyperparameters
NUM_EPOCHS = 30
NUM_CLASSES = 8
BATCH_SIZE = 64
LEARNING_RATE = 0.01
STRIDE = 1
KERNEL_SIZE = 3
DROPOUT_PROB = 0.5
TRAIN_TEST_SPLIT = 0.2

# Parameter choices for experiments
BATCH_SIZE_LIST = [32, 64, 128]
LEARNING_RATE_LIST = [0.01, 0.001, 0.0001]
KERNEL_SIZE_LIST = [3, 5]
DROPOUT_PROB_LIST = [0.3, 0.5, 0.7]

# Paths
TRAINING_DATA_PATH = "./data/raw/train_data.npy"
TRAINING_LABELS_PATH = "./data/raw/train_labels.npy"
DATA_STORE_PATH = "./data/"
MODEL_STORE_PATH = './models/'
METRICS_SAVE_PATH = './metrics/'
RESULTS_SAVE_PATH = './results/'

# Labels Map
MAP_LABELS = ['unknown', 'a', 'b','c','d','h', 'i', 'j', 'k']
MAP_CLASSES = [-1, 1, 2, 3, 4, 5, 6, 7, 8]
UNKNOWN_CLASS_THRESHOLD = -0.005