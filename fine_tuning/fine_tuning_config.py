# Learning rate parameters
BASE_LR = 0.001
EPOCH_DECAY = 30 # number of epochs after which the Learning rate is decayed exponentially.
DECAY_WEIGHT = 0.1 # factor by which the learning rate is reduced.


# DATASET INFO
NUM_CLASSES = 2 # set the number of classes in dataset
DATA_DIR = '/Users/Viola/CDS/Rearch/Langone/Vision-Mask-and-Attend/data'

# DATALOADER PROPERTIES
BATCH_SIZE = 10 # Set as high as possible if there are not out of memory error.


### GPU SETTINGS
CUDA_DEVICE = 0 # Enter device ID of the gpu if to run on gpu. Otherwise neglect.
GPU_MODE = 0 # set to 1 if run on gpu.


# SETTINGS FOR DISPLAYING ON TENSORBOARD
USE_TENSORBOARD = 1 # if want to use tensorboard set this to 1.
TENSORBOARD_LOGDIR = "ft_log" # directory of tensorboard log files
FLUSH_SECS = 30 # flush interval - number of seconds