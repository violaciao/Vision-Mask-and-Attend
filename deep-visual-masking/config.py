# Learning rate parameters
BASE_LR = 0.001
EPOCH_DECAY = 30 # number of epochs after which the Learning rate is decayed exponentially.
DECAY_WEIGHT = 0.1 # factor by which the learning rate is reduced.

# IS_PRETRAINED or not
IS_PRETRAINED = False

# DATASET INFO
NUM_CLASSES = 3 # set the number of classes in dataset
# DATA_DIR = '/Users/Viola/CDS/Rearch/Langone/Vision-Mask-and-Attend/data/data_pathology_small'
DATA_DIR = '/scratch/xc965/DL/TransLearn/data/data_pathology'

# DATALOADER PROPERTIES
BATCH_SIZE = 10 # Set as high as possible if there are not out of memory error.

# NUMBER OF TRAINING EPOCHS	
NUM_EPOCHS = 50

# MODEL FOR TRANSLEARNING
MODEL_FT = 'resnet18'

# MODEL SAVING PATH
MODEL_SAVING_PATH = 'saved_models/model_pth_resnet18_xavier.pt'
MODEL_CPU_SAVING_PATH = 'saved_models/model_cpu_pth_resnet18_xavier.pt'

### GPU SETTINGS
CUDA_DEVICE = 0 # Enter device ID of the gpu if to run on gpu. Otherwise neglect.
# GPU_MODE = 1 # set to 1 if run on gpu.


# SETTINGS FOR DISPLAYING ON TENSORBOARD
USE_TENSORBOARD = 0 # if want to use tensorboard set this to 1.
TENSORBOARD_LOGDIR = "logs" # directory of tensorboard log files
FLUSH_SECS = 30 # flush interval - number of seconds