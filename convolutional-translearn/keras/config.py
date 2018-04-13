# DATASET INFO
NUM_CLASSES = 5 # set the number of classes in dataset
DATA_DIR = '/scratch/xc965/DL/TransLearn/data/data_flower5'

# DATALOADER PROPERTIES
BATCH_SIZE = 10 # Set as high as possible if there are not out of memory error.

# NUMBER OF TRAINING EPOCHS
EPOCHS = 100

# MODEL FOR TRANSLEARNING
MODEL_FT = 'vgg16'

# MODEL SAVING PATH
MODEL_SAVING_PATH = 'saved_models/model_fl5_vgg.h5'

# Occlusion Size for Visualization
OCC_SIZE = 10