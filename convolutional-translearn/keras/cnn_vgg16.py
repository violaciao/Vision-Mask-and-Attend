import os
from config import *
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

# Upgrade Reference: 
# https://stackoverflow.com/questions/41823068/train-multi-class-image-classifier-in-keras

# print(K.tensorflow_backend._get_available_gpus())
print 'Model: {}, Batch_size: {}, saved_model: {}'.format(MODEL_FT, BATCH_SIZE, MODEL_SAVING_PATH)

# dimensions of our images.
img_width, img_height = 224, 224

data_dir = DATA_DIR
train_data_dir = os.path.join(data_dir, 'train')
validation_data_dir = os.path.join(data_dir, 'val')

nb_train_samples = sum([len(os.path.join(train_data_dir, i)) for i in os.listdir(train_data_dir)])
nb_validation_samples = sum([len(os.path.join(validation_data_dir, i)) for i in os.listdir(validation_data_dir)])

epochs = EPOCHS
batch_size = BATCH_SIZE

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(NUM_CLASSES))
model.add(Activation('softmax'))

model.compile(loss='categorical_crossentropy',
              # optimizer='rmsprop',
              optimizer=optimizers.SGD(lr=1e-4, momentum=0.9),
              metrics=['accuracy'])

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical',
    shuffle=True)

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

if not os.path.exists('saved_models'):
    os.makedirs('saved_models')

# model.save_weights('first_try.h5')
model.save(MODEL_SAVING_PATH)




