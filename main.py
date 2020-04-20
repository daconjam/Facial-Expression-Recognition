'''
Facial Expression Recognition
author: Jamell Dacon (daconjam@msu.edu)
'''

import math
import numpy as np
import pandas as pd

from skimage.transform import resize
from keras.layers import Dense, Flatten, Conv2D, Dropout, MaxPooling2D,MaxPooling1D
from keras.layers import Activation,Dropout,Flatten,BatchNormalization
from keras.models import Sequential
from keras.utils import np_utils
import tensorflow as tf
import keras
from keras import backend as K
K.image_data_format()
from keras import applications
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, MaxPooling2D
from keras.optimizers import Adadelta
from keras.utils import np_utils
from keras.regularizers import l2

import csv
import scipy.misc
import scipy
from scipy import ndimage
from keras.preprocessing.image import ImageDataGenerator

print("Importing the csv file")

df = pd.read_csv('fer2013.csv')

X = df.iloc[:, 1].values
y = df.iloc[:, 0].values

img_height = 48
img_width = 48
images = np.empty((len(X), img_height, img_width, 3))
i=0
for pixel_string in X:
  temp = [float(pixel) for pixel in pixel_string.split(' ')]
  temp = np.asarray(temp).reshape(img_height, img_width)
  temp = resize(temp, (img_height, img_width), order=3, mode='constant')

  channel = np.empty((img_height, img_width, 3))
  channel[:, :, 0] = temp
  channel[:, :, 1] = temp
  channel[:, :, 2] = temp

  images[i, :, :, :] = channel
  i = i + 1

images /= 255.0
labels = keras.utils.to_categorical(y, 7)

crossvalidation_set = images[32096:,:,:,:]
images = images[0:28709,:,:,:]

cross_label = labels[32096::,:]
labels = labels[0:28709,:]


datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(
        rescale = 1./1)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(images)

# fits the model on batches with real-time data augmentation:
training_datagen = datagen.flow(images, labels, batch_size=32)

validation_datagen = test_datagen.flow(crossvalidation_set,cross_label,batch_size = 32)

input_shape = (48, 48, 3)
batch_size = 64
epochs = 100
verbose = 1

model = Sequential()
model.add(Conv2D(32,kernel_size=(3, 3),activation='relu',input_shape=input_shape))

model.add(Dropout(0.35))
model.add(Conv2D(64,(3, 3),activation='relu'))
model.add(Dropout(0.25))
model.add(MaxPooling2D(pool_size=(2, 2),padding = 'same'))

model.add(Conv2D(
    128,
    (3, 3),
    activation='relu'
))

    256,
    (3, 3),
    activation='relu'
))

model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(2048, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.35))
model.add(Dense(1024, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.35))
model.add(Dense(512, activation='relu'))
model.add(BatchNormalization())
model.add(Dropout(0.35))
model.add(Dense(256, activation='relu'))
model.add(BatchNormalization())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.35))
model.add(Dense(7, activation='softmax'))

model.compile(
    optimizer='adam',
    metrics=['accuracy'],
    loss='categorical_crossentropy'
)

model.summary()

datagen = ImageDataGenerator(
    featurewise_center=True,
    featurewise_std_normalization=True,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True)

test_datagen = ImageDataGenerator(
        rescale = 1./1)

# compute quantities required for featurewise normalization
# (std, mean, and principal components if ZCA whitening is applied)
datagen.fit(images)

# fits the model on batches with real-time data augmentation:
training_datagen = datagen.flow(images, labels, batch_size=32)

validation_datagen = test_datagen.flow(crossvalidation_set,cross_label,batch_size = 32)

history = model.fit(
    images, labels,
    validation_data = (crossvalidation_set,cross_label),
    batch_size=batch_size,
    verbose=verbose,
    epochs=epochs
)
