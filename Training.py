# python 3 code for Training the model
# Import the necessary modules
import tensorflow as tf
from tensorflow.keras.optimizers import RMSprop
from keras.utils import np_utils, generic_utils
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential, load_model
from tensorflow.python.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.python.keras.layers.convolutional import Convolution3D, MaxPooling3D
import os
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import cv2
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_validate
from sklearn import preprocessing
from tensorflow.keras.callbacks import ModelCheckpoint
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *

# CNN training parameters
img_rows,img_cols,img_depth=112,112,300
patch_size=300
batch_size = 18
nb_classes = 2
nb_epoch = 50
num_samples=4363
# Load the Features stored in .npy files
from numpy import load
Y_train = load('Y_train_new.npy')
train_set= load('train_set_new.npy')

# Consider every 5th frame 
new_depth= img_depth//5
train_set1 = np.zeros((num_samples, img_rows, img_cols, new_depth,1))
print(train_set1.shape)
for h in range(60):
    train_set1[:,:,:,h,:] =train_set[:,:,:,5*h,:]
print(train_set1.shape)

# Using multiple GPU for training
strategy = tf.distribute.MirroredStrategy()
print('Number of devices: {}'.format(strategy.num_replicas_in_sync))
with strategy.scope():
    # Define model
    model = Sequential()
    print('input shape', img_rows, 'rows', img_cols, 'cols', patch_size, 'patchsize')
    model.add(Convolution3D(8,kernel_size=(3,3,3),strides=(1, 1, 1),padding="same", input_shape=(img_rows, img_cols, new_depth,1), activation='relu'))
    model.add(MaxPooling3D(pool_size=(1,2,2),strides=(2, 2, 2)))
    # model.add(Dropout(0.5))
    model.add(Convolution3D(16, kernel_size=(3, 3, 3),strides=(1, 1, 1),padding="same", activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2),strides=(2, 2, 2)))
    # model.add(Dropout(0.5))
    model.add(Convolution3D(32, kernel_size=(3, 3, 3),strides=(1, 1, 1),padding="same", activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2),strides=(2, 2, 2)))
    # model.add(Dropout(0.5))
    model.add(Convolution3D(64,kernel_size=(3, 3, 3),strides=(1, 1, 1),padding="same", activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2),strides=(2, 2, 2)))
    # model.add(Dropout(0.5))
    model.add(Convolution3D(64, kernel_size=(3, 3, 3),strides=(1, 1, 1),padding="same", activation='relu', kernel_initializer='he_uniform'))
    model.add(MaxPooling3D(pool_size=(2, 2, 2),strides=(2, 2, 2)))
    # model.add(Dropout(0.5))
    model.add(Flatten())
    model.add(Dense(512, activation='relu'))
    model.add(Dense(512, activation='relu'))
    # model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='RMSprop', metrics=['mse', 'accuracy'])

# Split the dataset for Training and Validation
X_train_new, X_val_new, y_train_new,y_val_new = train_test_split(train_set1, Y_train, test_size=0.2, random_state=4)

print(X_train_new.shape)
print(y_train_new.shape)
print(X_val_new.shape)
print(y_val_new.shape)

# model.summary()

# Train the model
hist = model.fit(
    X_train_new,
    y_train_new,
    validation_data=(X_val_new,y_val_new),
    batch_size=batch_size,
    epochs =nb_epoch,
    shuffle=True
    )

# Evaluate the model
score = model.evaluate(
   X_val_new,
   y_val_new,
   batch_size=batch_size,
   #show_accuracy=True
   )

# Results
train_loss=hist.history['loss']
val_loss=hist.history['val_loss']
train_acc=hist.history['accuracy']
val_acc=hist.history['val_accuracy']
print('**********************************************')
print('Test score:', score)
print('History', hist.history)
print('train_loss', train_loss)
print('val_loss', val_loss)
print('train_acc', train_acc)
print('val_acc', val_acc)

# Save the weights of the trained model
model.save("filename")