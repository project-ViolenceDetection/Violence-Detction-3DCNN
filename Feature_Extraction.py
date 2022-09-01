# Python 3 code for Feature Extraction
# Import necessary modules
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
print(os.getcwd())
l= os.listdir(os.getcwd())
X_tr=[]

img_rows,img_cols,img_depth=112,112,300
patch_size=300
for vid in l:
  if(vid[-4:]=='.mp4'):
    print(vid)
    frames = []
    cap = cv2.VideoCapture(vid)
    fps = cap.get(5)
    #print ("Frames per second using video.get(cv2.cv.CV_CAP_PROP_FPS): {0}".format(fps))
    clip = VideoFileClip(vid)
    duration = clip.duration
    n_frames=300
    try:
     for k in range(n_frames):
        ret, frame = cap.read()
        frame=cv2.resize(frame,(img_rows,img_cols),interpolation=cv2.INTER_AREA)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        frames.append(gray)
        #print(frames)

        #plt.imshow(gray, cmap = plt.get_cmap('gray'))
        #plt.xticks([]), plt.yticks([])  # to hide tick values on X and Y axis
        #plt.show()
        #cv2.imshow('frame',gray)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    except:
        n_frames=k
    cap.release()
    cv2.destroyAllWindows()
# Code to make the videos less than 10sec to 10sec each
    input=np.array(frames)
    shape = np.shape(input)
    if(shape[0]>149):
     pad_input=np.zeros((300,112,112))
     pad_input[:shape[0],:,:]=input
     pad_input[shape[0]:,:,:]=input[:300-shape[0],:,:]

    elif(shape[0]>100):
     pad_input=np.zeros((300,112,112))
     pad_input[:shape[0],:,:]=input
     pad_input[shape[0]:2*shape[0],:,:]=input
     pad_input[2*shape[0]:,:,:]=input[:300-2*shape[0],:,:]

    elif(shape[0]>75):
     pad_input=np.zeros((300,112,112))
     pad_input[:shape[0],:,:]=input
     pad_input[shape[0]:2*shape[0],:,:]=input
     pad_input[2*shape[0]:3*shape[0],:,:]=input
     pad_input[3*shape[0]:,:,:]=input[:300-3*shape[0],:,:]

    elif(shape[0]>60):
     pad_input=np.zeros((300,112,112))
     pad_input[:shape[0],:,:]=input
     pad_input[shape[0]:2*shape[0],:,:]=input
     pad_input[2*shape[0]:3*shape[0],:,:]=input
     pad_input[3*shape[0]:4*shape[0],:,:]=input
     pad_input[4*shape[0]:,:,:]=input[:300-4*shape[0],:,:]

    else:
     pad_input=np.zeros((300,112,112))
     pad_input[:shape[0],:,:]=input
     pad_input[shape[0]:2*shape[0],:,:]=input
     pad_input[2*shape[0]:3*shape[0],:,:]=input
     pad_input[3*shape[0]:4*shape[0],:,:]=input
     pad_input[4*shape[0]:5*shape[0],:,:]=input
     pad_input[5*shape[0]:,:,:]=input[:300-5*shape[0],:,:]
    ipt=np.rollaxis(np.rollaxis(pad_input,2,0),2,0)
    X_tr.append(ipt)
X_tr_array = np.array(X_tr)   # convert the frames read into array
num_samples = len(X_tr_array)
print (num_samples)
# Assign Label to each class (label is a 1-D array of 0,0,0,0,0,0,0,1,1,1,1,1,1,2,2,2,2,2,2,2,3,3,3,3,3,3, etc)
label=np.ones((num_samples,), dtype = int)
label[0:3043] = 1
label[3043:] = 0

# train_data is num_samples columns x 2 rows
# train_data[0] = X_tr_array  and  train_data[1] = label
train_data = [X_tr_array, label]
# X_train is now X_tr_array
# y_train is now label
(X_train, y_train) = (train_data[0],train_data[1])
print('X_Train shape:', X_train.shape)
train_set = np.zeros((num_samples, img_rows, img_cols, img_depth,1))
for h in range(num_samples):
    train_set[h,:,:,:,0]=X_train[h,:,:,:]

print(train_set.shape, 'train samples')

# convert class vectors to binary class matrices
Y_train = np_utils.to_categorical(y_train, nb_classes)
# Clear the memory
del X_tr
del X_train
del X_tr_array
del train_data
# Convert the datatype to float
train_set = train_set.astype('float32')
# Perform Normalization
train_set -= np.mean(train_set)
train_set /= np.max(train_set)
# Save the Features to .npy files
from numpy import save
save('Y_train_new.npy', Y_train)
save('train_set_new.npy', train_set)

