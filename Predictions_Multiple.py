# Python3 code to get the predictions
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
from moviepy.video.io.ffmpeg_tools import ffmpeg_extract_subclip
from moviepy.editor import *
from tensorflow.keras.callbacks import ModelCheckpoint

#Predictions for model trained with Batch size=10,12,18 and number of filetrs=8,16,32,64
for i in [10,12,18]:
    for j in [8,16,32,64]:
        # Provide the path for the weights obtained during training
        keras_model_path='1BS'+str(i)+'FE'+str(j)
        print("BATCH SIZE = {}     FEATURES ={} ".format(i,j))
        # Provide the path for the folder containing test videos
        path="/home/ccps/violence_detection/20-06-2022/Testing videos"
        l= os.listdir(path)
        l.sort()
        for vid in l:
          if(vid[-4:]=='.mp4'):
            print(vid)
            vid=path+"/"+vid 
            restore_keras_model = tf.keras.models.load_model(keras_model_path)
            img_rows,img_cols,img_depth=112,112,300
            X_tr=[]
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
            # Code to make the video 10 sec if it ids less than 10 sec
            input=np.array(frames)
            shape = np.shape(input)
            # print(shape)
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
            X_tr_array = np.array(X_tr)
            train_set = np.zeros((1, img_rows, img_cols, img_depth ,1))
            train_set[0,:,:,:,0]=X_tr_array[0,:,:,:]
            #print(train_set.shape)
            new_depth= img_depth//5
            train_set1 = np.zeros((1, img_rows, img_cols, new_depth,1))
            # print(train_set1.shape)
            for h in range(60):
                train_set1[:,:,:,h,:] =train_set[:,:,:,5*h,:]
            # print(train_set1.shape)
            # Code to make the prediction
            result=restore_keras_model.predict(train_set1)[0]
            print(result)
            scores = [1 - result[0], result[0]]
            class_names = ["normal", "abnormal"]
            if vid1[0:2]=="NV" and result[0]<0.5:
                nv_count+=1
            if vid1[0:1]=="V" and result[0]>0.5:
                v_count+=1
            if vid1[0:7]=="NV_test" and result[0]<0.5:
                nvt_count+=1
            if vid1[0:6]=="V_test" and result[0]>0.5:
                vt_count+=1
        print("BATCH SIZE = {}     FILTERS ={} ".format(i,j))
        print("****************************************")
        print("CORRECT PREDICTION")
        print("Non Violence = {}/8".format(nv_count))
        print("Non Violence Drone = {}/4".format(nvt_count))
        print("Violence = {}/12".format(v_count))
        print("Violence Drone = {}/4".format(vt_count))
        print("#########################################")

        file1 = open("predictions.txt","a")
        l=["BATCH SIZE = "+str(i)+"     FILTERS ="+str(j)+"\n","CORRECT PREDICTION\n","Non Violence = "+str(nv_count)+"/8\n","Non Violence Drone = "+str(nvt_count)+"/4\n","Violence = "+str(v_count)+"/12\n","Violence Drone = "+str(vt_count)+"/4\n","#########################################\n"]
        file1.writelines(l)
        file1.close()


