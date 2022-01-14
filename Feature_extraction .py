import cv2

import pickle
from tkinter import Tk
from tkinter.filedialog import askopenfilename
import time
import numpy as np
import csv
import scipy.io as sio
from keras.models import load_model
from keras.models import Model
from keras.preprocessing import image
from keras.applications import VGG19, resnet50, InceptionV3,mobilenet
from keras.applications.vgg19 import preprocess_input
import h5py
from keras.layers import Dense, Flatten
import os

model = resnet50.ResNet50(weights='imagenet' , include_top=True)
dataset_directory = "Dataset/test"
dataset_folder = os.listdir(dataset_directory)

DatabaseFeautres = []
DatabaseLabel = []

for dir_counter in range(0,len(dataset_folder)):
    single_class_dir = dataset_directory + "/" + dataset_folder[dir_counter]
    all_videos_one_class = os.listdir(single_class_dir)
    #print ('***********Processing', dataset_folder[dir_counter])
    
    for single_video_name in all_videos_one_class:
        video_path = single_class_dir + "/" + single_video_name
        # print ('Feature extracting: ', video_path)
        capture = cv2.VideoCapture(video_path)
        total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        video_features = []
        


        frames_counter = -1
        
        while(frames_counter < total_frames-1):

            frames_counter = frames_counter + 1
            ret, frame = capture.read()
            if (ret):
                frame = cv2.resize(frame, (224,224))
                img_data = image.img_to_array(frame)
                img_data = np.expand_dims(img_data, axis=0)
                img_data = preprocess_input(img_data)
                single_featurevector = model.predict(img_data)
                video_features.append(single_featurevector)
                #print ('Shape = ' , single_featurevector.shape, " max = ", max(single_featurevector))
                print(frames_counter%30)
                if frames_counter%30 == 29:
                    temp = np.asarray(video_features)
                    DatabaseFeautres.append(temp)
                    DatabaseLabel.append(dataset_folder[dir_counter])
                    print('extracted features len => ',len(DatabaseFeautres))
                    
                    video_features = []


            #cv2.imshow('v', frame)
            #cv2.waitKey(2)

        #print (single_video + "\n")

TotalFeatures= []
OneHotArray = []
for sample in DatabaseFeautres:
    TotalFeatures.append(sample.reshape([1,30000]))


TotalFeatures = np.asarray(TotalFeatures)
TotalFeatures = TotalFeatures.reshape([len(DatabaseFeautres),30000])

OneHotArray = []
kk=1;
for i in range(len(DatabaseFeautres)-1):
    OneHotArray.append(kk)
    if (DatabaseLabel[i] != DatabaseLabel[i+1]):
        kk=kk+1;

with open("OneHotArray.pickle", 'wb') as f:
  pickle.dump(OneHotArray, f)
    
OneHot=  np.zeros([len(DatabaseFeautres),2], dtype='int');


for i in range(len(DatabaseFeautres)-1):
    print(i)
    OneHot[i,OneHotArray[i]-1] = 1

#with open("OneHot.pickle", 'wb') as f:
#    pickle.dump(OneHot, f)

np.save('UCF_Test_full_resNET-50_features',TotalFeatures)
sio.savemat('UCF_Test_full_resnet_label.mat', mdict={'DatabaseLabel': OneHot})
#sio.savemat('train_features.mat', mdict={'TotalFeatures': TotalFeatures},appendmat=True, format='5',
#   long_field_names=False, do_compression=True, oned_as='row')


#import h5py
#import hdf5storage
#r,c = UCF_features.shape
#part1_ucf = UCF_features[0:int(r/6),0:int(c)]
#part2_ucf = UCF_features[int(r/6)+1:int(2*r/6),0:int(c)]
#part3_ucf = UCF_features[int(2*r/6)+1:int(3*r/6),0:int(c)]
#part4_ucf = UCF_features[int(3*r/6)+1:int(4*r/6),0:int(c)]
#part5_ucf = UCF_features[int(4*r/6)+1:int(5*r/6),0:int(c)]
#part6_ucf = UCF_features[int(5*r/6)+1:int(6*r/6),0:int(c)]
##part4_ucf = UCF_features[int(r/1):int(r),0:int(c)]
#sio.savemat('C:\\Users\\Imlab\\Desktop\\Part1_UCF.mat', mdict={'Part1Features':part1_ucf})
#sio.savemat('C:\\Users\\Imlab\\Desktop\\Part2_UCF.mat', mdict={'Part2Features':part2_ucf})
#sio.savemat('C:\\Users\\Imlab\\Desktop\\Part3_UCF.mat', mdict={'Part3Features':part3_ucf})
#sio.savemat('C:\\Users\\Imlab\\Desktop\\Part4_UCF.mat', mdict={'Part4Features':part4_ucf})
#sio.savemat('C:\\Users\\Imlab\\Desktop\\Part5_UCF.mat', mdict={'Part5Features':part5_ucf})
#sio.savemat('C:\\Users\\Imlab\\Desktop\\Part6_UCF.mat', mdict={'Part6Features':part6_ucf})
#
#hdf5storage.savemat('C:\\Users\\Imlab\\Desktop\\Part1_UCF.mat',mdict={'part1_ucf': part1_ucf}, appendmat=True, format='7.3',tore_python_metadata=True, action_for_matlab_incompatible='error', marshaller_collection=None, truncate_existing=False, truncate_invalid_matlab=False)
#
#hdf5storage.write(part2_ucf, '.', 'UCF_features.mat', matlab_compatible=True)





