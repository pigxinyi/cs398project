import numpy as np
import os
import sys
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
#from torchvision import datasets, transforms
from torch.autograd import Variable
import torch.distributed as dist
import torchvision

# from helperFunctions import getUCF101
# from helperFunctions import loadFrame

import h5py
import cv2

import os
import numpy as np
import cv2
import time
import h5py
import imageio

def getUCF101(base_directory = ''):

    # action class labels
    class_file = open(base_directory + 'ucfTrainTestlist/classInd.txt','r')
    lines = class_file.readlines()
    lines = [line.split(' ')[1].strip() for line in lines]
    class_file.close()
    class_list = np.asarray(lines)

    # training data
    train_file = open(base_directory + 'ucfTrainTestlist/trainlist01.txt','r')
    lines = train_file.readlines()
    filenames = ['UCF-101/' + line.split(' ')[0] for line in lines]
    y_train = [int(line.split(' ')[1].strip())-1 for line in lines]
    y_train = np.asarray(y_train)
    filenames = [base_directory + filename for filename in filenames]
    train_file.close()

    train = (np.asarray(filenames),y_train)

    # testing data
    test_file = open(base_directory + 'ucfTrainTestlist/testlist01.txt','r')
    lines = test_file.readlines()
    filenames = ['UCF-101/' + line.split(' ')[0].strip() for line in lines]
    classnames = [filename.split('/')[1] for filename in filenames]
    y_test = [np.where(classname == class_list)[0][0] for classname in classnames]
    y_test = np.asarray(y_test)
    filenames = [base_directory + filename for filename in filenames]
    test_file.close()

    test = (np.asarray(filenames),y_test)

    return class_list, train, test

def loadFrame(args):
    mean = np.asarray([0.485, 0.456, 0.406],np.float32)
    std = np.asarray([0.229, 0.224, 0.225],np.float32)

    curr_w = 320
    curr_h = 240
    height = width = 224
    (filename,augment) = args

    data = np.zeros((3,height,width),dtype=np.float32)

    try:
        ### load file from HDF5
        # filename = filename.replace('.avi','.hdf5')
        # filename = filename.replace('UCF-101','UCF-101-hdf5')
        # h = h5py.File(filename,'r')
        h=imageio. mimread(filename)
        nFrames = len(h['meta']) - 1
        frame_index = np.random.randint(nFrames)
        frame = h['meta'][frame_index]

        if(augment==True):
            ## RANDOM CROP - crop 70-100% of original size
            ## don't maintain aspect ratio
            if(np.random.randint(2)==0):
                resize_factor_w = 0.3*np.random.rand()+0.7
                resize_factor_h = 0.3*np.random.rand()+0.7
                w1 = int(curr_w*resize_factor_w)
                h1 = int(curr_h*resize_factor_h)
                w = np.random.randint(curr_w-w1)
                h = np.random.randint(curr_h-h1)
                frame = frame[h:(h+h1),w:(w+w1)]
            
            ## FLIP
            if(np.random.randint(2)==0):
                frame = cv2.flip(frame,1)

            frame = cv2.resize(frame,(width,height))
            frame = frame.astype(np.float32)

            ## Brightness +/- 15
            brightness = 30
            random_add = np.random.randint(brightness+1) - brightness/2.0
            frame += random_add
            frame[frame>255] = 255.0
            frame[frame<0] = 0.0

        else:
            # don't augment
            frame = cv2.resize(frame,(width,height))
            frame = frame.astype(np.float32)

        ## resnet model was trained on images with mean subtracted
        frame = frame/255.0
        frame = (frame - mean)/std
        frame = frame.transpose(2,0,1)
        data[:,:,:] = frame
    except:
        print("Exception: " + filename)
        data = np.array([])
    return data

data_directory = '~/scratch/dataset/'
class_list, train, test = getUCF101(base_directory = data_directory)
a=loadFrame(train[0][1],True)
print(a)
