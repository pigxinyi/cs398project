#training the 3D convolution model

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

from helperFunctions import getUCF101
from helperFunctions import loadSequence
import resnet_3d

import h5py
import cv2

from multiprocessing import Pool


IMAGE_SIZE = 224
NUM_CLASSES = 101
batch_size = 32
lr = 0.0001
num_of_epochs = 10

data_directory = '/projects/training/bawc/AR/'
class_list, train, test = getUCF101(base_directory = data_directory)

#importing loadSequence() this time as well as the model definition.
#we are defining the sample_duration=16. That is, we are going to be training on subsequences of 16 frames.

model =  resnet_3d.resnet50(sample_size=IMAGE_SIZE, sample_duration=16)

#The state dictionary for the pretrained model is loaded into pretrained.
pretrained = torch.load('/u/training/tra347/CS398/project/resnet-50-kinetics.pth')

#The names of the of layers are used to place the weights into the model we defined.
keys = [k for k,v in pretrained['state_dict'].items()]
pretrained_state_dict = {k[7:]: v.cpu() for k, v in pretrained['state_dict'].items()}
model.load_state_dict(pretrained_state_dict)

#Redefine the fully connected layer once again such that it has the appropriate number of outputs
model.fc = nn.Linear(model.fc.weight.shape[1],NUM_CLASSES)

#not have a large enough GPU to handle the blocks model.layer4[1] and model.layer4[2]
#The largest attainable batch size with these layers included is 1 which is not enough to train the model due to batch normalization.

#Instead, we will simply ignore these last two blocks and directly send the output of model.layer4[0] into the fully connected layer.

for param in model.parameters():
    param.requires_grad_(False)

#change the number of layers to be trained for UCF101
# for param in model.conv1.parameters():
#     param.requires_grad_(True)
# for param in model.bn1.parameters():
#     param.requires_grad_(True)
# for param in model.layer1.parameters():
#     param.requires_grad_(True)
# for param in model.layer2.parameters():
#     param.requires_grad_(True)
# for param in model.layer3.parameters():
#     param.requires_grad_(True)
for param in model.layer4[0].parameters():
    param.requires_grad_(True)
for param in model.fc.parameters():
    param.requires_grad_(True)

params = []
# for param in model.conv1.parameters():
#     params.append(param)
# for param in model.bn1.parameters():
#     params.append(param)
# for param in model.layer1.parameters():
#     params.append(param)
# for param in model.layer2.parameters():
#     params.append(param)
# for param in model.layer3.parameters():
#     params.append(param)
for param in model.layer4[0].parameters():
    params.append(param)
for param in model.fc.parameters():
    params.append(param)


model.cuda()

optimizer = optim.Adam(params,lr=lr)

criterion = nn.CrossEntropyLoss()

pool_threads = Pool(8,maxtasksperchild=200)



for epoch in range(0,num_of_epochs):

    ###### TRAIN
    train_accu = []
    model.train()
    random_indices = np.random.permutation(len(train[0]))
    start_time = time.time()

    for i in range(0, len(train[0])-batch_size,batch_size):

        augment = True
        video_list = [(train[0][k],augment)
                       for k in random_indices[i:(batch_size+i)]]
        data = pool_threads.map(loadSequence,video_list)

        next_batch = 0
        for video in data:
            if video.size==0: # there was an exception, skip this
                next_batch = 1
        if(next_batch==1):
            continue

        x = np.asarray(data,dtype=np.float32)
        x = Variable(torch.FloatTensor(x),requires_grad=False).cuda().contiguous()

        y = train[1][random_indices[i:(batch_size+i)]]
        y = torch.from_numpy(y).cuda()

        with torch.no_grad():
            h = model.conv1(x)
            h = model.bn1(h)
            h = model.relu(h)
            h = model.maxpool(h)

            h = model.layer1(h)
            h = model.layer2(h)
            h = model.layer3(h)
        h = model.layer4[0](h)

        h = model.avgpool(h)

        h = h.view(h.size(0), -1)
        output = model.fc(h)

        # output = model(x)

        loss = criterion(output, y)
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        prediction = output.data.max(1)[1]
        accuracy = ( float( prediction.eq(y.data).sum() ) /float(batch_size))*100.0
        if(epoch==0):
            print(i,accuracy)
        train_accu.append(accuracy)
    accuracy_epoch = np.mean(train_accu)
    print(epoch, accuracy_epoch,time.time()-start_time)

    torch.save(model,'temp3d_resnet.model')

    ##### TEST
    model.eval()
    test_accu = []
    random_indices = np.random.permutation(len(test[0]))
    t1 = time.time()

    for i in range(0,len(test[0])-batch_size,batch_size):
        augment = False
        video_list = [(test[0][k],augment)
                        for k in random_indices[i:(batch_size+i)]]
        data = pool_threads.map(loadSequence,video_list)

        next_batch = 0
        for video in data:
            if video.size==0: # there was an exception, skip this batch
                next_batch = 1
        if(next_batch==1):
            continue

        x = np.asarray(data,dtype=np.float32)
        x = Variable(torch.FloatTensor(x)).cuda().contiguous()

        y = test[1][random_indices[i:(batch_size+i)]]
        y = torch.from_numpy(y).cuda()

        #train layers after layer 4
        # with torch.no_grad():
        #     output = model(x)
        with torch.no_grad():
            h = model.conv1(x)
            h = model.bn1(h)
            h = model.relu(h)
            h = model.maxpool(h)

            h = model.layer1(h)
            h = model.layer2(h)
            h = model.layer3(h)
            h = model.layer4[0](h)
            # h = model.layer4[1](h)

            h = model.avgpool(h)

            h = h.view(h.size(0), -1)
            output = model.fc(h)

        prediction = output.data.max(1)[1]
        accuracy = ( float( prediction.eq(y.data).sum() ) /float(batch_size))*100.0
        test_accu.append(accuracy)
    accuracy_test = np.mean(test_accu)
    print('Testing',accuracy_test,time.time()-t1)

torch.save(model,'3d_resnet.model')
pool_threads.close()
pool_threads.terminate()


###########################################################################################

# TEST Model
model = torch.load('3d_resnet.model')
model.cuda()

# save predictions directory
prediction_directory = 'UCF-101-predictions/sequence/'
if not os.path.exists(prediction_directory):
    os.makedirs(prediction_directory)
for label in class_list:
    if not os.path.exists(prediction_directory + label + '/'):
        os.makedirs(prediction_directory + label + '/')


acc_top1 = 0.0
acc_top5 = 0.0
acc_top10 = 0.0

confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES), dtype=np.float32)
random_indices = np.random.permutation(len(test[0]))
mean = np.asarray([0.485, 0.456, 0.406], np.float32)
std = np.asarray([0.229, 0.224, 0.225], np.float32)
model.eval()

num_of_frames = 16
for i in range(len(test[0])):

    t1 = time.time()

    index = random_indices[i]

    filename = test[0][index]
    filename = filename.replace('.avi', '.hdf5')
    filename = filename.replace('UCF-101', 'UCF-101-hdf5-2')

    h = h5py.File(filename, 'r')
    nFrames = len(h['video'])
    frame_index = np.random.randint(nFrames - num_of_frames)
    video = h['video'][frame_index:(frame_index + num_of_frames)]

    data = []
    for frame in video:
        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE))
        frame = frame.astype(np.float32)
        frame = frame / 255.0
        frame = (frame - mean) / std
        data.append(frame)
    data = np.asarray(data)
    data = data.transpose(3, 0, 1, 2)

    h.close()

    prediction = np.zeros((num_of_frames, NUM_CLASSES), dtype=np.float32)

    #starting point 0 and ending point num_of_frame
    #cut len(data)
    #loop_i stores each starting point
    loop_i = list(range(0, num_of_frames, len(data)))
    #add the ending point num_of_frames
    loop_i.append(num_of_frames)

    for j in range(len(loop_i) - 1):
        data_batch = data[loop_i[j]:loop_i[j + 1]]

        with torch.no_grad():
            x = np.asarray(data_batch, dtype=np.float32)
            x = Variable(torch.FloatTensor(x)).cuda().contiguous()
            y = test[1][random_indices[i:(batch_size + i)]]
            y = torch.from_numpy(y).cuda()

            h = model.conv1(x)
            h = model.bn1(h)
            h = model.relu(h)
            h = model.maxpool(h)

            h = model.layer1(h)
            h = model.layer2(h)
            h = model.layer3(h)
            h = model.layer4[0](h)

            h = model.avgpool(h)
            h = h.view(h.size(0), -1)

            output = model.fc(h)

        prediction[loop_i[j]:loop_i[j + 1]] = output.cpu().numpy()

    # saves the `prediction` array in hdf5 format
    filename = filename.replace(
        data_directory + 'UCF-101-hdf5-2/', prediction_directory)
    if not os.path.isfile(filename):
        with h5py.File(filename, 'w') as h:
            h.create_dataset('predictions', data=prediction)

    # softmax
    for j in range(prediction.shape[0]):
        prediction[j] = np.exp(prediction[j]) / np.sum(np.exp(prediction[j]))

    prediction = np.sum(np.log(prediction), axis=0)
    argsort_pred = np.argsort(-prediction)[0:10]

    label = test[1][index]
    confusion_matrix[label, argsort_pred[0]] += 1
    if label == argsort_pred[0]:
        acc_top1 += 1.0
    if np.any(argsort_pred[0:5] == label):
        acc_top5 += 1.0
    if np.any(argsort_pred[:] == label):
        acc_top10 += 1.0

    print('i: %d nFrames: %d t: %f (%f, %f, %f)'
          % (i, nFrames, time.time() - t1, acc_top1 / (i + 1), acc_top5 / (i + 1), acc_top10 / (i + 1)))

number_of_examples = np.sum(confusion_matrix, axis=1)
for i in range(NUM_CLASSES):
    confusion_matrix[i, :] = confusion_matrix[i, :] / \
        np.sum(confusion_matrix[i, :])

results = np.diag(confusion_matrix)
indices = np.argsort(results)

sorted_list = np.asarray(class_list)
sorted_list = sorted_list[indices]
sorted_results = results[indices]

for i in range(NUM_CLASSES):
    print(sorted_list[i], sorted_results[i], number_of_examples[indices[i]])

np.save('3d_resnet_confusion_matrix.npy', confusion_matrix)
