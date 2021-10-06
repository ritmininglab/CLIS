from __future__ import division
import numpy as np
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.models import Model

import os

from imageio import imread as imread
from imageio import imwrite as imsave
from skimage.transform import resize as imresize



def getTrainData(N,h1,w1,dimclass):
    train_data = []
    datatrain_path = 'train/'
    
    training_images = [datatrain_path + f for f in os.listdir(datatrain_path) if f.endswith(('.jpg', '.jpeg'))] 
    for image in training_images:
        annotation_data = {'image': image}
        train_data.append(annotation_data) 
    
    images = np.zeros((N, h1,w1, 3), dtype=np.float32)
    for ii, data in enumerate(train_data):
        r_img = imread(data['image'])
        images[ii, :] = np.copy(r_img / 255)
    
    labels = np.zeros((N, h1, w1, dimclass))
    labelsnum = np.zeros((N, h1, w1), dtype=np.int8)
    labelpath = [datatrain_path + f for f in os.listdir(datatrain_path) if f.endswith(('.csv'))] 
    for ii, lb in enumerate(labelpath):
        y1 = np.loadtxt(open(lb, "rb"), delimiter=",")
        labelsnum[ii, :] = np.copy(y1)
        y1 = to_categorical(y1,dimclass)
        labels[ii, :] = np.copy(y1)
        
    labels = labels.astype(np.int8)
    
    return images, labels, labelsnum, training_images

def getTestData(N2,h1,w1,dimclass):

    test_data = []
    datatest_path = 'test/'
    
    testing_images = [datatest_path + f for f in os.listdir(datatest_path) if f.endswith(('.jpg', '.jpeg'))] 
    for image in testing_images:
        annotation_data = {'image': image}
        test_data.append(annotation_data) 
    
    images2 = np.zeros((N2, h1,w1, 3), dtype=np.float32)
    for ii, data in enumerate(test_data):
        img = imread(data['image'])    
        images2[ii, :] = np.copy(img / 255)
    
    labels2 = np.zeros((N2, h1, w1, dimclass))
    labelsnum2 = np.zeros((N2, h1, w1), dtype=np.int8)
    labelpath2 = [datatest_path + f for f in os.listdir(datatest_path) if f.endswith(('.csv'))] 
    for ii, lb in enumerate(labelpath2):
        y1 = np.loadtxt(open(lb, "rb"), delimiter=",")
        labelsnum2[ii, :] = np.copy(y1)
        y1 = to_categorical(y1,dimclass)
        labels2[ii, :] = np.copy(y1)
    
    labels2 = labels2.astype(np.int8)
    
    return images2, labels2, labelsnum2, testing_images

def getCoreData(N,h1,w1,dimclass):
    train_data = []
    datatrain_path = 'coreset/'
    
    training_images = [datatrain_path + f for f in os.listdir(datatrain_path) if f.endswith(('.jpg', '.jpeg'))] 
    training_images.sort()
    for image in training_images:
        annotation_data = {'image': image}
        train_data.append(annotation_data) 
    
    images = np.zeros((N+1, h1,w1, 3), dtype=np.float32)
    for ii, data in enumerate(train_data):
        r_img = imread(data['image'])
        images[ii, :] = np.copy(r_img / 255)
    
    labels = np.zeros((N+1, h1, w1, dimclass))
    labelsnum = np.zeros((N+1, h1, w1), dtype=np.int8)
    labelpath = [datatrain_path + f for f in os.listdir(datatrain_path) if f.endswith(('.csv'))] 
    labelpath.sort()
    for ii, lb in enumerate(labelpath):
        y1 = np.loadtxt(open(lb, "rb"), delimiter=",")
        labelsnum[ii, :] = np.copy(y1)
        y1 = to_categorical(y1,dimclass)
        labels[ii, :] = np.copy(y1)
        
    labels = labels.astype(np.int8)
    
    return images, labels, labelsnum, training_images
