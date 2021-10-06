
from __future__ import division
import numpy as np
import tensorflow as tf
import keras.backend as K
'''
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
'''
from tensorflow.keras import layers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda, Multiply, Add
from tensorflow.keras.layers import Softmax, ReLU
from tensorflow.keras.layers import BatchNormalization

from keras import initializers, regularizers

from keras.backend import categorical_crossentropy as CatCE


class unarylayer(layers.Layer):
    def __init__(self, batchsize=1, n=10, k=3, name='layername'):
        super(unarylayer, self).__init__(name=name)
        self.batchsize = batchsize
        self.n=n
        self.k=k
        w_init = tf.random_normal_initializer()
        self.w = tf.Variable(
            initial_value=w_init(shape=(batchsize, n, k), dtype="float32"),
            trainable=True)
    def call(self, inputs):
        return tf.nn.softmax(self.w, axis=-1)
    def compute_output_shape(self, input_shape):
        return (self.batchsize, self.n, self.k)


class pairlayer(layers.Layer):
    def __init__(self, batchsize=1, n=10, name='layername'):
        super(pairlayer, self).__init__(name=name)
        self.batchsize = batchsize
        self.n=n
    def call(self, inputs):
        temp = tf.expand_dims(inputs, axis=2)
        data1 = tf.tile(temp, [1,1,self.n,1])
        data2 = tf.transpose(data1, perm=[0,2,1,3])
        return CatCE(data1, data2) - CatCE(data1,data1)
    def compute_output_shape(self, input_shape):
        return (self.batchsize, self.n, self.n)

def unaryloss(mask):
    def uloss(ytrue, ypred):
        return K.sum(mask*CatCE(ytrue, ypred),axis=[0,1])
    return uloss

def pairloss(mask):
    def ploss(ytrue, ypred):
        return 0.5 * K.sum(mask*ypred,axis=[0,1,2])
    return ploss


catce = tf.losses.categorical_crossentropy

class pixellayer(layers.Layer):
    def __init__(self, h,w,dimclass, similaritys,simweight,name):
        super(pixellayer, self).__init__(name=name)
        self.h=h
        self.w=w
        self.dimclass=dimclass
        self.simw = simweight
        x_init = tf.random_normal_initializer()
        self.x = tf.Variable(
            initial_value=x_init(shape=(h,w,dimclass), dtype="float32"),
            trainable=True)
        self.simup = similaritys[0]
        self.simdown = similaritys[1]
        self.simleft = similaritys[2]
        self.simright = similaritys[3]
        self.simbl = similaritys[4]
        self.simbr = similaritys[5]
        self.simtl = similaritys[6]
        self.simtr = similaritys[7]
    def call(self, inputs):
        x = tf.nn.softmax(self.x, axis=-1)
        up = x[0:(self.h-2), 1:(self.w-1),:]
        down = x[2:(self.h), 1:(self.w-1),:]
        left = x[1:(self.h-1), 0:(self.w-2),:]
        right = x[1:(self.h-1), 2:(self.w),:]
        xc = x[1:(self.h-1), 1:(self.w-1),:]
        kl1 = self.simup * (catce(up,xc)+catce(xc,up)-catce(xc,xc)-catce(up,up))
        kl2 = self.simdown * (catce(down,xc)+catce(xc,down)-catce(xc,xc)-catce(down,down))
        kl3 = self.simleft * (catce(left,xc)+catce(xc,left)-catce(xc,xc)-catce(left,left))
        kl4 = self.simright * (catce(right,xc)+catce(xc,right)-catce(xc,xc)-catce(right,right))
        self.add_loss(tf.reduce_sum(kl1+kl2+kl3+kl4) *self.simw / (2*self.h*self.w))
        
        bl = x[0:(self.h-2), 0:(self.w-2),:]
        br = x[0:(self.h-2), 2:(self.w),:]
        tl = x[2:(self.h), 0:(self.w-2),:]
        tr = x[2:(self.h), 2:(self.w),:]
        kl5 = self.simbl * (catce(bl,xc)+catce(xc,bl)-catce(xc,xc)-catce(bl,bl))
        kl6 = self.simbr * (catce(br,xc)+catce(xc,br)-catce(xc,xc)-catce(br,br))
        kl7 = self.simtl * (catce(tl,xc)+catce(xc,tl)-catce(xc,xc)-catce(tl,tl))
        kl8 = self.simtr * (catce(tr,xc)+catce(xc,tr)-catce(xc,xc)-catce(tr,tr))
        self.add_loss(tf.reduce_sum(kl5+kl6+kl7+kl8) *self.simw / (2*self.h*self.w))
        return tf.expand_dims(x, axis=0)
    def compute_output_shape(self, input_shape):
        return (1, self.h, self.w, self.dimclass)
    
def mrf(dummy, params):
    h = params[0]
    w = params[1]
    dimclass = params[2]
    similaritys = params[3]
    simweight = params[4]
    
    lnow = pixellayer(h,w,dimclass, similaritys,simweight, name='lnow')(dummy)
    lmean = Lambda(lambda x: x, name='lmean')(lnow)
    return [lnow,lmean]


def pixelRGBsimilarity(x, colordivisor):
    h = x.shape[0]
    w = x.shape[1]
    up = x[0:(h-2), 1:(w-1),:]
    down = x[2:(h), 1:(w-1),:]
    left = x[1:(h-1), 0:(w-2),:]
    right = x[1:(h-1), 2:(w),:]
    xc = x[1:(h-1), 1:(w-1),:]
    bl = x[0:(h-2), 0:(w-2),:]
    br = x[0:(h-2), 2:(w),:]
    tl = x[2:(h), 0:(w-2),:]
    tr = x[2:(h), 2:(w),:]
    similarity = []
    sim = np.exp(-np.sum(np.square(xc-up),axis=-1) / colordivisor)
    similarity.append(sim)
    sim = np.exp(-np.sum(np.square(xc-down),axis=-1) / colordivisor)
    similarity.append(sim)
    sim = np.exp(-np.sum(np.square(xc-left),axis=-1) / colordivisor)
    similarity.append(sim)
    sim = np.exp(-np.sum(np.square(xc-right),axis=-1) / colordivisor)
    similarity.append(sim)
    sim = np.exp(-np.sum(np.square(xc-bl),axis=-1) / colordivisor)
    similarity.append(sim)
    sim = np.exp(-np.sum(np.square(xc-br),axis=-1) / colordivisor)
    similarity.append(sim)
    sim = np.exp(-np.sum(np.square(xc-tl),axis=-1) / colordivisor)
    similarity.append(sim)
    sim = np.exp(-np.sum(np.square(xc-tr),axis=-1) / colordivisor)
    similarity.append(sim)
    
    minsimilarity = 0.1
    rescaled = []
    for i in range(len(similarity)):
        temp = np.copy(similarity[i])
        temp[temp<minsimilarity] = 0
        rescaled.append(temp)
    return rescaled

def softlabelmap(meanpred, labelmap, maskuser):
    softmap = np.copy(meanpred)
    softmap[maskuser==2] = labelmap[maskuser==2]
    softmap[maskuser==1] = labelmap[maskuser==1]
    return softmap
    
def initialweightsMRF(mmrf, softmap, small, scaling):
    logit = np.log(softmap+small)
    meanlogit = np.mean(logit, axis=-1, keepdims=True)
    normalized = logit-meanlogit
    mmrf.set_weights([normalized])
    return mmrf
    
    
