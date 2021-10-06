from __future__ import division

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from skimage.transform import resize as imresize

def getlayeridx(model, layerName):
    index = None
    for idx, layer in enumerate(model.layers):
        if layer.name == layerName:
            index = idx
            break
    return index
def getlayerweights(model, layername):
    idx = getlayeridx(model, layername)
    return model.layers[idx].get_weights() 

from tensorflow.keras import layers
from tensorflow.keras.layers import Reshape, MaxPooling2D, Flatten, UpSampling2D
from tensorflow.keras.layers import Concatenate, Activation
from tensorflow.keras.layers import Conv2D, Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda, Multiply, Add
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras import regularizers


reg = regularizers.l2(1e-8)

class Sample(layers.Layer):
    def __init__(self, nbatch, dim1, kldivide, priormu=0, priorvar=1, name='layername'):
        super(Sample, self).__init__(name=name)
        self.nbatch = nbatch
        self.dim1 = dim1
        self.mu0 = priormu
        self.var0 = priorvar
        self.kldivide = kldivide
    def call(self, x):
        mu = x[0]
        logvar = x[1]
        std = tf.exp(0.5*logvar)
        eps = tf.random.truncated_normal((self.nbatch, self.dim1))
        mc = tf.add(mu, tf.multiply(eps, std))
        
        term0 = -0.5*self.dim1
        term1 = 0.5*tf.reduce_sum(np.log(self.var0) - logvar)
        term2 = 0.5*tf.reduce_sum((tf.exp(logvar) + (mu - self.mu0)**2) / self.var0)
        sumkl = term1 + term2 + term0
        self.add_loss(sumkl / self.kldivide)
        
        return mc
    def compute_output_shape(self, input_shape):
        return (self.nbatch, self.dim1)






def resizeTrainData(N,h1,w1,raw):
    images = np.zeros((N, h1,w1, 3), dtype=np.float32)
    for ii in range(N):
        img = raw[ii]
        r_img = imresize(img, (h1,w1))
        images[ii, :] = np.copy(r_img)
    return images
def reduceinitialization(m,divisor):
    ws = m.get_weights()
    wsnew = []
    for i in range(len(ws)):
        temp = ws[i]/divisor
        wsnew.append(temp)
    m.set_weights(wsnew)
    return m
def reducevarencoded(m,reduce):
    idx = getlayeridx(m, 'd2var')
    weight = m.layers[idx].get_weights()
    m.layers[idx].set_weights([weight[0],weight[1]-reduce])
    return m

def getEmbedding(m, img, h1, w1):
    resized = resizeTrainData(1,h1,w1,img)
    checks = m.predict(resized,  batch_size=1)
    muq = checks[-2][0]
    logvarq = checks[-1][0]
    return [muq, logvarq]


def retrieveZ(mulist, logvarlist, muq, varq, threshold):
    minkl = 1000*threshold
    dim1 = muq.shape[-1]
    
    distancelist = np.zeros((len(mulist),))
    
    for i in range(len(mulist)):
        mup = mulist[i]
        varp = logvarlist[i]
        
        term0 = -0.5*dim1
        term1 = 0.5*np.sum(varp - varq)
        term2 = 0.5*np.sum((np.exp(varq) + (muq - mup)**2) / np.exp(varp))
        sumkl = term1 + term2 + term0
        
        distancelist[i] = sumkl /1000
        
        if sumkl<minkl:
            result = i
            minkl = sumkl
    if minkl>threshold:
        result = -1
    return [result, distancelist]

def vaelight2(datas, dims, params):
    data = datas
    nbatch = params[0]
    kldivide = params[1]
    reshapesize = params[2]
    
    b1a = Conv2D(dims[0], (3,3), activation='relu',padding='same',kernel_regularizer=reg,bias_regularizer=reg,name='b1a')(data)
    p1 = MaxPooling2D((2,2), strides=(2,2), name='p1')(b1a)

    b2a = Conv2D(dims[2], (3,3), activation='relu',padding='same',kernel_regularizer=reg,bias_regularizer=reg,name='b2a')(p1)
    p2 = MaxPooling2D((2,2), strides=(2,2), name='p2')(b2a)
    
    b3a = Conv2D(dims[4], (3,3), activation='relu',padding='same',kernel_regularizer=reg,bias_regularizer=reg,name='b3a')(p2)
    p3 = MaxPooling2D((2,2), strides=(2,2), name='p3')(b3a)
    
    b4a = Conv2D(dims[6], (3,3), activation='relu',padding='same',kernel_regularizer=reg,bias_regularizer=reg,name='b4a')(p3)
    flat = Flatten(name='flat')(b4a)
    
    d1 = Dense(dims[8], activation='relu', kernel_regularizer=reg,bias_regularizer=reg,name='d1')(flat)
    d2mu = Dense(dims[9], activation=None, kernel_regularizer=reg,bias_regularizer=reg,name='d2mu')(d1)
    d2var = Dense(dims[9], activation=None, kernel_regularizer=reg,bias_regularizer=reg,name='d2var')(d1)
    sample = Sample(nbatch, dims[9], kldivide, priormu=0, priorvar=1)([d2mu, d2var])

    return [sample, d2mu, d2var]

def passweightsVAE2decoder(m1,m2):
    lvae = ['b1a','bn1','b2a','bn2','b3a','bn3','d1','bnd1','d2mu','d2var']
    for layername in lvae:
        temp = getlayerweights(m1, layername)
        idx = getlayeridx(m2, layername)
        m2.layers[idx].set_weights(temp)
    return m2
        
        
