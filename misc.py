from __future__ import division
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.utils import to_categorical
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input,Lambda,Concatenate
from tensorflow.keras.layers import Dense,ReLU,ELU,Softmax
from tensorflow.keras import regularizers
from tensorflow.keras.backend import categorical_crossentropy as CatCE
from tensorflow.keras.callbacks import Callback
from util2 import *

initialvar = -5.
small = 1e-5
nsample = 10
Ntask = 8
nepo = 201
nepo2 = 81
N = 60 
Nbatch = 6
Ncore = 6
N2 = 8
h1 = 256
w1 = 384
kldiv = Nbatch*h1*w1
dmdata = 3
dmclass = 8
dms = [[3,40,40,40,40,40,80,40,80,40,40],
        [40,40,40,40,40,40,40,40,40,40,8]]
trainables = []
for i in range(12):
    trainables.append(True)
verbose = 0


def counttotal(xlist):
    total = 0
    for i in range(len(xlist)):
        total += xlist[i].shape[0]
    return total
class CustomCallback(Callback):
    def on_train_begin(self, logs={}):    
        self.epochs = 0    
    def on_epoch_end(self, batch, logs={}):    
        self.epochs += 1     
        if self.epochs % 5 == 1:     
            print("epoch {}, loss {:3.3f}={:3.3f}+{:3.3f} acc {:1.3f}".format(
                self.epochs, logs["loss"], logs["lnow_loss"], logs["loss"]-logs["lnow_loss"],
                logs["lnow_accuracy"])
                )
def mloss(y_true, y_pred):
    temp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
        labels=y_true,logits=y_pred))
    return temp
def WCE(mask):
    def wce(y_true, y_pred):
        temp = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
            labels=y_true,logits=y_pred) * mask)
        return temp
    return wce

def exportz(m1):
    truncate = 40
    layernames = ['b1az','b1bz','b2az','b2bz','b3az','b3bz','b4az','b4bz','b5az','b5bz']
    zs = []
    for layername in layernames:
        idx1 = getlayeridx(m1, layername)
        abzweights = m1.layers[idx1].get_weights()
        zs.append(abzweights[2])
    Ndata = zs[0].shape[0]
    target = np.zeros((Ndata, truncate*10))
    for i in range(10):
        target[:, truncate*i:truncate*(i+1)] = np.copy(zs[i])
        
    target = 1/(1+np.exp(-target))
    return target
def importz(m2, zs):
    zs = np.log(zs/(1-zs))
    
    truncate = 40
    layernames = ['b1az','b1bz','b2az','b2bz','b3az','b3bz','b4az','b4bz','b5az','b5bz']    
    for i in range(10):
        layername = layernames[i]
        zweight = zs[:, truncate*i:truncate*(i+1)]
        idx1 = getlayeridx(m2, layername)
        abzweights = m2.layers[idx1].get_weights()
        newweight = [abzweights[0], abzweights[1], zweight]
        m2.layers[idx1].set_weights(newweight)
    return m2
def smarts(m1,m2, nbatch2):
    layernameW = ['b1a','b1b','b2a','b2b','b3a','b3b','b4a','b4b','b5a','b5b']
    layernameZ = ['b1az','b1bz','b2az','b2bz','b3az','b3bz','b4az','b4bz','b5az','b5bz']
    idxW = getlayeridx(m1, 'lnow')
    wweights = m1.layers[idxW].get_weights()
    m2.layers[idxW].set_weights(wweights)
    for idx in range(len(layernameW)):        
        idxW = getlayeridx(m1, layernameW[idx])
        wweights = m1.layers[idxW].get_weights()
        m2.layers[idxW].set_weights(wweights)
        
        idxZ = getlayeridx(m1, layernameZ[idx])
        abzweights = m1.layers[idxZ].get_weights()
        newzweights = np.mean(abzweights[2], axis=0, keepdims=True)
        
        newzweights = np.tile(newzweights, [nbatch2, 1])
        
        m2.layers[idxZ].set_weights([abzweights[0],abzweights[1],newzweights])
    return m2
def smarts2(m1,m2, nbatch2, idxcheck):
    layernameW = ['b1a','b1b','b2a','b2b','b3a','b3b','b4a','b4b','b5a','b5b']
    layernameZ = ['b1az','b1bz','b2az','b2bz','b3az','b3bz','b4az','b4bz','b5az','b5bz']
    idxW = getlayeridx(m1, 'lnow')
    wweights = m1.layers[idxW].get_weights()
    m2.layers[idxW].set_weights(wweights)
    for idx in range(len(layernameW)):        
        idxW = getlayeridx(m1, layernameW[idx])
        wweights = m1.layers[idxW].get_weights()
        m2.layers[idxW].set_weights(wweights)
        
        idxZ = getlayeridx(m1, layernameZ[idx])
        abzweights = m1.layers[idxZ].get_weights()
        newzweights = abzweights[2][idxcheck:idxcheck+1,:]
        
        newzweights = np.tile(newzweights, [nbatch2, 1])
        
        m2.layers[idxZ].set_weights([abzweights[0],abzweights[1],newzweights])
    return m2
def iniz(m, dm2,nbatch):
    a = 15. 
    b = 1.
    temp = a/(a+b) * np.ones((1,1,1,dm2))
    temp = np.cumprod(temp, axis=-1)
    temp[temp<0.1] = 0.1
    temp = np.log(temp / (1-temp))
    zweight = np.tile(temp, [nbatch,1,1,1])
    
    layernames = ['b1az','b1bz','b2az','b2bz','b3az','b3bz','b4az','b4bz','b5az','b5bz']
    for layername in layernames:
        idx1 = getlayeridx(m, layername)
        abzweights = m.layers[idx1].get_weights()
        newweight = [abzweights[0], abzweights[1], zweight]
        m.layers[idx1].set_weights(newweight)
    return m
def inizHard(m, dm2,nbatch):
    idx1 = int(dm2*0.4)
    idx2 = int(dm2*0.6)
    temp = 0.1*np.ones((1,1,1,dm2))
    temp[0,0,0,0:idx1] = 0.9
    temp[0,0,0,idx1:idx2] = 0.5
    temp = np.log(temp / (1-temp))
    zweight = np.tile(temp, [nbatch,1,1,1])
    
    layernames = ['b1az','b1bz','b2az','b2bz','b3az','b3bz','b4az','b4bz','b5az','b5bz']
    for layername in layernames:
        idx1 = getlayeridx(m, layername)
        abzweights = m.layers[idx1].get_weights()
        newweight = [abzweights[0], abzweights[1], zweight]
        m.layers[idx1].set_weights(newweight)
    return m
def inizAll(m, dm2,ndata):
    idx1 = int(dm2*0.4)
    idx2 = int(dm2*0.6)
    temp = 0.1*np.ones((1,dm2))
    temp[0,0:idx1] = 0.9
    temp[0,idx1:idx2] = 0.5
    temp = np.log(temp / (1-temp))
    zweight = np.tile(temp, [ndata,1])
    
    layernames = ['b1az','b1bz','b2az','b2bz','b3az','b3bz','b4az','b4bz','b5az','b5bz']
    for layername in layernames:
        idx1 = getlayeridx(m, layername)
        abzweights = m.layers[idx1].get_weights()
        newweight = [abzweights[0], abzweights[1], zweight]
        m.layers[idx1].set_weights(newweight)
    return m
def prepprs2(m):
    layernameW = ['b1a','b1b','b2a','b2b','b3a','b3b','b4a','b4b','b5a','b5b']
    layernameZ = ['b1az','b1bz','b2az','b2bz','b3az','b3bz','b4az','b4bz','b5az','b5bz']
    prs = []
    for idx in range(len(layernameZ)):        
        idxW = getlayeridx(m, layernameW[idx])
        idxZ = getlayeridx(m, layernameZ[idx])
        wweights = m.layers[idxW].get_weights()
        abzweights = m.layers[idxZ].get_weights()
        vpra = np.log(np.exp(abzweights[0])+1)
        vprb = np.log(np.exp(abzweights[1])+1)
        wprmu = wweights[0]
        wprvar = np.exp(wweights[1])
        bprmu = wweights[2]
        bprvar = np.exp(wweights[3])
        prs.append([vpra, vprb, wprmu, wprvar, bprmu, bprvar])
    
    idxW = getlayeridx(m, 'lnow')
    wweights = m.layers[idxW].get_weights()
    wprmu = wweights[0]
    wprvar = np.exp(wweights[1])
    bprmu = wweights[2]
    bprvar = np.exp(wweights[3])
    prs.append([1,1, wprmu, wprvar, bprmu, bprvar])
    return prs
def mcpred(m2,traindata,auxl2):
    nsample = 10
    result = np.zeros((nsample,) + (h1,w1,dmclass) )
    for sample in range(nsample):
        preds = m2.predict([traindata,auxl2])
        result[sample,:] = np.copy(preds[-1][-1,:])
    meanpred1 = np.mean(result, axis=0)
    meanpred1 = np.exp(meanpred1)/np.sum(np.exp(meanpred1), axis=-1, keepdims=True)
    
    temp1 = meanpred1 * np.log(meanpred1+1e-8)
    uncertainty1 = - np.sum(temp1, axis=-1)
    classpred = meanpred1.argmax(-1)
    uncertainty0 = 255*uncertainty1 / uncertainty1.max()
    return [meanpred1, classpred, uncertainty0]
