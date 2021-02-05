from __future__ import division
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import Callback
import tensorflow_probability as tfp

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
def getAllLayerOutput(m):
    layer_outs = []
    for layer in m.layers:
        if layer.name.startswith('input'):
            layer_outs.append([])
        else:
            keras_function = K.function([m.input], [layer.output])
            layer_outs.append(keras_function([[traindata[0:Nbatch,],], 0]))
    return layer_outs
def checklistminmax(lists):
    minval = np.min(lists[0])
    maxval = np.max(lists[0])
    for i in range(1,len(lists)):
        minval = min(np.min(lists[i]),minval)
        maxval = max(np.max(lists[i]),maxval)
    return [minval, maxval]
def checklayerweights(model, layerlist, idx2):
    for i in range(0, len(layerlist)):
        layername = layerlist[i]
        idx = getlayeridx(model, layername)
        temp = model.layers[idx].get_weights()
        print(layername, checklistminmax([temp[idx2]]))

def deBN(bnweights, convweights):
    epsilon = 0.001 
    
    gamma = bnweights[0]
    beta = bnweights[1]
    mamean = bnweights[2]
    mavar = bnweights[3]
    conv = convweights[0]
    bias = convweights[2]
    convvar = convweights[1] - 2
    biasvar = convweights[3] - 2
    temp = gamma / np.sqrt(mavar+epsilon)
    conv2 = conv * temp
    bias2 = (bias-mamean) * temp + beta
    return [conv2, convvar, bias2, biasvar]
def modeldeBN_noz(m, m3):
    layernames = ['b1b','b2a','b2b','b3a','b3b','b4a','b4b','b5a','b5b']
    for layername in layernames:
        idx1 = getlayeridx(m, layername)
        idx2 = getlayeridx(m3, layername)
        convweights = m.layers[idx1].get_weights()
        bnweights = m.layers[idx1+1].get_weights()
        newweight = deBN(bnweights, convweights)
        m3.layers[idx2].set_weights(newweight)
        
    layernames2 = ['b1a','lnow']
    for layername in layernames2:
        newweight = getlayerweights(m, layername)
        idx2 = getlayeridx(m3, layername)
        m3.layers[idx2].set_weights(newweight)
    return m3
def modeldeBN(m, m3):
    layernames = ['b1b','b2a','b2b','b3a','b3b','b4a','b4b','b5a','b5b']
    for layername in layernames:
        idx1 = getlayeridx(m, layername)
        idx2 = getlayeridx(m3, layername)
        convweights = m.layers[idx1].get_weights()
        bnweights = m.layers[idx1+1].get_weights()
        newweight = deBN(bnweights, convweights)
        m3.layers[idx2].set_weights(newweight)
        
    layernames2 = ['b1a','lnow','b1az','b1bz','b2az','b2bz','b3az','b3bz','b4az','b4bz','b5az','b5bz']
    for layername in layernames2:
        newweight = getlayerweights(m, layername)
        idx2 = getlayeridx(m3, layername)
        m3.layers[idx2].set_weights(newweight)
    return m3
