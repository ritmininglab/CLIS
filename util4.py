import numpy as np
import tensorflow as tf

import keras.backend as K
from keras import layers
from keras.layers import Dense,Concatenate,Flatten,Conv2D
from keras.layers import Lambda
from keras.callbacks import Callback



class CustomCallback(Callback):
    def on_train_begin(self, logs={}):    
        self.epochs = 0    
    def on_epoch_end(self, batch, logs={}):    
        self.epochs += 1     
        if self.epochs % 50 == 0:     
            print("epoch {} loss {:5.1f} accuracy {:5.3f}".format(
                self.epochs, logs["loss"], logs["layer4_accuracy"])
                )
class CustomCallback2debug(Callback):
    def on_train_begin(self, logs={}):    
        self.epochs = 0    
    def on_epoch_end(self, batch, logs={}):    
        self.epochs += 1     
        if self.epochs % 50 == 0:     
            print("epoch {} loss {:5.1f} accuracy {:5.3f} {:5.3f}".format(
                self.epochs, logs["loss"], logs["layer4_accuracy"], logs["layer4a_accuracy"])
                )
class CustomCallback2(Callback):
    def on_train_begin(self, logs={}):    
        self.epochs = 0    
    def on_epoch_end(self, batch, logs={}):    
        self.epochs += 1     
        if self.epochs % 1 == 0:     
            print("epoch {} loss {:5.1f} accuracy {:5.2f} kl {:5.2f} mse {:5.2f}".format(
                self.epochs, logs["loss"], logs["layer4_accuracy"], logs["allloss_mae"], logs["layer4a_mse"])
                )

def getlayeridx(model, layerName):
    index = None
    for idx, layer in enumerate(model.layers):
        if layer.name == layerName:
            index = idx
            break
    return index
def checklistminmax(lists):
    minval = np.min(lists[0])
    maxval = np.max(lists[0])
    for i in range(1,len(lists)):
        minval = min(np.min(lists[i]),minval)
        maxval = max(np.max(lists[i]),maxval)
    return[minval, maxval]
