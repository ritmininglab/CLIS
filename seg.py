from __future__ import division

import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import matplotlib.pyplot as plt
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
import tensorflow_probability as tfp
from util1 import *
from util2 import *
from util3 import *
from imageio import imread as imread
from imageio import imwrite as imsave
from skimage.transform import resize as imresize
from superp import *
from custom import *
import pickle
import math
from tqdm import trange
from misc import *
from matplotlib import pyplot as plt
import cv2
import os
import copy
import random
from skimage import io, color
from skimage import img_as_ubyte

adam = tf.keras.optimizers.Adam(learning_rate=0.005, beta_1=0.9, beta_2=0.99, epsilon=1e-07,) 
adamt = tf.keras.optimizers.Adam(learning_rate=0.0025, beta_1=0.9, beta_2=0.99, epsilon=1e-06,)
adams = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
sgd = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
sgd0step = tf.keras.optimizers.SGD(learning_rate=0.)
images2, labels2, labelsnum2, testingfiles = gtest(N2,h1,w1,dmclass)
images3, labels3, labelsnum3, corefiles = gcore(Ncore,h1,w1,dmclass)

prs = []
for idx in range(len(dms[0])):
    indm = dms[0][idx]
    outdm = dms[1][idx]
    vpra = 5./40
    vprb = 1.
    wprmu = 0.
    wprvar = 1. 
    bprmu = 0.
    bprvar = 1. 
    prs.append([vpra, vprb, wprmu, wprvar, bprmu, bprvar])

x = Input(batch_shape=(Nbatch, h1,w1, dmdata), name='inputx')
xaux = Input(batch_shape=(Nbatch, N), name='inputaux')
auxl = np.identity(N)


m = Model(inputs=[x, xaux], 
          outputs=myNNbnZ_noRes([x, xaux], dms, prs, trainables, kldiv, 0.95, True, N))
m.compile(loss={'lnow':mloss, 
                        },
          loss_weights={'lnow': 1.0,},
          optimizer=adam, 
          metrics={'lnow':'accuracy',})
m.load_weights('auxiliary/pre0.h5')
m0weights0 = m.get_weights()


m1 = Model(inputs=[x,xaux], 
          outputs=myNNZ_noRes([x,xaux], dms, prs, trainables, kldiv, 0.95, True, N))
m1.compile(loss={'lnow':mloss, 
                        },
          loss_weights={'lnow': 1.0,},
          optimizer=adamt,
          metrics={'lnow':'accuracy',})
m1.load_weights('auxiliary/pre1.h5')
m1weights0 = m1.get_weights()

target = exportz(m1)
xkn = Input(batch_shape=(Nbatch, h1,w1, dmdata), name='inputx')
mkn = Model(inputs=[xkn], 
          outputs=myNN_kn([xkn]))
mkn.compile(loss={'lnow':'binary_crossentropy',
                        },
          loss_weights={'lnow': 1.0,},
          optimizer=adam, 
          metrics={'lnow':'binary_accuracy',})
mkn.load_weights('auxiliary/pre2.h5')


oldmodel = m1

prs2 = prepprs2(m1)
auxl2 = np.identity(1)

x2 = Input(batch_shape=(1, h1,w1, dmdata), name='inputx') 
xaux2 = Input(batch_shape=(1, 1), name='inputaux')
m2 = Model(inputs=[x2,xaux2],
          outputs=myNNZ_noRes([x2,xaux2], dms, prs2, trainables, kldiv, 0.95, True, 1))

m2 = smarts(oldmodel,m2,1)
m2weights0 = m2.get_weights()


x3 = Input(batch_shape=(Ncore+1, h1,w1, dmdata), name='inputx') 
xaux3 = Input(batch_shape=(Ncore+1, Ncore+1), name='inputaux')
auxl3 = np.identity(Ncore+1)
m3 = Model(inputs=[x3,xaux3],
          outputs=myNNZ_noRes([x3,xaux3], dms, prs2, trainables, kldiv, 0.95, True, Ncore+1))

for taskid in range(0,1): 
    traindata = images2[taskid:taskid+1,]
    trainlabel = labels2[taskid:taskid+1,]
    clstruth = trainlabel[0,:].argmax(-1)
    check = mkn.predict([traindata])[0]
    m2 = importz(m2, check)
    meanpred1, classpred, uncertainty0 = mcpred(m2,traindata,auxl2)
    
    plt.figure()
    plt.imshow(clstruth,vmin=0, vmax=dmclass)
    plt.axis('off')
    plt.savefig('output/'+str(taskid)+'-tru.png',bbox_inches='tight')
    plt.show()
    plt.clf()    
    plt.figure()
    plt.imshow(classpred, vmin=0, vmax=dmclass)
    plt.axis('off')
    plt.savefig('output/'+str(taskid)+'-ini.png',bbox_inches='tight')
    plt.show()
    plt.clf()    
    plt.figure()
    plt.imshow(uncertainty0, cmap='Reds', vmin=0, vmax=320)
    plt.axis('off')
    plt.savefig('output/'+str(taskid)+'-unc.png',bbox_inches='tight')
    plt.show()
    plt.clf()
    
    
    newimage = cv2.imread(testingfiles[taskid], cv2.IMREAD_UNCHANGED)
    imgpath = "auxiliary/"+str(taskid)+".png"
    cv2.imwrite(imgpath, newimage)    
    filepath = "auxiliary/"+str(taskid)+".pkl"    
    numsp,masksp,sizesp,centersp,labsp,histosp = calculatesuper(imgpath)
    pickle.dump([numsp,masksp,sizesp,centersp,labsp,histosp],open(filepath, "wb"))
    annotatedsp = np.zeros((numsp))
    skipsp = np.zeros((numsp))
    
    for interactIter in range(1):
        tempfilepath = imgpath
        nclicks = 5
        imshow2(classpred, tempfilepath, dmclass, h1, w1)
        clickpos = clickXY(tempfilepath, h1, w1, nclicks)
        clicklabel = collectLabel(nclicks)
        
        clickspr = []
        keyloc = np.zeros((h1, w1))
        for i in range(nclicks):
            for idxnow in range(numsp):
                if masksp[idxnow, clickpos[i][1], clickpos[i][0]]==1:
                    keyloc += masksp[idxnow,:]
                    annotatedsp[idxnow] = 1
                    skipsp[idxnow] = 1
                    clickspr.append(idxnow);
                    break
        mask4d = np.expand_dims(np.expand_dims(keyloc, axis=0), axis=-1)  
        
        maskusr = np.zeros((h1,w1))
        pxlmap1 = np.zeros((h1,w1,dmclass))
        for i in range(nclicks):
            idxnow = clickspr[i]
            maskusr += masksp[idxnow]
            
            tempmask = np.expand_dims(masksp[idxnow], axis=-1)
            temprefined = np.zeros((1,1,dmclass))
            temprefined[0,0,clicklabel[i]] = 1
            pxlmap1 += tempmask*temprefined
        
        labelmap1 = np.argmax(pxlmap1, axis=-1)
        binmask = np.zeros((1, h1, w1, 1))
        binmask[0,:,:,0] = np.copy(maskusr)        
        datacore = images3
        datacore[-1:,:] = np.copy(traindata)
        labelcore = labels3
        labelcore[-1:,:] = np.copy(trainlabel)

        mymask = np.ones((Ncore+1,h1,w1))
        mymask[-1,:] = np.copy(binmask[:,:,:,0])*5e2
        mymask = tf.dtypes.cast(mymask, tf.float32)
        mywce = WCE(mymask)
        
        m3.compile(loss={'lnow':mywce, 
                                },
                  loss_weights={'lnow': 1.0,},
                  optimizer=adams,
                  metrics={'lnow':'accuracy',})
        m3 = smarts(oldmodel,m3, Ncore+1)
        m3.fit([datacore,auxl3],
              {'lnow': labelcore,
               },
              batch_size=Ncore+1, 
              epochs=101,
              verbose=verbose)
        m3weights0 = m3.get_weights()
        
        oldmodel = m3
        prs2 = prepprs2(oldmodel)
        m2 = smarts2(oldmodel,m2,1,Ncore)
        m2weights0 = m2.get_weights()
        meanpred1, classpred, uncertainty0 = mcpred(m2,traindata,auxl2)
        plt.figure()
        plt.imshow(classpred, vmin=0, vmax=dmclass)
        plt.axis('off')
        plt.savefig('output/'+str(taskid)+'-refine.png',bbox_inches='tight')
        plt.show()
        plt.clf()