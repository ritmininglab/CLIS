

from __future__ import division
import numpy as np
import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
        logical_gpus = tf.config.experimental.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        print(e)

import tensorflow.keras.backend as K
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import Callback
from imageio import imread as imread
from imageio import imwrite as imsave
from skimage.transform import resize as imresize
from skimage import io, color
from utilMisc import npsigmoid, npdesigmoid, clipsigmoid
from utilModel import getKernelMasksFromm5
from utilModel import preparePriors2
from utilModel import exportZ,exportZm3
from utilModel import mcpredNall
from utilModel import generateDummyMasks
from utilModel import smartloadmasknet
from utilModel import preparePriors5
from utilModel import targetraw2m3
from utilModel import relaxPriorNewKernel
from utilModel import updateWeightm3, prepareWeightm5
from utilModel import proposeNewKernel, initializeNewKernel
from utilMisc import CustomCallbackkernel
from utilMisc import CustomCallback, CustomCallbackpretrain, CustomCallbackmrf
from utilMisc import myloss, mylosssparse, configdims, passweights
from utilInteract import getChainlist,getLabelSpSynthetic,getLabelSpReal
from utilInteract import getConfiMask, InitialPred2Sp
from utilInteract import propAnnoMapsyn, propAnnoMapreal
from utilUser import imshow2, clickXY, collectLabel
from mBnn import myNNZ_pretrain as myNNZ_noRes
from mBnn import myNNZ_masked, myNNZ_maskedallowbias
from utilMisc import getWcls
from utilMisc import ioulogitIndividual as ioulogit
from utilSlic import calculateSlic
import pickle
from matplotlib import pyplot as plt
import cv2



def interseg(realuser, nclicks):

    from utilIO import getTestData
    adamverylarge = tf.keras.optimizers.Adam(learning_rate=0.01, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
    adamlarge = tf.keras.optimizers.Adam(learning_rate=0.002, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
    adam = tf.keras.optimizers.Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
    adamsmall = tf.keras.optimizers.Adam(learning_rate=0.0005, beta_1=0.9, beta_2=0.99, epsilon=1e-07,)
    adamverysmall = tf.keras.optimizers.Adam(learning_rate=0.00025, beta_1=0.9, beta_2=0.99, epsilon=1e-06,)
    sgd = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
    sgd0step = tf.keras.optimizers.SGD(learning_rate=0.)
    verbose = 0

    from config import N,N2,Nbatch,dimdata,dimcls,h1,w1,kldiv
    from config import dims,dims2,trainables,lW,lZ
    from config import dimsvae,paramsvae,h2,w2

    images2, labels2, labelsnum2, testingfiles = getTestData(N2,h1,w1,dimcls)
    wcls = np.ones((dimcls,))/dimcls
    priors = []
    for idx in range(len(dims[0])):
        indim = dims[0][idx]
        outdim = dims[1][idx]
        vpriora = 4./outdim
        vpriorb = 1.
        wpriormu = 0.
        wpriorvar = 1/2 
        bpriormu = 0.
        bpriorvar = 1/2 
        priors.append([vpriora, vpriorb, wpriormu, wpriorvar, bpriormu, bpriorvar])
        
    x = Input(batch_shape=(Nbatch, h1,w1, dimdata), name='inputx')
    xaux = Input(batch_shape=(Nbatch, N), name='inputaux')
    auxi = np.identity(N, dtype=np.int8)
    
    m2 = Model(inputs=[x,], 
              outputs=myNNZ_noRes([x,], dims2, priors, trainables, kldiv, 0.99, True, ))
    m2.compile(loss={'lnow':mylosssparse, 
                            },
              loss_weights={'lnow': 1.0,},
              optimizer=adamverysmall, 
              metrics={'lnow':'accuracy',})
    
    m2.load_weights('auxiliary/m2.h5')
    
    
    priors2 = preparePriors2(m2,0,0, lW, lZ)
    
    
    
    x2 = Input(batch_shape=(1, h1,w1, dimdata), name='inputx')
    xaux2 = Input(batch_shape=(1, 1), name='inputaux')
    auxi2 = np.identity(1)
    
    Nsample = 2
    x3 = Input(batch_shape=(Nsample, h1,w1, dimdata), name='inputx')
    xaux3 = Input(batch_shape=(Nsample, Nsample), name='inputaux')
    auxi3 = np.identity(Nsample)
    
    
    
    
    
    targetraw0 = exportZ(m2, lZ)
    targetraws = []
    for i in range(20):
        targetraws.append([])
    
    
    kldiv = h1*w1
    
    
    dummymasks = generateDummyMasks(dims2)
    m3 = Model(inputs=[x2,xaux2],
              outputs=myNNZ_maskedallowbias([x2,xaux2], dims2, priors2, trainables, kldiv,
                                   1, dummymasks))
    
    m3 = smartloadmasknet(m3,m2, targetraw0,1, lW,lZ)
    m3weights = m3.get_weights()
    
    
    
    from mMRF import pixelRGBsimilarity, mrf
    from mMRF import softlabelmap, initialweightsMRF

    from mEmbed import vaelight2
    from mEmbed import getEmbedding, retrieveZ

    xvae = Input(batch_shape=(1, h2,w2, dimdata), name='inputx') 
    mvae1 = Model(inputs=[xvae,], 
              outputs=vaelight2(xvae, dimsvae, paramsvae))
    mvae1.load_weights('auxiliary/embed.h5')
    
    mulist = []
    logvarlist = []
    
    
    
    for taskid in range(0,N2):
        trndata = images2[taskid:taskid+1,]
        trnlabel = labelsnum2[taskid:taskid+1,].astype(np.int32)
        trnlabel1hot = labels2[taskid:taskid+1,]
        clstruth = trnlabel[0,:]
    
        plt.figure()
        plt.imshow(images2[taskid])
        plt.axis('off')
        plt.show()
        plt.clf()
        plt.figure()
        plt.imshow(clstruth,vmin=-1, vmax=dimcls)
        plt.axis('off')
        plt.show()
        plt.clf()
        
        
        muq, varq = getEmbedding(mvae1, trndata, h2, w2)
        result, distancelist = retrieveZ(mulist, logvarlist, muq, varq, 8)
        if result==-1:
            m3 = targetraw2m3(m3, targetraw0,1, lZ)
        else:
            m3 = targetraw2m3(m3, targetraws[result], 1, lZ)
        
        
        meanpred, clspred, uncertain = mcpredNall(m3,trndata,auxi2, dimcls, 50)
        plt.figure()
        plt.imshow(clspred[0], vmin=-1, vmax=dimcls)
        plt.axis('off')
        plt.show()
        plt.clf()
        plt.figure()
        plt.imshow(uncertain[0], cmap='Reds', vmin=0, vmax=1.6)
        plt.axis('off')
        plt.show()
        plt.clf()
        
        result = ioulogit(to_categorical(clstruth,dimcls), to_categorical(clspred[0],dimcls),\
                             wcls, taskid, 'initialpred')
        m3.set_weights(m3weights)
        
        
        
     
        
        newimage = cv2.imread(testingfiles[taskid], cv2.IMREAD_UNCHANGED)
        imgpath = "auxiliary/"+str(taskid)+".jpg"
        filepath = "auxiliary/"+str(taskid)+".pkl"
        cv2.imwrite(imgpath, newimage)
        
        
        mode = 0
        if mode==0:
            num_sp,mask_sp,size_sp,center_sp,lab_sp,hist_sp,hist_sp2 \
                = calculateSlic(imgpath)
            pickle.dump([num_sp,mask_sp,size_sp,center_sp,lab_sp,hist_sp,hist_sp2], \
                        open(filepath, "wb"))
        else:
            num_sp,mask_sp,size_sp,center_sp,lab_sp,hist_sp,hist_sp2 \
                = pickle.load(open(filepath,"rb"))
        
        
        confimask = getConfiMask(uncertain[0], 1)
        
        chainlist = getChainlist(num_sp, center_sp, hist_sp,hist_sp2)
        
        from utilInteract import getNearlist
        nearlist = getNearlist(num_sp, center_sp)
        from utilInteract import portionSp, criteriaSp
        label_sp, prop_sp = portionSp(mask_sp,size_sp,trnlabel1hot)
        
        criteria_sp = criteriaSp(chainlist, nearlist, label_sp, prop_sp, center_sp, h1,w1)
        
        
        kernelmasks, kernelidx = getKernelMasksFromm5(m3, 4, 0.1, lZ, dims2)
        priors2relaxed = relaxPriorNewKernel(priors2, 1, kernelidx)
        
        m5 = Model(inputs=[x3,xaux3],
                  outputs=myNNZ_masked([x3,xaux3], dims2, priors2relaxed, trainables, kldiv,
                                       1, kernelmasks))
    
        m5 = passweights(m3, m5)
        m5 = prepareWeightm5(m5, lW,lZ)
        """
        """
        m5 = proposeNewKernel(m5,kernelidx, 2, -8, lW,lZ)
        
        m5 = initializeNewKernel(m5,kernelidx, 1e-2, lW)
        m5.compile(loss={'lnow':myloss, 
                                },
                  loss_weights={'lnow': 1.0,},
                  optimizer = adamlarge, 
                  metrics={'lnow':'accuracy',})
        
        
        anno_sp = np.zeros((num_sp))
        skip_sp = np.zeros((num_sp))
        clickpos = []
        clicklabel = []

        
        correct = (clstruth==clspred[0])
        acc_sp = np.zeros((num_sp))
        confi_sp = np.zeros((num_sp))
        
        for i in range(num_sp):
            temp = np.sum(correct * mask_sp[i,:]) / size_sp[i]
            acc_sp[i] = temp
            temp2 = np.sum(confimask * mask_sp[i,:]) / size_sp[i]
            confi_sp[i] = temp2
        
        
        for usriter in range(1):
            if realuser==0: 
                countnow = 0
                threshold = 25**2
                accthresh =0.25
                tempacc = np.copy(acc_sp)
                tempacc[skip_sp==1] = 4
                tempacc[acc_sp>accthresh] = 3
                
                tempacc[criteria_sp==False] = 5
                
                accmarked = np.zeros((num_sp))
                for i in range(nclicks):
                    idxnow = np.argmin(tempacc)
                    if tempacc[idxnow]<1:
                        anno_sp[idxnow] = 1
                        countnow +=1
                        for j in range(num_sp):
                            distsquare = np.sum(np.square(center_sp[idxnow]-center_sp[j]))
                            if distsquare<threshold:
                                accmarked[j] = 1
                            if distsquare<threshold and tempacc[j]<=1:
                                tempacc[j] = 2 
                        tempacc[idxnow] = 4
                
                if countnow < nclicks:
                    tempconfi = np.copy(confi_sp)
                    tempconfi[skip_sp==1] = 4                    
                    tempconfi[criteria_sp==False] = 5
                    
                    tempconfi[anno_sp==1] = 4
                    tempconfi[accmarked==1] = 3 
                    for i in range(countnow, nclicks):
                        idxnow = np.argmin(tempconfi)
                        anno_sp[idxnow] = 1
                        countnow +=1
                        for j in range(num_sp):
                            distsquare = np.sum(np.square(center_sp[idxnow]-center_sp[j]))
                            if distsquare<threshold and tempconfi[j]<=1:
                                tempconfi[j] += 1
                        tempconfi[idxnow] = 4
                mode = 0
                if mode==0:
                    spmap1 = np.zeros((h1,w1))
                    spacc1 = np.zeros((h1,w1))
                    for i in range(num_sp):
                        spacc1 += mask_sp[i]*(1-acc_sp[i])
                        if anno_sp[i]==1:                        
                            spmap1 += mask_sp[i]*label_sp[i]
                mask_usr,labelmap1 \
                    = propAnnoMapsyn(clspred[0],chainlist,center_sp,anno_sp,mask_sp,label_sp,2)
            elif realuser==1: 
                
                imshow2(clspred[0], uncertain[0], dimcls, h1, w1)
                clickpos = clickXY(2*h1, w1, nclicks, clickpos)
                clicklabel = collectLabel(nclicks, clicklabel)
                label_sp,anno_sp,skip_sp,click_sp \
                    = getLabelSpReal(mask_sp,clickpos,clicklabel,anno_sp,skip_sp)

                mask_usr,labelmap1 \
                    = propAnnoMapreal(clspred[0], clickpos,clicklabel, chainlist,anno_sp,mask_sp,2)  

            

            
            labarr = color.rgb2lab(trndata[0])/100
            similaritys = pixelRGBsimilarity(labarr, 0.1)
            
            
            
            mask_usr_weighted = np.copy(mask_usr)
            mask_usr_weighted += 8*(mask_usr==2)
            mask_usr_weighted += 1*(mask_usr==1)
            
            mask_usr_weighted += 1/2*(confimask+1)*(mask_usr==0)
            mrfmask = np.expand_dims(mask_usr_weighted, axis=0)
            
            
            softmap = softlabelmap(meanpred[0], to_categorical(labelmap1, dimcls), mask_usr)
            
            xmrf = Input(batch_shape=(1,))
            paramsmrf = [h1,w1,dimcls, similaritys, 0.5]
            mmrf = Model(inputs=xmrf,
                      outputs=mrf(xmrf,paramsmrf))
            
            mmrf.compile(loss={'lnow': 'categorical_crossentropy',}, 
                      loss_weights={'lnow': 1.,},
                      optimizer=adamverylarge,
                      metrics={'lnow':'accuracy',})   
            
            mmrf = initialweightsMRF(mmrf, softmap, 1e-2, 0.5)
            
            mrfdummy = np.zeros((1,))
            mmrf.fit(mrfdummy,
                  {'lnow': np.expand_dims(softmap, axis=0)},
                  batch_size=1,
                  sample_weight = mrfmask,
                  epochs=401,
                  verbose=0,)
            
            propraw = mmrf.get_weights()[0]
            prophard = propraw.argmax(-1)
            propsoft = propraw - np.max(propraw, axis=-1, keepdims=True)
            propsoft = np.exp(propsoft)/np.sum(np.exp(propsoft), axis=-1, keepdims=True)
        
        
        
    
            
            trndata3 = np.tile(trndata, [Nsample,1,1,1])
            target = np.expand_dims(propsoft, axis=0)
            target3 = np.tile(target, [Nsample,1,1,1])
            m5mask = np.clip(mrfmask,0,10)
            m5mask3 = np.tile(m5mask, [Nsample,1,1])
            
            
            
            m5.fit([trndata3,auxi3],
                  {'lnow': target3, 
                   },
                  epochs=201,
                  sample_weight = m5mask3,
                  verbose=verbose,)
            
            m3 = Model(inputs=[x2,xaux2],
                      outputs=myNNZ_maskedallowbias([x2,xaux2], dims2, priors2relaxed, trainables, kldiv,
                                           1, dummymasks))
            m3 = passweights(m5,m3)
            
            m3 = updateWeightm3(m3, kernelidx, lW, lZ)

            m3weights = m3.get_weights()
            
            mode = 1 
            if mode==1: 
                m3.compile(loss={'lnow':myloss, 
                                        },
                          loss_weights={'lnow': 1.0,},
                          optimizer = adamsmall, 
                          metrics={'lnow':'accuracy',})
                m3.fit([trndata,auxi2],
                      {'lnow': target, 
                       },
                      epochs=201,
                      verbose=verbose,)
            m3weights = m3.get_weights()
            
            
            meanpred, clspred, uncertain = mcpredNall(m3,trndata,auxi2, dimcls, 50)
        
        if realuser==0: 
            _,labelmap2 \
                = propAnnoMapsyn(clspred[0],chainlist,center_sp,anno_sp,mask_sp,label_sp,2)
        elif realuser==1: 
            _,labelmap2 \
                = propAnnoMapreal(clspred[0], clickpos,clicklabel, chainlist,anno_sp,mask_sp,2)
        
        
        softmap = softlabelmap(meanpred[0], to_categorical(labelmap2, dimcls), mask_usr)
        mmrf = Model(inputs=xmrf,
                  outputs=mrf(xmrf,paramsmrf))
        mmrf.compile(loss={'lnow': 'categorical_crossentropy',}, 
                  loss_weights={'lnow': 1.,},
                  optimizer=adamverylarge,
                  metrics={'lnow':'accuracy',})   
        mmrf = initialweightsMRF(mmrf, softmap, 1e-2, 2)
        mmrf.fit(mrfdummy,
              {'lnow': np.expand_dims(softmap, axis=0)},
              batch_size=1,
              sample_weight = mrfmask,
              epochs=401,
              verbose=0,)
        propraw = mmrf.get_weights()[0]
        prophard = propraw.argmax(-1)
        propsoft = propraw - np.max(propraw, axis=-1, keepdims=True)
        propsoft = np.exp(propsoft)/np.sum(np.exp(propsoft), axis=-1, keepdims=True)
        
        plt.figure()
        plt.imshow(prophard, vmin=-1, vmax=dimcls)
        plt.axis('off')
        plt.show()
        plt.clf()
        result = ioulogit(to_categorical(clstruth,dimcls), to_categorical(prophard,dimcls),\
                             wcls, taskid, 'refine')
        priors2 = preparePriors5(m3, 0,0, lW,lZ)
        
        
        targetraws[taskid] = exportZm3(m3, lZ)
        mulist.append(muq)
        logvarlist.append(varq)
    

        
        
        if taskid==3:
            for testid in range(taskid+1,N2):
                trndata = images2[testid:testid+1,]
                trnlabel = labels2[testid:testid+1,]
                clstruth = trnlabel[0].argmax(-1)
                
                muq, varq = getEmbedding(mvae1, trndata, h2, w2)
                result, distancelist = retrieveZ(mulist, logvarlist, muq, varq, 8)
                if result==-1:
                    m3 = targetraw2m3(m3, targetraw0,1, lZ)
                else:
                    m3 = targetraw2m3(m3, targetraws[result], 1, lZ)
                
                meanpred, clspred, uncertain = mcpredNall(m3,trndata,auxi2, dimcls, 50)
                result = ioulogit(to_categorical(clstruth,dimcls),\
                                     to_categorical(clspred[0],dimcls),wcls, taskid, 'forward')
                
            m3.set_weights(m3weights)
        
        if taskid==N2-1:
            for testid in range(0,taskid):
                trndata = images2[testid:testid+1,]
                trnlabel = labelsnum2[testid:testid+1,]
                clstruth = trnlabel[0]
                
                muq, varq = getEmbedding(mvae1, trndata, h2, w2)
                result, distancelist = retrieveZ(mulist, logvarlist, muq, varq, 8)
                if result==-1:
                    m3 = targetraw2m3(m3, targetraw0,1, lZ)
                else:
                    m3 = targetraw2m3(m3, targetraws[result], 1, lZ)
                
                meanpred, clspred, uncertain = mcpredNall(m3,trndata,auxi2, dimcls, 50)
                result = ioulogit(to_categorical(clstruth,dimcls), \
                                     to_categorical(clspred[0],dimcls), wcls, taskid, 'backward')
            
            m3.set_weights(m3weights)


