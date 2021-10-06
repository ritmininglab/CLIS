from __future__ import division
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras import layers

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

def initializezAll(m, dims, ndata, layernames):
    
    for i in range(len(layernames)):
        layername = layernames[i]
        idx1 = getlayeridx(m, layername)
        abzweights = m.layers[idx1].get_weights()
        
        dim2 = dims[-1][i]
        idx2a = int(dim2*0.4)
        idx2 = int(dim2*0.5)
        temp = 0.05*np.ones((1,dim2))
        temp[0,0:idx2a] = 0.95
        temp[0,idx2a:idx2] = 0.5
        temp = np.log(temp / (1-temp))
        zweight = np.tile(temp, [ndata,1])
    
        newweight = [abzweights[0], abzweights[1], zweight]
        m.layers[idx1].set_weights(newweight)
    return m

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
    temp2 = 2*np.log(temp)
    convvar2 = convvar + temp2
    biasvar2 = biasvar + temp2
    return [conv2, convvar2, bias2, biasvar2]

def modeldeBN_noz(m, m3, lW, lZ):
    for layername in lW:
        idx1 = getlayeridx(m, layername)
        idx2 = getlayeridx(m3, layername)
        convweights = m.layers[idx1].get_weights()
        bnweights = m.layers[idx1+1].get_weights()
        newweight = deBN(bnweights, convweights)
        m3.layers[idx2].set_weights(newweight)
    for layername in lZ:
        idx1 = getlayeridx(m, layername)
        idx2 = getlayeridx(m3, layername)
        newweight = m.layers[idx1].get_weights()
        m3.layers[idx2].set_weights(newweight)
    
    newweight = getlayerweights(m, 'lnow')
    idx2 = getlayeridx(m3, 'lnow')
    m3.layers[idx2].set_weights(newweight)
    return m3


def exportZ(m1, layernames):
    zs = []
    for layername in layernames:
        idx1 = getlayeridx(m1, layername)
        abzweights = m1.layers[idx1].get_weights()
        
        target = abzweights[2]
        zs.append(target)
    return zs
def exportZm3(m1, layernames):
    zs = []
    for layername in layernames:
        idx1 = getlayeridx(m1, layername+'a')
        abzweights = m1.layers[idx1].get_weights()
        
        target = abzweights[2]
        zs.append(target)
    return zs

def mcpredNall(m2,images2,auxiliary2, dimcls, nsample):
    Nall = images2.shape[0]
    h1 = images2.shape[1]
    w1 = images2.shape[2]
    result = np.zeros((nsample,Nall,h1,w1,dimcls))
    for sample in range(nsample):
        preds = m2.predict([images2,auxiliary2])
        logit = preds[0]
        logit = logit - np.max(logit, axis=-1, keepdims=True)
        softmax = np.exp(logit)/np.sum(np.exp(logit), axis=-1, keepdims=True)
        result[sample,:] = np.copy(softmax)
    meanpred1s = np.mean(result, axis=0)
    temp1 = meanpred1s * np.log(meanpred1s + 1e-8)
    uncertainty1 = - np.sum(temp1, axis=-1)
    classpreds = meanpred1s.argmax(-1)
    maxentropy = - np.log(1/dimcls+1e-8)
    uncertainty0s = uncertainty1 / maxentropy
    return [meanpred1s, classpreds, uncertainty0s]



def findNewKernelIdxfromzab(betamean,npropose,threshold):
    
    nkernelidx = []
    for idx in range(len(betamean)):
        betameannow = betamean[idx]
        mask = (betameannow<threshold)
        if np.sum(mask)>1:
            idxs = np.where(mask==True)
            numcandidate = min(len(idxs[0]), npropose)
            candidate = np.copy(idxs[0][0:numcandidate])
            nkernelidx.append(candidate)
        else:
            mask = (betameannow == np.min(betameannow))
            idxs = np.where(mask==True)
            candidate = np.zeros((1,), dtype=np.int64)
            candidate[0] = idxs[0][0]
            nkernelidx.append(candidate)
    return nkernelidx

def getKernelMasksFromm5(m3, npropose, threshold, lZ, dims2):
    
    betameans = []
    for idx in range(len(lZ)):      
        idxZ = getlayeridx(m3, lZ[idx]+'a')
        abweights = m3.layers[idxZ].get_weights()
        asoftplus = np.log(1+np.exp(abweights[0]))
        bsoftplus = np.log(1+np.exp(abweights[1]))
        betameans.append(asoftplus / (asoftplus+bsoftplus))
    
    nkernelidx = findNewKernelIdxfromzab(betameans,npropose,threshold)
    
    nlayers = len(nkernelidx)
    
    kernelmasks = []
    for i in range(nlayers):
        maskactive = np.zeros(dims2[1][i]) 
        for j in range(len(nkernelidx[i])):
            maskactive[nkernelidx[i][j]] = 1
        kernelmasks.append([maskactive, 1-maskactive])
    return kernelmasks, nkernelidx

def generateDummyMasks(dims2):
    kernelmasks = []
    for i in range(len(dims2[1])):
        dim = dims2[1][i]
        maskactive = np.zeros((dim,))
        kernelmasks.append([maskactive, 1-maskactive])
    return kernelmasks


def smartloadmasknet(m4,m1, initialpienk, nbatch, lW, lZ):
    idxW = getlayeridx(m1, 'lnow')
    wweights = m1.layers[idxW].get_weights()
    idxW2 = getlayeridx(m4, 'lnowa')
    m4.layers[idxW2].set_weights(wweights)
    idxW2 = getlayeridx(m4, 'lnowf')
    m4.layers[idxW2].set_weights(wweights)
    
    for idx in range(len(lW)):        
        idxW = getlayeridx(m1, lW[idx])
        wweights = m1.layers[idxW].get_weights()
        idxW2 = getlayeridx(m4, lW[idx]+'f')
        m4.layers[idxW2].set_weights(wweights)
        idxW2 = getlayeridx(m4, lW[idx]+'a')
        m4.layers[idxW2].set_weights(wweights)
    for idx in range(len(lZ)): 
        idxZ = getlayeridx(m1, lZ[idx])
        abweights = m1.layers[idxZ].get_weights()
        
        pienknow = np.tile(initialpienk[idx], [nbatch,1])
        idxZ2 = getlayeridx(m4, lZ[idx]+'f')
        m4.layers[idxZ2].set_weights([abweights[0],abweights[1], pienknow])
        idxZ2 = getlayeridx(m4, lZ[idx]+'a')
        m4.layers[idxZ2].set_weights([abweights[0],abweights[1], pienknow])
    return m4


def proposeNewKernel(m4,nkernelidx, activateZvalue, clipvalue, lW, lZ):
    for idx in range(len(lW)):        
        idxW = getlayeridx(m4, lW[idx]+'a')
        wweights = m4.layers[idxW].get_weights()
        newwv = np.clip(wweights[1], -16, clipvalue)
        newbv = np.clip(wweights[3], -16, clipvalue)
        m4.layers[idxW].set_weights([wweights[0], newwv, wweights[2], newbv])
    for idx in range(len(lZ)): 
        idxZ = getlayeridx(m4, lZ[idx]+'a')
        abzweights = m4.layers[idxZ].get_weights()
        zweight = abzweights[2]
        zweight[-1,nkernelidx[idx]] = activateZvalue
        m4.layers[idxZ].set_weights([abzweights[0], abzweights[1], zweight])
        
    idxa = getlayeridx(m4, 'lnowa')
    aweights = m4.layers[idxa].get_weights()
    newwu = aweights[0]
    newwv = aweights[1]
    newbu = aweights[2]
    newbv = aweights[3]
    aidx = nkernelidx[-1]
    newwv[:,:,aidx,:] = np.clip(newwv[:,:,aidx,:], -16, clipvalue)
    m4.layers[idxa].set_weights([newwu, newwv, newbu, newbv])
    return m4
    
    
def initializeNewKernel(m4,nkernelidx, scaling, lW):
    for idx in range(len(lW)):        
        idxa = getlayeridx(m4, lW[idx]+'a')
        aweights = m4.layers[idxa].get_weights()
        newwu = aweights[0]
        newwv = aweights[1]
        newbu = aweights[2]
        newbv = aweights[3]
        aidx = nkernelidx[idx]
        temp = np.zeros(newwu.shape)
        temp[:,:,:,aidx] = 1
        newwu += np.random.normal(0, 1, temp.shape)  * temp *scaling
        temp = np.zeros(newbu.shape)
        temp[aidx] = 1
        newbu += np.random.normal(0, 1, temp.shape) * temp *scaling
        m4.layers[idxa].set_weights([newwu, newwv, newbu, newbv])
        
    idxa = getlayeridx(m4, 'lnowa')
    aweights = m4.layers[idxa].get_weights()
    newwu = aweights[0]
    newwv = aweights[1]
    newbu = aweights[2]
    newbv = aweights[3]
    aidx = nkernelidx[-1]
    temp = np.zeros(newwu.shape)
    temp[:,:,aidx,:] = 1
    newwu += np.random.normal(0, 1, temp.shape) * temp *scaling
    m4.layers[idxa].set_weights([newwu, newwv, newbu, newbv])
    return m4
    

def smartreduceVar(m, lW, value):
    for idx in range(len(lW)):        
        idxW = getlayeridx(m, lW[idx])
        wweights = m.layers[idxW].get_weights()
        newweights = [wweights[0],wweights[1]-value,wweights[2], wweights[3]-value]
        m.layers[idxW].set_weights(newweights)
    return m

def preparePriors2(m, reducevalue, reducevaluelnow, lW, lZ):

    priors = []
    for idx in range(len(lZ)):        
        idxW = getlayeridx(m, lW[idx])
        idxZ = getlayeridx(m, lZ[idx])
        wweights = m.layers[idxW].get_weights()
        abzweights = m.layers[idxZ].get_weights()
        vpriora = np.log(np.exp(abzweights[0])+1)
        vpriorb = np.log(np.exp(abzweights[1])+1)
        wpriormu = wweights[0]
        wpriorvar = np.exp(wweights[1]-reducevalue)
        bpriormu = wweights[2]
        bpriorvar = np.exp(wweights[3]-reducevalue)
        priors.append([vpriora, vpriorb, wpriormu, wpriorvar, bpriormu, bpriorvar])
    
    idxW = getlayeridx(m, 'lnow')
    wweights = m.layers[idxW].get_weights()
    wpriormu = wweights[0]
    wpriorvar = np.exp(wweights[1]-reducevaluelnow)
    bpriormu = wweights[2]
    bpriorvar = np.exp(wweights[3]-reducevaluelnow)
    priors.append([1,1, wpriormu, wpriorvar, bpriormu, bpriorvar])
    return priors




def checkactivemasknet(m4,nkernelidx, lZ):
    result = np.zeros((len(lZ),))
    for idx in range(len(lZ)): 
        idxZ = getlayeridx(m4, lZ[idx]+'a')
        abzweights = m4.layers[idxZ].get_weights()
        zweight = abzweights[2]
        result[idx] = zweight[-1,nkernelidx[idx,0]]
    return result


def relaxPriorNewKernel(priors2, relaxvalue, nkernelidx):
    priors = []
    hardassign = 1
    for idx in range(len(priors2)-1):
        priornow = priors2[idx]
        vpriora = np.copy(priornow[0])
        vpriorb = np.copy(priornow[1])
        wpriormu = np.copy(priornow[2])
        wpriorvar = np.copy(priornow[3]) 
        bpriormu = np.copy(priornow[4])
        bpriorvar = np.copy(priornow[5]) 
        kernelidx = nkernelidx[idx]
        wpriorvar[:,:,:,kernelidx] = hardassign
        bpriorvar[kernelidx] = hardassign
        priors.append([vpriora, vpriorb, wpriormu, wpriorvar, bpriormu, bpriorvar])
    priornow = priors2[len(priors2)-1]
    wpriormu = np.copy(priornow[2])
    wpriorvar = np.copy(priornow[3]) 
    bpriormu = np.copy(priornow[4])
    bpriorvar = np.copy(priornow[5]) 
    kernelidx = nkernelidx[len(priors2)-2]
    wpriorvar[:,:,kernelidx,:] = hardassign
    priors.append([1,1, wpriormu, wpriorvar, bpriormu, bpriorvar])
    return priors

def preparePriors5(m, reducevalue, lnowreducevalue, lW,lZ):
    priors = []
    for idx in range(len(lZ)):        
        idxW = getlayeridx(m, lW[idx]+'a')
        idxZ = getlayeridx(m, lZ[idx]+'a')
        wweights = m.layers[idxW].get_weights()
        abzweights = m.layers[idxZ].get_weights()
        vpriora = np.log(np.exp(abzweights[0])+1)
        vpriorb = np.log(np.exp(abzweights[1])+1)
        wpriormu = wweights[0]
        wpriorvar = np.exp(wweights[1]-reducevalue)
        bpriormu = wweights[2]
        bpriorvar = np.exp(wweights[3]-reducevalue)
        priors.append([vpriora, vpriorb, wpriormu, wpriorvar, bpriormu, bpriorvar])
    
    idxW = getlayeridx(m, 'lnowa')
    wweights = m.layers[idxW].get_weights()
    wpriormu = wweights[0]
    wpriorvar = np.exp(wweights[1]-lnowreducevalue)
    bpriormu = wweights[2]
    bpriorvar = np.exp(wweights[3]-lnowreducevalue)
    priors.append([1,1, wpriormu, wpriorvar, bpriormu, bpriorvar])
    return priors

def prepareWeightm5(m5, lW, lZ):
    for idx in range(len(lW)):        
        idxf = getlayeridx(m5, lW[idx]+'f')
        idxa = getlayeridx(m5, lW[idx]+'a')
        aweights = m5.layers[idxa].get_weights()
        m5.layers[idxf].set_weights(aweights)
    for idx in range(len(lZ)): 
        idxf = getlayeridx(m5, lZ[idx]+'f')
        idxa = getlayeridx(m5, lZ[idx]+'a')
        aweights = m5.layers[idxa].get_weights()
        m5.layers[idxf].set_weights(aweights)
    idxf = getlayeridx(m5, 'lnowf')
    idxa = getlayeridx(m5, 'lnowa')
    aweights = m5.layers[idxa].get_weights()
    m5.layers[idxf].set_weights(aweights)
    return m5

def updateWeightm3(m5, nkernelidx, lW, lZ):
    for idx in range(len(lW)):        
        idxf = getlayeridx(m5, lW[idx]+'f')
        idxa = getlayeridx(m5, lW[idx]+'a')
        fweights = m5.layers[idxf].get_weights()
        aweights = m5.layers[idxa].get_weights()
        newwu = fweights[0]
        newwv = fweights[1]
        newbu = fweights[2]
        newbv = fweights[3]
        aidx = nkernelidx[idx]
        newwu[:,:,:,aidx] = np.copy(aweights[0][:,:,:,aidx])
        newwv[:,:,:,aidx] = np.copy(aweights[1][:,:,:,aidx])
        newbu[aidx] = np.copy(aweights[2][aidx])
        newbv[aidx] = np.copy(aweights[3][aidx])
        m5.layers[idxf].set_weights([newwu, newwv, newbu, newbv])
        m5.layers[idxa].set_weights([newwu, newwv, newbu, newbv])
    for idx in range(len(lZ)): 
        idxf = getlayeridx(m5, lZ[idx]+'f')
        idxa = getlayeridx(m5, lZ[idx]+'a')
        fweights = m5.layers[idxf].get_weights()
        aweights = m5.layers[idxa].get_weights()
        newa = fweights[0]
        newb = fweights[1]
        newpie = fweights[2]
        aidx = nkernelidx[idx]
        newa[aidx] = np.copy(aweights[0][aidx])
        newb[aidx] = np.copy(aweights[1][aidx])
        newpie[:,aidx] = np.copy(aweights[2][:,aidx])
        m5.layers[idxf].set_weights([newa, newb, newpie])
        m5.layers[idxa].set_weights([newa, newb, newpie])
    idxf = getlayeridx(m5, 'lnowf')
    idxa = getlayeridx(m5, 'lnowa')
    fweights = m5.layers[idxf].get_weights()
    aweights = m5.layers[idxa].get_weights()
    newwu = fweights[0]
    newwv = fweights[1]
    newbu = fweights[2]
    newbv = fweights[3]
    aidx = nkernelidx[-1]
    newwu[:,:,aidx,:] = np.copy(aweights[0][:,:,aidx,:])
    newwv[:,:,aidx,:] = np.copy(aweights[1][:,:,aidx,:])
    m5.layers[idxf].set_weights([newwu, newwv, newbu, newbv])
    m5.layers[idxa].set_weights([newwu, newwv, newbu, newbv])
    return m5

def smartreduceVarMasked(m, reducevalue, lW):
    for idx in range(len(lW)):        
        idxW = getlayeridx(m, lW[idx]+'a')
        wweights = m.layers[idxW].get_weights()
        newweights = [wweights[0],wweights[1]-reducevalue,wweights[2], wweights[3]-reducevalue]
        m.layers[idxW].set_weights(newweights)
    idxW = getlayeridx(m, 'lnowa')
    wweights = m.layers[idxW].get_weights()
    newweights = [wweights[0],wweights[1]-reducevalue,wweights[2], wweights[3]-reducevalue]
    m.layers[idxW].set_weights(newweights)
    return m

def adjustFrozenVarMasked(m, reducevalue, lW):
    for idx in range(len(lW)):        
        idxW = getlayeridx(m, lW[idx]+'f')
        wweights = m.layers[idxW].get_weights()
        
        newweights = [wweights[0],wweights[1]-reducevalue,wweights[2], wweights[3]-reducevalue]
        m.layers[idxW].set_weights(newweights)
    idxW = getlayeridx(m, 'lnowf')
    wweights = m.layers[idxW].get_weights()
    newweights = [wweights[0],wweights[1]-reducevalue,wweights[2], wweights[3]-reducevalue]
    m.layers[idxW].set_weights(newweights)
    return m




def targetraw2m3(m3, initialpienk, nbatch, lZ):
    
    for idx in range(len(lZ)): 
        idxZ = getlayeridx(m3, lZ[idx]+'a')
        abweights = m3.layers[idxZ].get_weights()
        
        pienknow = np.tile(initialpienk[idx], [nbatch,1])
        idxZ2 = getlayeridx(m3, lZ[idx]+'f')
        m3.layers[idxZ2].set_weights([abweights[0],abweights[1], pienknow])
        idxZ2 = getlayeridx(m3, lZ[idx]+'a')
        m3.layers[idxZ2].set_weights([abweights[0],abweights[1], pienknow])
    return m3

def expandm1(m1,m5,dims,dims2,lW,lZ, half1, half3):
    
    
    for i in range(len(lW)):        
        idx = getlayeridx(m1, lW[i])
        wsold = m1.layers[idx].get_weights()
        wold = wsold[0]
        wvold = wsold[1]
        bold = wsold[2]
        bvold = wsold[3]
        wsnew = m5.layers[idx].get_weights()
        wnew = np.copy(wsnew[0]) * 1e-5
        wvnew = np.copy(wsnew[1]) - 4
        bnew = np.copy(wsnew[2]) * 1e-5
        bvnew = np.copy(wsnew[3]) - 4
        dold1 = dims[0][i]
        dold2 = dims[1][i]
        if i in {8,10,12}:
            half1b = dold1-half1
            
            wnew[:,:,0:half1,0:dold2] = np.copy(wold[:,:,0:half1,:])
            wnew[:,:,half3:half3+half1b,0:dold2] = np.copy(wold[:,:,half1:dold1,:])
            wvnew[:,:,0:half1,0:dold2] = np.copy(wvold[:,:,0:half1,:])
            wvnew[:,:,half3:half3+half1b,0:dold2] = np.copy(wvold[:,:,half1:dold1,:])
            bnew[0:dold2] = np.copy(bold)
            bvnew[0:dold2] = np.copy(bvold)
        else:
            wnew[:,:,0:dold1,0:dold2] = np.copy(wold)
            wvnew[:,:,0:dold1,0:dold2] = np.copy(wvold)
            bnew[0:dold2] = np.copy(bold)
            bvnew[0:dold2] = np.copy(bvold)
        m5.layers[idx].set_weights([wnew,wvnew,bnew,bvnew])
    for i in range(len(lZ)):        
        idx = getlayeridx(m1, lZ[i])
        wsold = m1.layers[idx].get_weights()
        aold = wsold[0]
        bold = wsold[1]
        zold = wsold[2]
        wsnew = m5.layers[idx].get_weights()
        anew = np.copy(wsnew[0]) - 2 
        bnew = np.copy(wsnew[1])
        znew = np.copy(wsnew[2]) - 4 
        dold2 = dims[1][i]
        anew[0:dold2] = np.copy(aold)
        bnew[0:dold2] = np.copy(bold)
        znew[:,0:dold2] = np.copy(zold)
        m5.layers[idx].set_weights([anew,bnew,znew])
    
    idx = getlayeridx(m1, 'lnow')
    wsold = m1.layers[idx].get_weights()
    wold = wsold[0]
    wvold = wsold[1]
    bold = wsold[2]
    bvold = wsold[3]
    wsnew = m5.layers[idx].get_weights()
    wnew = np.copy(wsnew[0]) * 1e-5
    wvnew = np.copy(wsnew[1]) - 4
    dold1 = dims[0][-1]
    dold2 = dims[1][-1]
    wnew[:,:,0:dold1,0:dold2] = np.copy(wold)
    wvnew[:,:,0:dold1,0:dold2] = np.copy(wvold)
    m5.layers[idx].set_weights([wnew,wvnew,bold,bvold])
    return m5
