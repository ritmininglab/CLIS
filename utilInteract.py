from __future__ import division
import numpy as np
import tensorflow as tf

def getChainlist(num_sp, center_sp, hist_sp,hist_sp2):
    distthreshold = 25*2
    similaritythreshold = 0.6
    labdiv = 2 
    hogdiv = 10 
    chainlist = []
    for i in range(num_sp):
        chainlist.append([])
    for i in range(num_sp):
        for j in range(i+1,num_sp):
            distsquare = (center_sp[i,0]-center_sp[j,0])**2+(center_sp[i,1]-center_sp[j,1])**2
            if distsquare < distthreshold:
                temp = np.sum((hist_sp[i]-hist_sp[j])**2)
                temp2 = np.sum((hist_sp2[i]-hist_sp2[j])**2)
                similarity = np.exp(- temp/labdiv - temp2/hogdiv)
                if similarity > similaritythreshold:
                    chainlist[i].append(j)
                    chainlist[j].append(i)
    return chainlist

def getNearlist(num_sp, center_sp):
    distthreshold = 8*2 
    chainlist = []
    for i in range(num_sp):
        chainlist.append([])
    for i in range(num_sp):
        for j in range(i+1,num_sp):
            distsquare = (center_sp[i,0]-center_sp[j,0])**2+(center_sp[i,1]-center_sp[j,1])**2
            if distsquare < distthreshold:
                chainlist[i].append(j)
                chainlist[j].append(i)
    return chainlist

def portionSp(mask_sp,size_sp,trnlabel):
    num_sp = mask_sp.shape[0]
    label_sp = np.zeros((num_sp), dtype=np.int8)
    prop_sp = np.zeros((num_sp))
    for i in range(num_sp):
        temp = np.expand_dims(mask_sp[i], axis=-1) * trnlabel[0]
        temp2 = np.sum(np.sum(temp,axis=0), axis=0) / size_sp[i]
        label_sp[i] = np.argmax(temp2)
        prop_sp[i] = temp2[label_sp[i]]
    return label_sp, prop_sp

def criteriaSp(chainlist, nearlist, label_sp, prop_sp, center_sp, h1, w1):
    num_sp = label_sp.shape[0]
    criteria_sp = np.zeros((num_sp,), dtype=bool)
    for idxsp in range(num_sp):
        result = True
        label = label_sp[idxsp]
        if min(center_sp[idxsp][0], h1-center_sp[idxsp][0])<3:
            result = False
        if min(center_sp[idxsp][1], w1-center_sp[idxsp][1])<3:
            result = False
        if prop_sp[idxsp] < 0.8:
            result = False
        for i in range(len(chainlist[idxsp])):
            label2 = label_sp[chainlist[idxsp][i]]
            if label2 != label:
                result = False
        for i in range(len(nearlist[idxsp])):
            label2 = label_sp[nearlist[idxsp][i]]
            if label2 != label:
                result = False
        criteria_sp[idxsp]  = result
    return criteria_sp

def getLabelSpSynthetic(mask_sp, size_sp, trnlabel):
    num_sp = mask_sp.shape[0]
    label_sp = np.zeros((num_sp), dtype=np.int8)
    for i in range(num_sp):
        temp = np.expand_dims(mask_sp[i], axis=-1) * trnlabel[0]
        temp2 = np.sum(np.sum(temp,axis=0), axis=0) / size_sp[i]
        label_sp[i] = np.argmax(temp2)
    return label_sp
def getLabelSpReal(mask_sp, clickpos, clicklabel, anno_sp, skip_sp):
    num_sp = mask_sp.shape[0]
    label_sp = np.zeros((num_sp), dtype=np.int8)
    click_sp = []
    for i in range(len(clickpos)):
        picked = mask_sp[:, clickpos[i][1], clickpos[i][0]]
        check = np.where(picked==1)
        idxnow = check[0][0]
        anno_sp[idxnow] = 1
        skip_sp[idxnow] = 1
        
        click_sp.append(idxnow);
        label_sp[idxnow] = clicklabel[i]
        
    return label_sp, anno_sp, skip_sp, click_sp


def propAnnoMapFromNone(chainlist, anno_sp, mask_sp, label_sp):
    h1 = mask_sp[0].shape[0]
    w1 = mask_sp[0].shape[1]
    label_user = np.copy(label_sp)
    mask_user = np.zeros((h1,w1), dtype=np.int8)
    labelmap1 = np.zeros((h1,w1), dtype=np.int8)
    
    num_sp = len(anno_sp)
    for i in range(num_sp):
        if anno_sp[i]==1:
            mask_user += mask_sp[i]*2
            labelmap1 += mask_sp[i]*label_sp[i]
            
            for j in range(len(chainlist[i])):
                idxneighbor = chainlist[i][j]
                if anno_sp[idxneighbor]==0:
                    label_user[idxneighbor] = label_sp[i]
                    mask_user += mask_sp[idxneighbor]
                    labelmap1 += mask_sp[idxneighbor]*label_sp[i]
    
    return label_user, mask_user, labelmap1


def propAnnoMap(chainlist, clspred, anno_sp, mask_sp, label_sp):
    h1 = mask_sp[0].shape[0]
    w1 = mask_sp[0].shape[1]
    label_user = np.copy(label_sp)
    mask_user = np.zeros((h1,w1), dtype=np.float32)
    labelmap1 = clspred
    
    num_sp = len(anno_sp)
    for i in range(num_sp):
        if anno_sp[i]==1:
            mask_user += mask_sp[i]*2
            labelmap1[mask_sp[i]==1] = 0 
            labelmap1 += mask_sp[i]*label_sp[i]
            
            for j in range(len(chainlist[i])):
                idxneighbor = chainlist[i][j]
                if anno_sp[idxneighbor]==0:
                    label_user[idxneighbor] = label_sp[i]
                    mask_user += mask_sp[idxneighbor]
                    labelmap1[mask_sp[idxneighbor]==1] = 0
                    labelmap1 += mask_sp[idxneighbor]*label_sp[i]
    
    return label_user, mask_user, labelmap1

def getConfiMask(uncertain, scaling):
    certainmask = (1-uncertain)*scaling
    certainmask = np.clip(certainmask, 0, 1)
    certainmask = certainmask.astype(np.float32)
    return certainmask

def InitialPred2Sp(meanpred, mask_sp, size_sp):
    h1 = meanpred.shape[0]
    w1 = meanpred.shape[1]
    superlabelmap = np.zeros((h1,w1), dtype=np.float32)
    
    num_sp = mask_sp.shape[0]
    for i in range(num_sp):
        tempmask = np.expand_dims(mask_sp[i], axis=-1)
        temp = tempmask * meanpred
        temp = np.sum(np.sum(temp, axis=0), axis=0) / size_sp[i]
        
        labelnow = temp.argmax()
        superlabelmap += mask_sp[i]*labelnow
    return superlabelmap

def propAnnoMapPixel(clspred0, center_sp, anno_sp, label_sp, rr):
    h1 = clspred0.shape[0]
    w1 = clspred0.shape[1]
    labelmap1 = np.copy(clspred0)
    mask_user = np.zeros(clspred0.shape)
    for i in range(center_sp.shape[0]):
        if anno_sp[i]==1:
            h0 = int(center_sp[i][0])
            w0 = int(center_sp[i][1])
            class0 = label_sp[i]
            labelmap1[max(0,h0-rr):min(h1,h0+rr), max(0,w0-rr):min(w1,w0+rr)] = class0
            mask_user[max(0,h0-rr):min(h1,h0+rr), max(0,w0-rr):min(w1,w0+rr)] = 2
    return [mask_user,labelmap1]
def propAnnoMapPixelreal(clspred0, clickpos, clicklabel, rr):
    h1 = clspred0.shape[0]
    w1 = clspred0.shape[1]
    labelmap1 = np.copy(clspred0)
    mask_user = np.zeros(clspred0.shape)
    for i in range(len(clickpos)):
        h0 = clickpos[i][1]
        w0 = clickpos[i][0]
        class0 = clicklabel[i]
        labelmap1[max(0,h0-rr):min(h1,h0+rr), max(0,w0-rr):min(w1,w0+rr)] = class0
        mask_user[max(0,h0-rr):min(h1,h0+rr), max(0,w0-rr):min(w1,w0+rr)] = 2
    return [mask_user,labelmap1]

def propAnnoMapsyn(clspred0, chainlist,center_sp,anno_sp,mask_sp,label_sp, rr):
    h1 = clspred0.shape[0]
    w1 = clspred0.shape[1]
    labelmap1 = np.copy(clspred0)
    mask_user = np.zeros(clspred0.shape)
    for i in range(center_sp.shape[0]):
        if anno_sp[i]==1:
            h0 = int(center_sp[i][0])
            w0 = int(center_sp[i][1])
            labelmap1[mask_sp[i]==1] = label_sp[i]
            
            mask_user[mask_sp[i]==1] = 1
            mask_user[max(0,h0-rr):min(h1,h0+rr), max(0,w0-rr):min(w1,w0+rr)] = 2
            
            for j in range(len(chainlist[i])):
                idxneighbor = chainlist[i][j]
                if anno_sp[idxneighbor]==0:
                    mask_user[mask_sp[idxneighbor]==1] = 1
                    labelmap1[mask_sp[idxneighbor]==1] = label_sp[i]
            
    return [mask_user,labelmap1]

def propAnnoMapreal(clspred0, clickpos,clicklabel, chainlist,anno_sp,mask_sp, rr):
    h1 = clspred0.shape[0]
    w1 = clspred0.shape[1]
    labelmap1 = np.copy(clspred0)
    mask_user = np.zeros(clspred0.shape)
    for i in range(len(clickpos)):
        picked = mask_sp[:, clickpos[i][1], clickpos[i][0]]
        check = np.where(picked==1)
        idxsp = check[0][0]
        labelmap1[mask_sp[idxsp]==1] = clicklabel[i]
        mask_user[mask_sp[idxsp]==1] = 1
        
        h0 = clickpos[i][1]
        w0 = clickpos[i][0]
        labelmap1[max(0,h0-rr):min(h1,h0+rr), max(0,w0-rr):min(w1,w0+rr)] = clicklabel[i]
        mask_user[max(0,h0-rr):min(h1,h0+rr), max(0,w0-rr):min(w1,w0+rr)] = 2
        
        for j in range(len(chainlist[idxsp])):
            idxneighbor = chainlist[idxsp][j]
            if anno_sp[idxneighbor]==0:
                mask_user[mask_sp[idxneighbor]==1] = 1
                labelmap1[mask_sp[idxneighbor]==1] = clicklabel[i]
        
    return [mask_user,labelmap1]

def visualizeAnno(clspred0, center_sp,anno_sp, rr):
    h1 = clspred0.shape[0]
    w1 = clspred0.shape[1]
    mask_user = np.zeros(clspred0.shape)
    for i in range(center_sp.shape[0]):
        if anno_sp[i]==1:
            h0 = int(center_sp[i][0])
            w0 = int(center_sp[i][1])
            mask_user[max(0,h0-rr):min(h1,h0+rr), max(0,w0-rr):min(w1,w0+rr)] = 1
    return mask_user


