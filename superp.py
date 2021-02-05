from __future__ import division
import pickle
import numpy as np
import math
import cv2
from skimage import io, color
from tqdm import trange
from matplotlib import pyplot as plt
from skimage import img_as_ubyte

def hog1(nbins, mag, angle):
    if angle>=180:
        angle = angle-180
    bin_width = int(180 / nbins)
    
    hog = np.zeros(nbins)
    lower_bin_idx = int(angle / bin_width)
    hog[lower_bin_idx] = mag
    return hog

def hog2(nbins, mag, angle):
    if angle>=180:
        angle = angle-180
    bin_width = int(180 / nbins)
    
    hog = np.zeros(nbins)
    lower_bin_idx = int(angle / bin_width)
    hog[lower_bin_idx] = 1
    return hog


class Cluster(object):
    clst_index = 1

    def __init__(self, h, w, l=0, a=0, b=0):
        self.update(h, w, l, a, b)
        self.pixels = []
        self.no = self.clst_index
        Cluster.clst_index += 1
    
    def update(self, h, w, l, a, b):
        self.h = h
        self.w = w
        self.l = l
        self.a = a
        self.b = b

    def __str__(self):
        return "{},{}:{} {} {} ".format(self.h, self.w, self.l, self.a, self.b)

    def __repr__(self):
        return self.__str__()


class SLICProcessor(object):
    @staticmethod
    def openimg(path):
        rgb = io.imread(path)
        lab_arr = color.rgb2lab(rgb)
        return lab_arr

    @staticmethod
    def svlabimg(path, lab_arr):
        rgb_arr = color.lab2rgb(lab_arr)
        io.imsave(path, rgb_arr)

    def mkclst(self, h, w):
        h = int(h)
        w = int(w)
        return Cluster(h, w,
                       self.data[h][w][0],
                       self.data[h][w][1],
                       self.data[h][w][2])

    def __init__(self, filename, K, M):
        self.K = K 
        self.M = M 

        self.data = self.openimg(filename)
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.N = self.image_height * self.image_width
        self.S = int(math.sqrt(self.N / self.K))

        self.clsts = []
        self.label = {}
        self.dis = np.full((self.image_height, self.image_width), np.inf)

    def initclsts(self):
        h = self.S / 2
        w = self.S / 2
        while h < self.image_height:
            while w < self.image_width:
                self.clsts.append(self.mkclst(h, w))
                w += self.S
            w = self.S / 2
            h += self.S

    def getgrad(self, h, w):
        if w + 1 >= self.image_width:
            w = self.image_width - 2
        if h + 1 >= self.image_height:
            h = self.image_height - 2

        gradient = self.data[h + 1][w + 1][0] - self.data[h][w][0] + \
                   self.data[h + 1][w + 1][1] - self.data[h][w][1] + \
                   self.data[h + 1][w + 1][2] - self.data[h][w][2]
        return gradient

    def mvclsts(self):
        for clst in self.clsts:
            clst_gradient = self.getgrad(clst.h, clst.w)
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    _h = clst.h + dh
                    _w = clst.w + dw
                    new_gradient = self.getgrad(_h, _w)
                    if new_gradient < clst_gradient:
                        clst.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])
                        clst_gradient = new_gradient

    def assignment(self):
        for clst in self.clsts:
            for h in range(clst.h - 2 * self.S, clst.h + 2 * self.S):
                if h < 0 or h >= self.image_height: continue
                for w in range(clst.w - 2 * self.S, clst.w + 2 * self.S):
                    if w < 0 or w >= self.image_width: continue
                    L, A, B = self.data[h][w]
                    Dc = (math.pow(L - clst.l, 2) +
                        math.pow(A - clst.a, 2) +
                        math.pow(B - clst.b, 2))
                    Ds = (math.pow(h - clst.h, 2) +
                        math.pow(w - clst.w, 2))
                    D = Dc / math.pow(self.M, 2) + Ds / math.pow(self.S, 2)
                    if D < self.dis[h][w]:
                        if (h, w) not in self.label:
                            self.label[(h, w)] = clst
                            clst.pixels.append((h, w))
                        else:
                            self.label[(h, w)].pixels.remove((h, w))
                            self.label[(h, w)] = clst
                            clst.pixels.append((h, w))
                        self.dis[h][w] = D

    def upclst(self):
        
        for clst in self.clsts:
            if not clst.pixels: 
                self.clsts.remove(clst)
            else:
                sum_h = sum_w = number = 0
                for p in clst.pixels:
                    sum_h += p[0]
                    sum_w += p[1]
                    number += 1
                _h = int(sum_h / number)
                _w = int(sum_w / number)
                clst.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])

    def svcurrentimg(self, name):
        image_arr = np.copy(self.data)
        for clst in self.clsts:
            for p in clst.pixels:
                image_arr[p[0]][p[1]][0] = clst.l
                image_arr[p[0]][p[1]][1] = clst.a
                image_arr[p[0]][p[1]][2] = clst.b
            image_arr[clst.h][clst.w][0] = 0
            image_arr[clst.h][clst.w][1] = 0
            image_arr[clst.h][clst.w][2] = 0
        self.svlabimg(name, image_arr)

    def itrtimes(self, times, imgname):
        self.initclsts()
        self.mvclsts()
        for i in range(times):
            self.assignment()
            self.upclst()

def calculatesuper(imgname):
    p = SLICProcessor(imgname, 2500, 10) 
    p.itrtimes(5, imgname)
    
    h = p.image_height
    w = p.image_width
    clsts = p.clsts
    labs = p.data
    
    nums = len(clsts)
    masks = np.zeros((nums, h, w))
    sizes = np.zeros((nums))
    centers = np.zeros((nums,2))
    labsp = np.zeros((nums,3))
    for i in range(0,nums):
        temp1 = np.asarray(clsts[i].pixels)
        tempsize = len(temp1)
        tempmask = np.zeros((h, w))
        tempmask[temp1[:,0],temp1[:,1]] = 1
        
        sizes[i] = tempsize
        masks[i] = tempmask
        temp2 = sum(temp1)
        centers[i,0] = temp2[0]/tempsize
        centers[i,1] = temp2[1]/tempsize
        
        for k in range(0,3):
            temp3 = tempmask*labs[:,:,k]
            labsp[i,k] = np.sum(temp3)/tempsize
    

    num_bin = 25
    num_bins = num_bin*3

    histos = np.zeros((nums,num_bins),dtype="float32")
    
    cv2image = cv2.imread(imgname)
    chans = cv2.split(cv2image)
    
    for i in range(0,nums):
        tempmask = img_as_ubyte(masks[i])
        features = np.zeros((3,num_bin))
        
        for (ii, chan) in enumerate(chans):
            hist1d = cv2.calcHist([chan], [0], tempmask, [num_bin], [0, 256])
            features[ii,:] = hist1d.T[0]
            
        temphist = features.flatten()
        histos[i] = temphist /  sizes[i]
    
    return nums,masks,sizes,centers,labsp, histos
