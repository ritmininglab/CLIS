import pickle
import math
from skimage import io, color
import numpy as np
from tqdm import trange

from matplotlib import pyplot as plt
import cv2
from skimage import img_as_ubyte

def hogpixel(nbins, mag, angle):
    if angle>=180:
        angle = angle-180
    bin_width = int(180 / nbins)
    
    hog = np.zeros(nbins)
    lower_bin_idx = int(angle / bin_width)
    hog[lower_bin_idx] = mag
    return hog

def hogpixel2(nbins, mag, angle):
    if angle>=180:
        angle = angle-180
    bin_width = int(180 / nbins)
    
    hog = np.zeros(nbins)
    lower_bin_idx = int(angle / bin_width)
    hog[lower_bin_idx] = 1
    return hog


class Cluster(object):
    cluster_index = 1

    def __init__(self, h, w, l=0, a=0, b=0):
        self.update(h, w, l, a, b)
        self.pixels = []
        self.no = self.cluster_index
        Cluster.cluster_index += 1
    
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
    def open_image(path):
        rgb = io.imread(path)
        lab_arr = color.rgb2lab(rgb)
        return lab_arr

    @staticmethod
    def save_lab_image(path, lab_arr):
        rgb_arr = color.lab2rgb(lab_arr)
        io.imsave(path, rgb_arr)

    def make_cluster(self, h, w):
        h = int(h)
        w = int(w)
        return Cluster(h, w,
                       self.data[h][w][0],
                       self.data[h][w][1],
                       self.data[h][w][2])

    def __init__(self, filename, K, M):
        self.K = K 
        self.M = M 

        self.data = self.open_image(filename)
        self.image_height = self.data.shape[0]
        self.image_width = self.data.shape[1]
        self.N = self.image_height * self.image_width
        self.S = int(math.sqrt(self.N / self.K))

        self.clusters = []
        self.label = {}
        self.dis = np.full((self.image_height, self.image_width), np.inf)

    def init_clusters(self):
        h = self.S / 2
        w = self.S / 2
        while h < self.image_height:
            while w < self.image_width:
                self.clusters.append(self.make_cluster(h, w))
                w += self.S
            w = self.S / 2
            h += self.S

    def get_gradient(self, h, w):
        if w + 1 >= self.image_width:
            w = self.image_width - 2
        if h + 1 >= self.image_height:
            h = self.image_height - 2

        gradient = self.data[h + 1][w + 1][0] - self.data[h][w][0] + \
                   self.data[h + 1][w + 1][1] - self.data[h][w][1] + \
                   self.data[h + 1][w + 1][2] - self.data[h][w][2]
        return gradient

    def move_clusters(self):
        for cluster in self.clusters:
            cluster_gradient = self.get_gradient(cluster.h, cluster.w)
            for dh in range(-1, 2):
                for dw in range(-1, 2):
                    _h = cluster.h + dh
                    _w = cluster.w + dw
                    new_gradient = self.get_gradient(_h, _w)
                    if new_gradient < cluster_gradient:
                        cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])
                        cluster_gradient = new_gradient

    def assignment(self):
        for cluster in self.clusters:
            for h in range(cluster.h - 2 * self.S, cluster.h + 2 * self.S):
                if h < 0 or h >= self.image_height: continue
                for w in range(cluster.w - 2 * self.S, cluster.w + 2 * self.S):
                    if w < 0 or w >= self.image_width: continue
                    L, A, B = self.data[h][w]
                    Dc = (math.pow(L - cluster.l, 2) +
                        math.pow(A - cluster.a, 2) +
                        math.pow(B - cluster.b, 2))
                    Ds = (math.pow(h - cluster.h, 2) +
                        math.pow(w - cluster.w, 2))
                    D = Dc / math.pow(self.M, 2) + Ds / math.pow(self.S, 2)
                    if D < self.dis[h][w]:
                        if (h, w) not in self.label:
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        else:
                            self.label[(h, w)].pixels.remove((h, w))
                            self.label[(h, w)] = cluster
                            cluster.pixels.append((h, w))
                        self.dis[h][w] = D

    def update_cluster(self):
        
        for cluster in self.clusters:
            if not cluster.pixels: 
                self.clusters.remove(cluster)
            else:
                sum_h = sum_w = number = 0
                for p in cluster.pixels:
                    sum_h += p[0]
                    sum_w += p[1]
                    number += 1
                _h = int(sum_h / number)
                _w = int(sum_w / number)
                cluster.update(_h, _w, self.data[_h][_w][0], self.data[_h][_w][1], self.data[_h][_w][2])

    def save_current_image(self, name):
        image_arr = np.copy(self.data)
        for cluster in self.clusters:
            for p in cluster.pixels:
                image_arr[p[0]][p[1]][0] = cluster.l
                image_arr[p[0]][p[1]][1] = cluster.a
                image_arr[p[0]][p[1]][2] = cluster.b
            image_arr[cluster.h][cluster.w][0] = 0
            image_arr[cluster.h][cluster.w][1] = 0
            image_arr[cluster.h][cluster.w][2] = 0
        self.save_lab_image(name, image_arr)

    def iterate_times(self, times, imgname):
        self.init_clusters()
        self.move_clusters()
        for i in range(times):
            self.assignment()
            self.update_cluster()

def calculateSlic(imgname):
    num_initialsuper = 1600 
    p = SLICProcessor(imgname, num_initialsuper, 10) 
    p.iterate_times(8, imgname)
    
    h = p.image_height
    w = p.image_width
    clusters = p.clusters
    labs = p.data
    
    num_sp = len(clusters)
    mask_sp = np.zeros((num_sp, h, w), dtype=bool)
    size_sp = np.zeros((num_sp), dtype=np.int32)
    center_sp = np.zeros((num_sp,2), dtype=np.float32)
    lab_sp = np.zeros((num_sp,3), dtype=np.float32)
    for i in range(0,num_sp):
        temp1 = np.asarray(clusters[i].pixels)
        tempsize = len(temp1)
        tempmask = np.zeros((h, w), dtype=bool)
        tempmask[temp1[:,0],temp1[:,1]] = True
        
        size_sp[i] = tempsize
        mask_sp[i] = tempmask
        temp2 = sum(temp1)
        center_sp[i,0] = temp2[0]/tempsize
        center_sp[i,1] = temp2[1]/tempsize
        
        for k in range(0,3):
            temp3 = tempmask*labs[:,:,k]
            lab_sp[i,k] = np.sum(temp3)/tempsize
    

    num_bin = 25
    num_bins = num_bin*3

    histo_sp = np.zeros((num_sp,num_bins),dtype="float32")
    
    cv2image = cv2.imread(imgname)
    chans = cv2.split(cv2image)
    
    for i in range(0,num_sp):
        tempmask = img_as_ubyte(mask_sp[i])
        features = np.zeros((3,num_bin))
        
        for (ii, chan) in enumerate(chans):
            hist1d = cv2.calcHist([chan], [0], tempmask, [num_bin], [0, 256])
            features[ii,:] = hist1d.T[0]
            
        temphist = features.flatten()
        histo_sp[i] = temphist /  size_sp[i]
    

    nbins = 18
    img = cv2.imread(imgname,0)
    img = np.float32(img) / 255.0
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=1)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=1)
    
    mag, angle = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    
    hogs = np.zeros((h,w,nbins))
    for i in range(0,h):
        for j in range(0,w):
            hogs[i,j,:] = hogpixel(nbins, mag[i,j], angle[i,j])
    histo_sp2 = np.zeros((num_sp,nbins),dtype="float32")
    tempmask2 = np.expand_dims(mask_sp, -1)
    tempmask2 = np.repeat(tempmask2, nbins, axis=-1)
    for i in range(0,num_sp):
        temphog = np.sum(tempmask2[i]*hogs, axis=1)
        temphog = np.sum(temphog, axis=0)
        histo_sp2[i] = temphog / size_sp[i]
    return num_sp,mask_sp,size_sp,center_sp,lab_sp,histo_sp,histo_sp2
