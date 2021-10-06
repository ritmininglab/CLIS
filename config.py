import numpy as np


from utilMisc import configdims

N = 7200 
Nbatch = 4
N2 = 8
dimdata = 3
dimcls = 2
h1 = 256
w1 = 352
kldiv = h1*w1
Ncore = 63
Ncorebatch = Nbatch

lW = ['b1a','b1b','b2a','b2b','b3a','b3b','b4a','b4b','b5a','b5b',
              'b6a','b6b','b7a','b7b']
lZ = ['b1az','b1bz','b2az','b2bz','b3az','b3bz','b4az','b4bz','b5az','b5bz',
              'b6az','b6bz','b7az','b7bz']
lWlnow = ['b1a','b1b','b2a','b2b','b3a','b3b','b4a','b4b','b5a','b5b',
              'b6a','b6b','b7a','b7b','lnow']

dims = configdims(dimdata, dimcls, 36,56,56)
dims2 = configdims(dimdata, dimcls, 36,56,56)

trainables = []
for i in range(len(dims[0])):
    trainables.append(True)

h2 = 64
w2 = 88
nc = 20
dimsvae = [20,nc,nc,nc,nc,nc,nc,nc,400,400,400,8*11*nc,nc,nc,nc,nc,nc,nc,nc,20]
paramsvae = [Nbatch, 800*h2*w2*3, [8,11,nc]]
