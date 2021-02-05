import numpy as np
import tensorflow as tf

import keras.backend as K
from keras import layers
from keras.layers import Dense,Concatenate,Flatten,Conv2D
from keras.layers import Lambda
from keras.callbacks import Callback

from keras import regularizers



class lpar(layers.Layer): 
    def __init__(self, nwh=3, inch=1,outch=4, mu=0., std=0.5, name='layername'):
        super(lpar, self).__init__(name=name)
        self.nwh = nwh
        self.inch = inch
        self.outch = outch
        p_init = tf.keras.initializers.RandomNormal(mean=mu, stddev=std)
        self.p = tf.Variable(initial_value=p_init(
            shape=(self.nwh,self.nwh,self.inch,self.outch), 
            dtype='float32'), 
            trainable=True)
    def call(self, inputs):
        return self.p
    def compute_output_shape(self, input_shape):
        return (self.nwh,self.nwh,self.inch,self.outch)

def sconv(x, mu, logstd, nwh,inch,outch,strides,nsample):
    rngs = tf.random.normal(shape=(nsample,nwh,nwh,inch,outch))
    out = []
    for i in range(0,nsample):
        tempkernel = mu + tf.math.multiply(rngs[i,:],tf.exp(logstd))
        tempout = tf.nn.conv2d(x, tempkernel, strides, padding='SAME')
        tempout = tf.nn.elu(tempout)
        tempout = tf.expand_dims(tempout, axis=-2)
        out.append(tempout)
    result = tf.concat(out, axis=-2)
    return result
def sconv2(x, mu, logstd, nwh,inch,outch,strides,nsample):
    rngs = tf.random.normal(shape=(nsample,nwh,nwh,inch,outch))
    out = []
    for i in range(0,nsample):
        tempkernel = mu + tf.math.multiply(rngs[i,:],tf.exp(logstd))
        tempout = tf.nn.conv2d(x[:,:,:,i,:], tempkernel, strides, padding='SAME')
        tempout = tf.nn.elu(tempout)
        tempout = tf.expand_dims(tempout, axis=-2)
        out.append(tempout)
    result = tf.concat(out, axis=-2)
    return result
def sconv3(x, mu, logstd, nwh,inch,outch,strides,nsample):
    rngs = tf.random.normal(shape=(nsample,nwh,nwh,inch,outch))
    out = []
    for i in range(0,nsample):
        tempkernel = mu + tf.math.multiply(rngs[i,:],tf.exp(logstd))
        tempout = tf.nn.conv2d(x[:,:,:,0,:], tempkernel, strides, padding='SAME')
        tempout = tf.nn.elu(tempout)
        tempout = tf.expand_dims(tempout, axis=-2)
        out.append(tempout)
    result = tf.concat(out, axis=-2)
    return result

class lmy(layers.Layer):
    def __init__(self, nwh=2, inch=1,outch=2,strides=1,nsample=20, name='layername'):
        super(lmy, self).__init__(name=name)
        self.nwh = nwh
        self.inch = inch
        self.outch = outch
        self.strides = strides
        self.nsample = nsample
    def call(self, x):
        return  sconv(x[0],x[1],x[2],self.nwh,self.inch,
                             self.outch,[1, 1, 1, 1],self.nsample)
    def compute_output_shape(self, inshape):
        return (inshape[0][0],inshape[0][1],inshape[0][2], self.nsample, self.outch)
class lmy1(layers.Layer):
    def __init__(self, nwh=2, inch=1,outch=2,strides=1,nsample=20, name='layername'):
        super(lmy1, self).__init__(name=name)
        self.nwh = nwh
        self.inch = inch
        self.outch = outch
        self.strides = strides
        self.nsample = nsample
    def call(self, x):
        temp = sconv(x[0],x[1],x[2],self.nwh,self.inch,
                             self.outch,[1, 1, 1, 1],self.nsample)
        return  tf.math.reduce_mean(temp, axis=-2, keepdims=True)
    def compute_output_shape(self, inshape):
        return (inshape[0][0],inshape[0][1],inshape[0][2], 1, self.outch)


class lmy2(layers.Layer):
    def __init__(self, nwh=2, inch=1,outch=2,strides=1,nsample=20, name='layername'):
        super(lmy2, self).__init__(name=name)
        self.nwh = nwh
        self.inch = inch
        self.outch = outch
        self.strides = strides
        self.nsample = nsample
    def call(self, x):
        return  sconv2(x[0],x[1],x[2],self.nwh,self.inch,
                             self.outch,[1, 1, 1, 1],self.nsample)
    def compute_output_shape(self, inshape):
        return (inshape[0][0],inshape[0][1],inshape[0][2], self.nsample, self.outch)
class lmy3(layers.Layer):
    def __init__(self, nwh=2, inch=1,outch=2,strides=1,nsample=20, name='layername'):
        super(lmy3, self).__init__(name=name)
        self.nwh = nwh
        self.inch = inch
        self.outch = outch
        self.strides = strides
        self.nsample = nsample
    def call(self, x):
        temp = sconv3(x[0],x[1],x[2],self.nwh,self.inch,
                             self.outch,[1, 1, 1, 1],self.nsample)
        return  tf.math.reduce_mean(temp,axis=-2, keepdims=True)
    def compute_output_shape(self, inshape):
        return (inshape[0][0],inshape[0][1],inshape[0][2], 1, self.outch)


    
class lpar2b(layers.Layer):
    def __init__(self, indim=1, initialvalue=1, name='layername'):
        super(lpar2b, self).__init__(name=name)
        a_init = tf.constant_initializer(initialvalue)
        self.p = tf.Variable(initial_value=a_init(shape=(1,1,1,indim), 
                                                  dtype='float32'), 
                             trainable=True)
        self.indim = indim
    def call(self, inputs):
        return tf.exp(self.p)
    def compute_output_shape(self, input_shape):
        return (1,1,1,self.indim)

class lpar3b(layers.Layer):
    def __init__(self, indim=1, initialvalue=1, name='layername'):
        super(lpar3b, self).__init__(name=name)
        a_init = tf.constant_initializer(initialvalue)
        self.p = tf.Variable(initial_value=a_init(shape=(1,1,1,indim), 
                                                  dtype='float32'), 
                             trainable=True)
        self.indim = indim
    def call(self, inputs):
        return tf.sigmoid(self.p)
    def compute_output_shape(self, input_shape):
        return (1,1,1,self.indim)
class lmeanp(layers.Layer):
    def __init__(self, name='layername'):
        super(lmeanp, self).__init__(name=name)
    def call(self, inputs):
        return tf.math.reduce_mean(inputs, axis=-2, keepdims=True)
    def compute_output_shape(self, input_shape):
        return (1,1,1, input_shape[3])

class skuma(layers.Layer):
    
    def __init__(self, outdim=10, nsample=20, name='layername'):
        super(skuma, self).__init__(name=name)
        self.rng = tf.random.uniform(shape=(1,1,nsample, outdim))
        self.outdim = outdim
        self.nsample = nsample
    def call(self, x):
        alpha = x[0]
        beta = x[1]
        return tf.math.pow(1-tf.math.pow(self.rng, 1/beta), 1/alpha)
    def compute_output_shape(self, input_shape):
        return (1,1,self.nsample, self.outdim)

class spi(layers.Layer):
    
    def __init__(self, outdim=10, nsample=20, name='layername'):
        super(spi, self).__init__(name=name)
        self.outdim = outdim
        self.nsample = nsample
    def call(self, inputs):
        return tf.math.cumprod(inputs, axis=-1)
    def compute_output_shape(self, input_shape):
        return (1,1,self.nsample, self.outdim)

class spi2(layers.Layer):
    def __init__(self, outdim=10, nsample=20, name='layername'):
        super(spi2, self).__init__(name=name)
        self.outdim = outdim
    def call(self, x):
        alpha = x[0]
        beta = x[1]
        return tf.math.cumprod(tf.divide(alpha,alpha+beta), axis=-1)
    def compute_output_shape(self, input_shape):
        return (1,1,1, self.outdim)


class sber2(layers.Layer):
    def __init__(self, nbatch=9, h=5,w=5,outch=10, nsample=20, temperature=1, name='layername'):
        super(sber2, self).__init__(name=name)
        self.w=w
        self.h=h
        self.outch = outch
        self.nsample = nsample
        self.nbatch = nbatch
        self.rng = tf.random.uniform(shape=(1,1,nsample,outch))
        self.temperature = temperature
    def call(self, inputs):
        temp = tf.tile(inputs, [1,1,self.nsample,1])
        term1 = tf.math.log(temp)-tf.math.log(1-temp)
        term2 = tf.math.log(self.rng)-tf.math.log(1-self.rng)
        softbernoulli = 1/(1+tf.exp(-1/self.temperature*(term1+term2)))
        return tf.tile(tf.expand_dims(softbernoulli, 0), [self.nbatch, self.h, self.w,1,1])
    def compute_output_shape(self, inshape):
        return (self.nbatch, self.h, self.w, self.nsample, self.outch)
class sber1(layers.Layer):
    def __init__(self, nbatch=9, h=5,w=5,outch=10, nsample=20, temperature=1, name='layername'):
        super(sber1, self).__init__(name=name)
        self.w=w
        self.h=h
        self.outch = outch
        self.nsample = nsample
        self.nbatch = nbatch
        self.temperature = temperature
    def call(self, inputs):
        term1 = tf.math.log(inputs)-tf.math.log(1-inputs)
        softbernoulli = 1/(1+tf.exp(-(term1)/self.temperature))
        return tf.tile(tf.expand_dims(softbernoulli, 0), [self.nbatch, self.h, self.w,1,1])
    def compute_output_shape(self, inshape):
        return (self.nbatch, self.h, self.w, 1, self.outch)


class llg(layers.Layer):
    def __init__(self, nbatch=9, nsample=20, priormu=0, priorlogstd=0, name='layername'):
        super(llg, self).__init__(name=name)
        self.nsample = nsample
        self.nbatch = nbatch
        self.priormu = priormu
        self.priorlogstd = priorlogstd
    def call(self, x):
        term1 = self.priorlogstd -x[1] + 0.5*tf.exp(2*x[1] - 2*self.priorlogstd)
        term2 = 0.5*tf.divide(tf.square(x[0] - self.priormu), tf.exp(2*self.priorlogstd))
        sumkl = tf.reduce_sum(term1+term2)
        return  sumkl * tf.ones([self.nbatch, 1])
    def compute_output_shape(self, input_shape):
        return (self.nbatch, 1) 

class ll3b(layers.Layer):
    def __init__(self, nbatch=9, nsample=20, name='layername'):
        super(ll3b, self).__init__(name=name)
        self.nsample = nsample
        self.nbatch = nbatch
    def call(self, x):
        temp = tf.tile(x[0], [1,1,self.nsample, 1])
        term1 = tf.multiply(temp, tf.math.log(temp) - tf.math.log(x[1]))
        term2 = tf.multiply(1-temp, tf.math.log(1-temp) - tf.math.log(1-x[1]))
        sumkl = tf.reduce_sum(term1+term2)
        return  sumkl * tf.ones([self.nbatch, 1])
    def compute_output_shape(self, input_shape):
        return (self.nbatch, 1) 
class ll3b2(layers.Layer):
    def __init__(self, nbatch=9, nsample=20, prior=1, name='layername'):
        super(ll3b2, self).__init__(name=name)
        self.nsample = nsample
        self.nbatch = nbatch
        self.prior = prior
    def call(self, x):
        term1 = tf.multiply(x[0], tf.math.log(x[0]) - tf.math.log(self.prior))
        term2 = tf.multiply(1-x[0], tf.math.log(1-x[0]) - tf.math.log(1-self.prior))
        sumkl = tf.reduce_sum(term1+term2) 
        return  sumkl * tf.ones([self.nbatch, 1])
    def compute_output_shape(self, input_shape):
        return (self.nbatch, 1) 
class ll3b1(layers.Layer):
    def __init__(self, nbatch=9, nsample=20, name='layername'):
        super(ll3b1, self).__init__(name=name)
        self.nsample = nsample
        self.nbatch = nbatch
    def call(self, x):
        term1 = tf.multiply(x[0], tf.math.log(x[0]) - tf.math.log(x[1]))
        term2 = tf.multiply(1-x[0], tf.math.log(1-x[0]) - tf.math.log(1-x[1]))
        sumkl = tf.reduce_sum(term1+term2)
        return  sumkl * tf.ones([self.nbatch, 1])
    def compute_output_shape(self, input_shape):
        return (self.nbatch, 1) 



class lmean(layers.Layer):
    def __init__(self, name='layername'):
        super(lmean, self).__init__(name=name)
    def call(self, x):
        return  tf.math.reduce_mean(x,axis=-2)
    def compute_output_shape(self, inshape):
        return (inshape[0],inshape[1],inshape[2], inshape[4])

class ll2b(layers.Layer):
    def __init__(self, nbatch=9, priora=1, priorb=1, name='layername'):
        super(ll2b, self).__init__(name=name)
        self.nbatch = nbatch
        self.priora = priora
        self.priorb = priorb
    def call(self, x):
        avec = x[0]
        bvec = x[1]
        abvec = x[0]+x[1]
        term0 = tf.math.lgamma(self.priora)+tf.math.lgamma(self.priorb)-tf.math.lgamma(self.priora+self.priorb)
        term1 = -tf.math.lgamma(avec)-tf.math.lgamma(bvec)+tf.math.lgamma(abvec)
        term2 = tf.multiply(avec-self.priora, tf.math.digamma(avec))
        term3 = tf.multiply(bvec-self.priorb, tf.math.digamma(bvec))
        term4 = tf.multiply(-abvec+self.priora+self.priorb, tf.math.digamma(abvec))
        sumkl = tf.reduce_sum(term0+term1+term2+term3+term4)
        return  sumkl * tf.ones([self.nbatch, 1])
    def compute_output_shape(self, input_shape):
        return (self.nbatch, 1) 


class lla(layers.Layer):
    def __init__(self, nbatch=9, nmember=3, name='layername'):
        super(lla, self).__init__(name=name)
        self.nbatch = nbatch
        self.nmember = nmember
    def call(self, x):
        output = x[0]
        for i in range(1,self.nmember):
            output = output + x[i]
        return  output
    def compute_output_shape(self, input_shape):
        return (self.nbatch, 1)



def samplep(x, nsample):
    out = []
    for i in range(0,nsample):
        tempout = tf.nn.max_pool2d(x[:,:,:,i,:],ksize=(2,2),strides=(2,2),padding='VALID')
        tempout = tf.expand_dims(tempout, axis=-2) 
        out.append(tempout)
    result = tf.concat(out, axis=-2)
    return result
class lp(layers.Layer):
    def __init__(self, nsample=20, name='layername'):
        super(lp, self).__init__(name=name)
        self.nsample = nsample
    def call(self, x):
        return  samplep(x, self.nsample)
    def compute_output_shape(self, inshape):
        return (inshape[0],inshape[1]>>1,inshape[2]>>1, inshape[3], inshape[4])

def sampleu(x, nsample):
    out = []
    for i in range(0,nsample):
        tempout = tf.keras.backend.resize_images(x[:,:,:,i,:],2,2,data_format='channels_last')
        tempout = tf.expand_dims(tempout, axis=-2) 
        out.append(tempout)
    result = tf.concat(out, axis=-2)
    return result
class lup(layers.Layer):
    def __init__(self, nsample=20, name='layername'):
        super(lup, self).__init__(name=name)
        self.nsample = nsample
    def call(self, x):
        return  sampleu(x, self.nsample)
    def compute_output_shape(self, inshape):
        return (inshape[0],inshape[1]<<1,inshape[2]<<1, inshape[3], inshape[4])

class sgauss(layers.Layer):
    def __init__(self, batchsize=1, h=1, w=2, k=1, nsample=10, name='layername'):
        super(sgauss, self).__init__(name=name)
        self.k = k
        self.nsample = nsample
        self.rng = tf.random.normal(shape=(batchsize,h,w,k,nsample))
        self.batchsize = batchsize
        self.w=w
        self.h=h
    def call(self, x):
        mu = tf.tile(tf.expand_dims(x[0],-1),[1,1,1,1, self.nsample])
        logvar = tf.tile(tf.expand_dims(x[1],-1),[1,1,1,1, self.nsample])
        return mu + tf.multiply(tf.exp(0.5*logvar), self.rng)        
        
    def compute_output_shape(self, input_shape):
        return (self.batchsize, self.h, self.w, self.k, self.nsample)

class sgauss2(layers.Layer):
    def __init__(self, batchsize=1, h=1, w=2, k=1, nsample=10, name='layername'):
        super(sgauss2, self).__init__(name=name)
        self.k = k
        self.nsample = nsample
        self.rng = tf.random.normal(shape=(batchsize,h,w,k,nsample))
        self.batchsize = batchsize
        self.w=w
        self.h=h
    def call(self, x):
        mu = tf.tile(tf.expand_dims(x[0],-1),[1,1,1,1, self.nsample])
        logvar = tf.tile(tf.expand_dims(x[1],-1),[1,1,1,1, self.nsample])
        temp1 = mu + tf.multiply(tf.exp(0.5*logvar), self.rng)
        temp2 = tf.nn.softmax(temp1, axis=-2)
        return tf.reduce_mean(temp2, axis=-1, keepdims=False)
        
    def compute_output_shape(self, input_shape):
        return (self.batchsize, self.h, self.w, self.k)
