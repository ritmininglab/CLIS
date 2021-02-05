from __future__ import division
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Reshape, MaxPooling2D, Flatten, RepeatVector, UpSampling2D, Concatenate, Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda, Multiply, Add
from tensorflow.keras.layers import Softmax, ReLU
from tensorflow.keras.layers import BatchNormalization

initialvar = -5.
small = 1e-6

class lexpand(layers.Layer):
    def __init__(self, nsample, name='layername'):
        super(lexpand, self).__init__(name=name)
        self.nsample = nsample
    def call(self, x):
        return tf.tile(tf.expand_dims(x, 1), [1, self.nsample, 1,1,1])
    def compute_output_shape(self, ins):
        return (ins[0], self.nsample, ins[1], ins[2], ins[3])

class lmpool(layers.Layer):
    def __init__(self, nsample, name='rname'):
        super(lmpool, self).__init__(name=name)
        self.nsample = nsample
    def call(self, x):
        out = []
        for i in range(0,self.nsample):
            tempout = tf.nn.max_pool2d(x[:,i,:],ksize=(2,2),strides=(2,2),padding='VALID')
            out.append(tempout)
        stacked = tf.stack(out, axis=1)
        return  stacked
    def compute_output_shape(self, ins):
        return (ins[0],ins[1],ins[2]>>1, ins[3]>>1, ins[4])
class lupool(layers.Layer):
    def __init__(self, nsample, name='name'):
        super(lupool, self).__init__(name=name)
        self.nsample = nsample
    def call(self, x):
        out = []
        for i in range(0,self.nsample):
            tempout = tf.keras.backend.resize_images(x[:,i,:,:,:],2,2,data_format='channels_last')
            out.append(tempout)
        stacked = tf.stack(out, axis=1)
        return  stacked
    def compute_output_shape(self, ins):
        return (ins[0],ins[1],ins[2]<<1, ins[3]<<1, ins[4])

class lbconvRN(layers.Layer):
    def __init__(self, dms, prs, trainables, idx, kld, wh, nsample, name='name'):
        super(lbconvRN, self).__init__(name=name)
        self.d1 = dms[0][idx]
        self.d2 = dms[1][idx]
        self.trainable = trainables[idx]
        self.wm0 = prs[idx][2]
        self.wv0 = prs[idx][3]
        self.bm0 = prs[idx][4]
        self.bv0 = prs[idx][5]
        self.wh = wh
        self.kld = kld
        self.nsample = nsample
    def build(self, ins):
        w_init = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
        v_init = tf.keras.initializers.Constant(initialvar)
        self.w = self.add_weight("w", trainable=self.trainable,
                                 initializer=w_init, shape=[self.wh,self.wh,self.d1, self.d2])
        self.wv = self.add_weight("wv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.wh,self.wh,self.d1, self.d2])
        self.b = self.add_weight("b", trainable=self.trainable,
                                 initializer=w_init, shape=[self.d2])
        self.bv = self.add_weight("bv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.d2])
    def call(self, x):
        rngw = tf.random.truncated_normal([self.nsample,self.wh,self.wh,self.d1,self.d2])
        rngb = tf.random.truncated_normal([self.nsample,self.d2])
        resultlist = []
        for i in range(0,self.nsample):
            wnow = self.w + tf.math.multiply(rngw[i,:], tf.exp(self.wv))
            bnow = self.b + tf.math.multiply(rngb[i,:], tf.exp(self.bv))
            tempout = tf.add(tf.nn.conv2d(x[:,i,:], wnow, [1,1,1,1], padding='SAME'), bnow)
            resultlist.append(tempout)
        stacked = tf.stack(resultlist, axis=1)
        output = tf.nn.relu(stacked)
        
        term0 = -0.5*self.wh*self.wh*self.d1*self.d2
        term1 = 0.5*tf.reduce_sum(np.log(self.wv0) - self.wv)
        term2 = 0.5*tf.reduce_sum((tf.exp(self.wv) + (self.w - self.wm0)**2) / self.wv0)
        term0b = -0.5*self.d2
        term1b = 0.5*tf.reduce_sum(np.log(self.bv0) - self.bv)
        term2b = 0.5*tf.reduce_sum((tf.exp(self.bv) + (self.b - self.bm0)**2) / self.bv0)
        sumkl = term1 + term2 + term0 + term1b + term2b + term0b
        self.add_loss(sumkl / self.kld)
        
        return output
    def compute_output_shape(self, ins):
        return (ins[0], ins[1], ins[2], ins[3], self.d2)
class lbconvN(layers.Layer):
    def __init__(self, dms, prs, trainables, idx, kld, wh, nsample, name='name'):
        super(lbconvN, self).__init__(name=name)
        self.d1 = dms[0][idx]
        self.d2 = dms[1][idx]
        self.trainable = trainables[idx]
        self.wm0 = prs[idx][2]
        self.wv0 = prs[idx][3]
        self.bm0 = prs[idx][4]
        self.bv0 = prs[idx][5]
        self.wh = wh
        self.kld = kld
        self.nsample = nsample
    def build(self, ins):
        w_init = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
        v_init = tf.keras.initializers.Constant(initialvar)
        self.w = self.add_weight("w", trainable=self.trainable,
                                 initializer=w_init, shape=[self.wh,self.wh,self.d1, self.d2])
        self.wv = self.add_weight("wv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.wh,self.wh,self.d1, self.d2])
        self.b = self.add_weight("b", trainable=self.trainable,
                                 initializer=w_init, shape=[self.d2])
        self.bv = self.add_weight("bv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.d2])
    def call(self, x):
        rngw = tf.random.truncated_normal([self.nsample,self.wh,self.wh,self.d1,self.d2])
        rngb = tf.random.truncated_normal([self.nsample,self.d2])
        resultlist = []
        for i in range(0,self.nsample):
            wnow = self.w + tf.math.multiply(rngw[i,:], tf.exp(self.wv))
            bnow = self.b + tf.math.multiply(rngb[i,:], tf.exp(self.bv))
            tempout = tf.add(tf.nn.conv2d(x[:,i,:], wnow, [1,1,1,1], padding='SAME'), bnow)
            resultlist.append(tempout)
        stacked = tf.stack(resultlist, axis=1)
        
        term0 = -0.5*self.wh*self.wh*self.d1*self.d2
        term1 = 0.5*tf.reduce_sum(np.log(self.wv0) - self.wv)
        term2 = 0.5*tf.reduce_sum((tf.exp(self.wv) + (self.w - self.wm0)**2) / self.wv0)
        term0b = -0.5*self.d2
        term1b = 0.5*tf.reduce_sum(np.log(self.bv0) - self.bv)
        term2b = 0.5*tf.reduce_sum((tf.exp(self.bv) + (self.b - self.bm0)**2) / self.bv0)
        sumkl = term1 + term2 + term0 + term1b + term2b + term0b
        self.add_loss(sumkl / self.kld)
        
        return stacked
    def compute_output_shape(self, ins):
        return (ins[0], ins[1], ins[2], ins[3], self.d2)

class lbconvR(layers.Layer):
    def __init__(self, dms, prs, trainables, idx, kld, wh, name='name'):
        super(lbconvR, self).__init__(name=name)
        self.d1 = dms[0][idx]
        self.d2 = dms[1][idx]
        self.trainable = trainables[idx]
        self.wm0 = prs[idx][2]
        self.wv0 = prs[idx][3]
        self.bm0 = prs[idx][4]
        self.bv0 = prs[idx][5]
        self.wh = wh
        self.kld = kld
    def build(self, ins):
        w_init = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
        v_init = tf.keras.initializers.Constant(initialvar)
        self.w = self.add_weight("w", trainable=self.trainable,
                                 initializer=w_init, shape=[self.wh,self.wh,self.d1, self.d2])
        self.wv = self.add_weight("wv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.wh,self.wh,self.d1, self.d2])
        self.b = self.add_weight("b", trainable=self.trainable,
                                 initializer=w_init, shape=[self.d2])
        self.bv = self.add_weight("bv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.d2])
    def call(self, x):
        rngw = tf.random.truncated_normal([self.wh,self.wh,self.d1,self.d2])
        rngb = tf.random.truncated_normal([self.d2])
        wnow = self.w + tf.math.multiply(rngw, tf.exp(self.wv))
        bnow = self.b + tf.math.multiply(rngb, tf.exp(self.bv))
        tempout = tf.add(tf.nn.conv2d(x, wnow, [1,1,1,1], padding='SAME'), bnow)
        output = tf.nn.relu(tempout)
        
        term0 = -0.5*self.wh*self.wh*self.d1*self.d2
        term1 = 0.5*tf.reduce_sum(np.log(self.wv0) - self.wv)
        term2 = 0.5*tf.reduce_sum((tf.exp(self.wv) + (self.w - self.wm0)**2) / self.wv0)
        term0b = -0.5*self.d2
        term1b = 0.5*tf.reduce_sum(np.log(self.bv0) - self.bv)
        term2b = 0.5*tf.reduce_sum((tf.exp(self.bv) + (self.b - self.bm0)**2) / self.bv0)
        sumkl = term1 + term2 + term0 + term1b + term2b + term0b
        self.add_loss(sumkl / self.kld)
        
        return output
    def compute_output_shape(self, ins):
        return (ins[0], ins[1], ins[2], self.d2)
class lbconv(layers.Layer):
    def __init__(self, dms, prs, trainables, idx, kld, wh, name='name'):
        super(lbconv, self).__init__(name=name)
        self.d1 = dms[0][idx]
        self.d2 = dms[1][idx]
        self.trainable = trainables[idx]
        self.wm0 = prs[idx][2]
        self.wv0 = prs[idx][3]
        self.bm0 = prs[idx][4]
        self.bv0 = prs[idx][5]
        self.wh = wh
        self.kld = kld
    def build(self, ins):
        w_init = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
        v_init = tf.keras.initializers.Constant(initialvar)
        self.w = self.add_weight("w", trainable=self.trainable,
                                 initializer=w_init, shape=[self.wh,self.wh,self.d1, self.d2])
        self.wv = self.add_weight("wv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.wh,self.wh,self.d1, self.d2])
        self.b = self.add_weight("b", trainable=self.trainable,
                                 initializer=w_init, shape=[self.d2])
        self.bv = self.add_weight("bv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.d2])
    def call(self, x):
        rngw = tf.random.truncated_normal([self.wh,self.wh,self.d1,self.d2])
        rngb = tf.random.truncated_normal([self.d2])
        wnow = self.w + tf.math.multiply(rngw, tf.exp(self.wv))
        bnow = self.b + tf.math.multiply(rngb, tf.exp(self.bv))
        tempout = tf.add(tf.nn.conv2d(x, wnow, [1,1,1,1], padding='SAME'), bnow)
        
        term0 = -0.5*self.wh*self.wh*self.d1*self.d2
        term1 = 0.5*tf.reduce_sum(np.log(self.wv0) - self.wv)
        term2 = 0.5*tf.reduce_sum((tf.exp(self.wv) + (self.w - self.wm0)**2) / self.wv0)
        term0b = -0.5*self.d2
        term1b = 0.5*tf.reduce_sum(np.log(self.bv0) - self.bv)
        term2b = 0.5*tf.reduce_sum((tf.exp(self.bv) + (self.b - self.bm0)**2) / self.bv0)
        sumkl = term1 + term2 + term0 + term1b + term2b + term0b
        self.add_loss(sumkl / self.kld)
        
        return tempout
    def compute_output_shape(self, ins):
        return (ins[0], ins[1], ins[2], self.d2)

class lbz(layers.Layer):
    def __init__(self, dms, prs, trainables, idx, kld, name='layername'):
        super(lbz, self).__init__(name=name)
        self.d1 = dms[0][idx]
        self.d2 = dms[1][idx]
        self.pra = prs[idx][0]
        self.prb = prs[idx][1]
        self.kld = kld
        self.trainable = trainables[idx]
        self.temperature = 1
    def build(self, ins): 
        self.nbatch = ins[0]
        a_init = tf.keras.initializers.Constant(5./40)
        b_init = tf.keras.initializers.Constant(1.)
        z_init = tf.keras.initializers.Constant(2.)
        self.a = self.add_weight("a",trainable=self.trainable,
                                 initializer=a_init,
                                  shape=[self.d2],)
        self.b = self.add_weight("b",trainable=self.trainable,
                                 initializer=b_init,
                                  shape=[self.d2],)
        self.pnk = self.add_weight("pnk",trainable=self.trainable,
                                 initializer=z_init,
                                  shape=[self.nbatch,1,1,self.d2],)
    def call(self, x):
        alpha = tf.nn.softplus(self.a)
        beta = tf.nn.softplus(self.b)
        epsab = tf.random.uniform(shape=tf.shape(alpha), minval=0.01, maxval=0.99)
        kuma = tf.math.pow(1 - tf.math.pow(epsab, 1/beta), 1/alpha)
        priorlogp = tf.math.log(kuma)
        
        pnk = tf.sigmoid(self.pnk)
        epsp = tf.random.uniform(shape=tf.shape(pnk), minval=0.01, maxval=0.99)
        temp2 = tf.math.log(epsp)-tf.math.log(1-epsp)
        logpnk = tf.math.log(pnk+small) - tf.math.log(1-pnk+small)
        softber = 1/(1 + tf.exp(-(logpnk+temp2)/self.temperature))
        
        ab = alpha+beta
        term0 = tf.math.lgamma(self.pra)+tf.math.lgamma(self.prb)-tf.math.lgamma(self.pra+self.prb)
        term1 = -tf.math.lgamma(alpha)-tf.math.lgamma(beta)+tf.math.lgamma(ab)
        term2 = tf.multiply(alpha-self.pra, tf.math.digamma(alpha))
        term3 = tf.multiply(beta-self.prb, tf.math.digamma(beta))
        term4 = tf.multiply(-ab+self.pra+self.prb, tf.math.digamma(ab))
        sumklbeta = tf.reduce_sum(term0+term1+term2+term3+term4)
        
        self.add_loss( sumklbeta / (self.kld) )
        
        temp = pnk
        term11 = tf.multiply(temp, tf.math.log(temp + small) - tf.math.log(tf.exp(priorlogp) + small))
        term12 = tf.multiply(1-temp, tf.math.log(1-temp + small) - tf.math.log(1-tf.exp(priorlogp) + small))
        sumklber = tf.reduce_sum(term11+term12)
        
        self.add_loss( sumklber / (self.kld) )
        
        result = tf.multiply(x, softber)
        return result
    def compute_output_shape(self, ins):
        return (ins[0], ins[1], ins[2], self.d2)
class lbzn(layers.Layer):
    def __init__(self, dms, prs, trainables, idx, kld, nsample, name='layername'):
        super(lbz, self).__init__(name=name)
        self.d1 = dms[0][idx]
        self.d2 = dms[1][idx]
        self.pra = prs[idx][0]
        self.prb = prs[idx][1]
        self.kld = kld
        self.trainable = trainables[idx]
        self.temperature = 1
        self.nsample = nsample
    def build(self, ins): 
        self.nbatch = ins[0]
        a_init = tf.keras.initializers.Constant(5./40)
        b_init = tf.keras.initializers.Constant(1.)
        z_init = tf.keras.initializers.Constant(2.)
        self.a = self.add_weight("a",trainable=self.trainable,
                                 initializer=a_init,
                                  shape=[self.d2],)
        self.b = self.add_weight("b",trainable=self.trainable,
                                 initializer=b_init,
                                  shape=[self.d2],)
        self.pnk = self.add_weight("pnk",trainable=self.trainable,
                                 initializer=z_init,
                                  shape=[self.nbatch,1,1,1,self.d2],)
    def call(self, x):
        alpha = tf.nn.softplus(self.a)
        beta = tf.nn.softplus(self.b)
        epsab = tf.random.uniform(shape=tf.shape(alpha), minval=0.01, maxval=0.99)
        kuma = tf.math.pow(1 - tf.math.pow(epsab, 1/beta), 1/alpha)
        """
        temp1 = tf.math.log(kuma + small)
        priorlogp = tf.math.cumsum(temp1, axis=-1)
        """
        priorlogp = tf.math.log(kuma)
        
        pnk = tf.sigmoid(self.pnk)
        epsp = tf.random.uniform(shape=[self.nbatch,self.nsample,1,1,self.d2], minval=0.01, maxval=0.99)
        temp2 = tf.math.log(epsp)-tf.math.log(1-epsp)
        logpnk = tf.math.log(pnk+small) - tf.math.log(1-pnk+small)
        softber = 1/(1 + tf.exp(-(logpnk+temp2)/self.temperature))
        
        ab = alpha+beta
        term0 = tf.math.lgamma(self.pra)+tf.math.lgamma(self.prb)-tf.math.lgamma(self.pra+self.prb)
        term1 = -tf.math.lgamma(alpha)-tf.math.lgamma(beta)+tf.math.lgamma(ab)
        term2 = tf.multiply(alpha-self.pra, tf.math.digamma(alpha))
        term3 = tf.multiply(beta-self.prb, tf.math.digamma(beta))
        term4 = tf.multiply(-ab+self.pra+self.prb, tf.math.digamma(ab))
        sumklbeta = tf.reduce_sum(term0+term1+term2+term3+term4)
        
        self.add_loss( sumklbeta / (self.kld) )
        
        temp = pnk
        term11 = tf.multiply(temp, tf.math.log(temp + small) - tf.math.log(tf.exp(priorlogp) + small))
        term12 = tf.multiply(1-temp, tf.math.log(1-temp + small) - tf.math.log(1-tf.exp(priorlogp) + small))
        sumklber = tf.reduce_sum(term11+term12)
        
        self.add_loss( sumklber / (self.kld) )
        
        result = tf.multiply(x, softber)
        return result
    def compute_output_shape(self, ins):
        return (ins[0], ins[1], ins[2], ins[3], self.d2)
class fcsig(layers.Layer):
    def __init__(self, dms, priors, trainables, idx, kldiv, name='name'):
        super(fcsig, self).__init__(name=name)
        self.dm1 = dms[0][idx]
        self.dm2 = dms[1][idx]
        self.trainable = trainables[idx]
        self.kldiv = kldiv
    def build(self, input_shape):
        mu_init = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
        self.w = self.add_weight("w",trainable=self.trainable,
                                 initializer=mu_init,
                                  shape=[self.dm1, self.dm2],)
        self.b = self.add_weight("b",trainable=self.trainable,
                                 initializer=mu_init,
                                  shape=[self.dm2,],)
    def call(self, x):
        result = tf.add(tf.matmul(x, self.w), self.b)
        l2 = tf.reduce_sum(tf.square(self.w))+tf.reduce_sum(tf.square(self.b))
        self.add_loss(l2 / (1e3*self.kldiv))
        
        return tf.math.sigmoid(result)
    def compute_output_shape(self, ins):
        return (ins[0], self.dm2)
class lstore(layers.Layer):
    def __init__(self, dms, prs, trainables, idx, kld,ndata,name='layername'):
        super(lstore, self).__init__(name=name)
        self.ndata = ndata
        self.d1 = dms[0][idx]
        self.d2 = dms[1][idx]
        self.pra = prs[idx][0]
        self.prb = prs[idx][1]
        self.kld = kld
        self.trainable = trainables[idx]
        self.temperature = 1
    def build(self, ins): 
        self.nbatch = ins[0]
        a_init = tf.keras.initializers.Constant(5./40)
        b_init = tf.keras.initializers.Constant(1.)
        z_init = tf.keras.initializers.Constant(2.)
        self.a = self.add_weight("a",trainable=self.trainable,
                                 initializer=a_init,
                                  shape=[self.d2],)
        self.b = self.add_weight("b",trainable=self.trainable,
                                 initializer=b_init,
                                  shape=[self.d2],)
        self.pall = self.add_weight("pall",trainable=self.trainable,
                                 initializer=z_init,
                                  shape=[self.ndata,self.d2],)
    def call(self, xs):
        x = xs[0]
        auxiliary = xs[1]
        
        alpha = tf.nn.softplus(self.a)
        beta = tf.nn.softplus(self.b)
        epsab = tf.random.uniform(shape=tf.shape(alpha), minval=0.01, maxval=0.99)
        kuma = tf.math.pow(1 - tf.math.pow(epsab, 1/beta), 1/alpha)
        priorlogp = tf.math.log(kuma)
        
        pnk = tf.matmul(auxiliary, self.pall)
        pnk = tf.sigmoid(pnk)
        epsp = tf.random.uniform(shape=tf.shape(pnk), minval=0.01, maxval=0.99)
        temp2 = tf.math.log(epsp)-tf.math.log(1-epsp)
        logpnk = tf.math.log(pnk+small) - tf.math.log(1-pnk+small)
        softber = 1/(1 + tf.exp(-(logpnk+temp2)/self.temperature))
        
        ab = alpha+beta
        term0 = tf.math.lgamma(self.pra)+tf.math.lgamma(self.prb)-tf.math.lgamma(self.pra+self.prb)
        term1 = -tf.math.lgamma(alpha)-tf.math.lgamma(beta)+tf.math.lgamma(ab)
        term2 = tf.multiply(alpha-self.pra, tf.math.digamma(alpha))
        term3 = tf.multiply(beta-self.prb, tf.math.digamma(beta))
        term4 = tf.multiply(-ab+self.pra+self.prb, tf.math.digamma(ab))
        sumklbeta = tf.reduce_sum(term0+term1+term2+term3+term4)
        
        self.add_loss( sumklbeta / (self.kld) )
        
        temp = pnk
        term11 = tf.multiply(temp, tf.math.log(temp + small) - tf.math.log(tf.exp(priorlogp) + small))
        term12 = tf.multiply(1-temp, tf.math.log(1-temp + small) - tf.math.log(1-tf.exp(priorlogp) + small))
        sumklber = tf.reduce_sum(term11+term12)
        
        self.add_loss( sumklber / (self.kld) )
        
        softber = tf.expand_dims(tf.expand_dims(softber, axis=1), axis=1)
        result = tf.multiply(x, softber)
        return result
    def compute_output_shape(self, ins):
        return (ins[0][0], ins[0][1], ins[0][2], self.d2)

def myNNbnZ_noRes(datas, dms, prs, trainables, kld, mmt, bn, ndata):
    
    b1a = lbconvR(dms, prs, trainables, 0, kld,3,name='b1a')(datas[0])
    b1az = lstore(dms, prs, trainables, 0, kld,ndata,name='b1az')([b1a, datas[1]])
    b1b = lbconv(dms, prs, trainables, 1, kld,3,name='b1b')(b1az)
    d1b = BatchNormalization(momentum=mmt, trainable=bn, name='d1b')(b1b, training=bn) 
    b1br = ReLU(name='b1br')(d1b)
    b1bz = lstore(dms, prs, trainables, 1, kld,ndata,name='b1bz')([b1br, datas[1]])
    p1 = MaxPooling2D((2, 2), strides=(2, 2), name='p1')(b1bz)

    b2a = lbconv(dms, prs, trainables, 2, kld,3,name='b2a')(p1)
    d2a = BatchNormalization(momentum=mmt, trainable=bn, name='d2a')(b2a, training=bn) 
    b2ar = ReLU(name='b2ar')(d2a)
    b2az = lstore(dms, prs, trainables, 2, kld,ndata,name='b2az')([b2ar, datas[1]])
    b2b = lbconv(dms, prs, trainables, 3, kld,3,name='b2b')(b2az)
    d2b = BatchNormalization(momentum=mmt, trainable=bn, name='d2b')(b2b, training=bn) 
    b2br = ReLU(name='b2br')(d2b)
    b2bz = lstore(dms, prs, trainables, 3, kld,ndata,name='b2bz')([b2br, datas[1]])
    p2 = MaxPooling2D((2, 2), strides=(2, 2), name='p2')(b2bz)

    b3a = lbconv(dms, prs, trainables, 4, kld,3,name='b3a')(p2)
    d3a = BatchNormalization(momentum=mmt, trainable=bn, name='d3a')(b3a, training=bn) 
    b3ar = ReLU(name='b3ar')(d3a)
    b3az = lstore(dms, prs, trainables, 4, kld,ndata,name='b3az')([b3ar, datas[1]])
    b3b = lbconv(dms, prs, trainables, 5, kld,3,name='b3b')(b3az)
    d3b = BatchNormalization(momentum=mmt, trainable=bn, name='d3b')(b3b, training=bn) 
    b3br = ReLU(name='b3br')(d3b)
    b3bz = lstore(dms, prs, trainables, 5, kld,ndata,name='b3bz')([b3br, datas[1]])
    p3 = UpSampling2D(size=(2, 2), name='p3')(b3bz)

    b4e = Concatenate(name='b4e')([p3, b2br])
    b4a = lbconv(dms, prs, trainables, 6, kld,3,name='b4a')(b4e)
    d4a = BatchNormalization(momentum=mmt, trainable=bn, name='d4a')(b4a, training=bn) 
    b4ar = ReLU(name='b4ar')(d4a)
    b4az = lstore(dms, prs, trainables, 6, kld,ndata,name='b4az')([b4ar, datas[1]])
    b4b = lbconv(dms, prs, trainables, 7, kld,3,name='b4b')(b4az)
    d4b = BatchNormalization(momentum=mmt, trainable=bn, name='d4b')(b4b, training=bn) 
    b4br = ReLU(name='b4br')(d4b)
    b4bz = lstore(dms, prs, trainables, 7, kld,ndata,name='b4bz')([b4br, datas[1]])
    p4 = UpSampling2D(size=(2, 2), name='p4')(b4bz)

    b5e = Concatenate(name='b5e')([p4, b1br])
    b5a = lbconv(dms, prs, trainables, 8, kld,3,name='b5a')(b5e)
    d5a = BatchNormalization(momentum=mmt, trainable=bn, name='d5a')(b5a, training=bn) 
    b5ar = ReLU(name='b5ar')(d5a)
    b5az = lstore(dms, prs, trainables, 8, kld,ndata,name='b5az')([b5ar, datas[1]])
    b5b = lbconv(dms, prs, trainables, 9, kld,3,name='b5b')(b5az)
    d5b = BatchNormalization(momentum=mmt, trainable=bn, name='d5b')(b5b, training=bn) 
    b5br = ReLU(name='b5br')(d5b)
    b5bz = lstore(dms, prs, trainables, 9, kld,ndata,name='b5bz')([b5br, datas[1]])

    lnow = lbconv(dms, prs, trainables, 10, kld,1,name='lnow')(b5bz)
    lmean = Lambda(lambda x: x, name='lmean') (lnow)
    
    return [lnow, lmean]
def myNN_kn(datas):
    dmdata = 3
    dmclass = 40*10
    h1 = 256
    w1 = 384
    kldiv = 100*h1*w1
    dms = [[dmdata,40,40,40,40,24*40],
            [40,40,40,40,40,dmclass]]
    trainables = []
    for i in range(12):
        trainables.append(True)
    priors = []
    for idx in range(len(dms[0])):
        vpriora = 5./40
        vpriorb = 1.
        wpriormu = 0.
        wpriorvar = 1.
        bpriormu = 0.
        bpriorvar = 1.
        priors.append([vpriora, vpriorb, wpriormu, wpriorvar, bpriormu, bpriorvar])
    p0 = MaxPooling2D((4, 4), strides=(4, 4), name='p0')(datas[0])
    b1a = lbconvR(dms, priors, trainables, 0, kldiv,3,name='b1a')(p0)
    p1 = MaxPooling2D((2, 2), strides=(2, 2), name='p1')(b1a)
    b2a = lbconvR(dms, priors, trainables, 1, kldiv,3,name='b2a')(p1)
    p2 = MaxPooling2D((2, 2), strides=(2, 2), name='p2')(b2a)
    b3a = lbconvR(dms, priors, trainables, 2, kldiv,3,name='b3a')(p2)
    p3 = MaxPooling2D((2, 2), strides=(2, 2), name='p3')(b3a)
    b4a = lbconvR(dms, priors, trainables, 3, kldiv,3,name='b4a')(p3)
    p4 = MaxPooling2D((2, 2), strides=(2, 2), name='p4')(b4a)
    b5a = lbconvR(dms, priors, trainables, 4, kldiv,3,name='b5a')(p4)
    f0 = Flatten()(b5a)
    lnow = fcsig(dms, priors, trainables, 5, kldiv,name='lnow')(f0)
    lmean = Lambda(lambda x: x, name='lmean') (lnow)
    return [lnow, lmean]
def myNNZ_noRes(datas, dms, prs, trainables, kld, mmt, bn, ndata):
    
    b1a = lbconvR(dms, prs, trainables, 0, kld,3,name='b1a')(datas[0])
    b1az = lstore(dms, prs, trainables, 0, kld,ndata,name='b1az')([b1a, datas[1]])
    b1b = lbconv(dms, prs, trainables, 1, kld,3,name='b1b')(b1az)
    b1br = ReLU(name='b1br')(b1b)
    b1bz = lstore(dms, prs, trainables, 1, kld,ndata,name='b1bz')([b1br, datas[1]])
    p1 = MaxPooling2D((2, 2), strides=(2, 2), name='p1')(b1bz)

    b2a = lbconv(dms, prs, trainables, 2, kld,3,name='b2a')(p1)
    b2ar = ReLU(name='b2ar')(b2a)
    b2az = lstore(dms, prs, trainables, 2, kld,ndata,name='b2az')([b2ar, datas[1]])
    b2b = lbconv(dms, prs, trainables, 3, kld,3,name='b2b')(b2az)
    b2br = ReLU(name='b2br')(b2b)
    b2bz = lstore(dms, prs, trainables, 3, kld,ndata,name='b2bz')([b2br, datas[1]])
    p2 = MaxPooling2D((2, 2), strides=(2, 2), name='p2')(b2bz)

    b3a = lbconv(dms, prs, trainables, 4, kld,3,name='b3a')(p2)
    b3ar = ReLU(name='b3ar')(b3a)
    b3az = lstore(dms, prs, trainables, 4, kld,ndata,name='b3az')([b3ar, datas[1]])
    b3b = lbconv(dms, prs, trainables, 5, kld,3,name='b3b')(b3az)
    b3br = ReLU(name='b3br')(b3b)
    b3bz = lstore(dms, prs, trainables, 5, kld,ndata,name='b3bz')([b3br, datas[1]])
    p3 = UpSampling2D(size=(2, 2), name='p3')(b3bz)

    b4e = Concatenate(name='b4e')([p3, b2br])
    b4a = lbconv(dms, prs, trainables, 6, kld,3,name='b4a')(b4e)
    b4ar = ReLU(name='b4ar')(b4a)
    b4az = lstore(dms, prs, trainables, 6, kld,ndata,name='b4az')([b4ar, datas[1]])
    b4b = lbconv(dms, prs, trainables, 7, kld,3,name='b4b')(b4az)
    b4br = ReLU(name='b4br')(b4b)
    b4bz = lstore(dms, prs, trainables, 7, kld,ndata,name='b4bz')([b4br, datas[1]])
    p4 = UpSampling2D(size=(2, 2), name='p4')(b4bz)

    b5e = Concatenate(name='b5e')([p4, b1br])
    b5a = lbconv(dms, prs, trainables, 8, kld,3,name='b5a')(b5e) 
    b5ar = ReLU(name='b5ar')(b5a)
    b5az = lstore(dms, prs, trainables, 8, kld,ndata,name='b5az')([b5ar, datas[1]])
    b5b = lbconv(dms, prs, trainables, 9, kld,3,name='b5b')(b5az)
    b5br = ReLU(name='b5br')(b5b)
    b5bz = lstore(dms, prs, trainables, 9, kld,ndata,name='b5bz')([b5br, datas[1]])

    lnow = lbconv(dms, prs, trainables, 10, kld,1,name='lnow')(b5bz)
    lmean = Lambda(lambda x: x, name='lmean') (lnow)
    
    return [lnow, lmean]

