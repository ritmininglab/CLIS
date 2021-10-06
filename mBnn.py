import numpy as np
import tensorflow as tf

from tensorflow.keras import layers
from tensorflow.keras.layers import Reshape, MaxPooling2D, Flatten, RepeatVector, UpSampling2D, Concatenate, Activation
from tensorflow.keras.layers import Conv2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Lambda, Multiply, Add
from tensorflow.keras.layers import Softmax, ReLU
from tensorflow.keras.layers import BatchNormalization

from tensorflow.keras import regularizers

initialvar = -8.
small = 1e-5

class ConvBayesRelu(layers.Layer):
    def __init__(self, dims, priors, trainables, idx, kldiv, wh, name='name'):
        super(ConvBayesRelu, self).__init__(name=name)
        self.dim1 = dims[0][idx]
        self.dim2 = dims[1][idx]
        self.trainable = trainables[idx]
        self.wmu0 = priors[idx][2]
        self.wvar0 = priors[idx][3]
        self.bmu0 = priors[idx][4]
        self.bvar0 = priors[idx][5]
        self.wh = wh
        self.kldiv = kldiv
    def build(self, inshape):
        w_init = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
        v_init = tf.keras.initializers.Constant(initialvar)
        self.w = self.add_weight("w", trainable=self.trainable,
                                 initializer=w_init, shape=[self.wh,self.wh,self.dim1, self.dim2])
        self.wv = self.add_weight("wv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.wh,self.wh,self.dim1, self.dim2])
        self.b = self.add_weight("b", trainable=self.trainable,
                                 initializer=w_init, shape=[self.dim2])
        self.bv = self.add_weight("bv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.dim2])
    def call(self, x):
        rngw = tf.random.truncated_normal([self.wh,self.wh,self.dim1,self.dim2])
        rngb = tf.random.truncated_normal([self.dim2])
        wnow = self.w + tf.math.multiply(rngw, tf.exp(0.5*self.wv))
        bnow = self.b + tf.math.multiply(rngb, tf.exp(0.5*self.bv))
        tempout = tf.add(tf.nn.conv2d(x, wnow, [1,1,1,1], padding='SAME'), bnow)
        output = tf.nn.relu(tempout)
        
        term0 = -0.5*self.wh*self.wh*self.dim1*self.dim2
        term1 = 0.5*tf.reduce_sum(np.log(self.wvar0) - self.wv)
        term2 = 0.5*tf.reduce_sum((tf.exp(self.wv) + (self.w - self.wmu0)**2) / self.wvar0)
        term0b = -0.5*self.dim2
        term1b = 0.5*tf.reduce_sum(np.log(self.bvar0) - self.bv)
        term2b = 0.5*tf.reduce_sum((tf.exp(self.bv) + (self.b - self.bmu0)**2) / self.bvar0)
        sumkl = term1 + term2 + term0 + term1b + term2b + term0b
        self.add_loss(sumkl / self.kldiv)
        
        return output
    def compute_output_shape(self, inshape):
        return (inshape[0], inshape[1], inshape[2], self.dim2)
class ConvBayes(layers.Layer):
    def __init__(self, dims, priors, trainables, idx, kldiv, wh, name='name'):
        super(ConvBayes, self).__init__(name=name)
        self.dim1 = dims[0][idx]
        self.dim2 = dims[1][idx]
        self.trainable = trainables[idx]
        self.wmu0 = priors[idx][2]
        self.wvar0 = priors[idx][3]
        self.bmu0 = priors[idx][4]
        self.bvar0 = priors[idx][5]
        self.wh = wh
        self.kldiv = kldiv
    def build(self, inshape):
        w_init = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
        v_init = tf.keras.initializers.Constant(initialvar)
        self.w = self.add_weight("w", trainable=self.trainable,
                                 initializer=w_init, shape=[self.wh,self.wh,self.dim1, self.dim2])
        self.wv = self.add_weight("wv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.wh,self.wh,self.dim1, self.dim2])
        self.b = self.add_weight("b", trainable=self.trainable,
                                 initializer=w_init, shape=[self.dim2])
        self.bv = self.add_weight("bv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.dim2])
    def call(self, x):
        rngw = tf.random.truncated_normal([self.wh,self.wh,self.dim1,self.dim2])
        rngb = tf.random.truncated_normal([self.dim2])
        wnow = self.w + tf.math.multiply(rngw, tf.exp(0.5*self.wv))
        bnow = self.b + tf.math.multiply(rngb, tf.exp(0.5*self.bv))
        tempout = tf.add(tf.nn.conv2d(x, wnow, [1,1,1,1], padding='SAME'), bnow)
        
        term0 = -0.5*self.wh*self.wh*self.dim1*self.dim2
        term1 = 0.5*tf.reduce_sum(np.log(self.wvar0) - self.wv)
        term2 = 0.5*tf.reduce_sum((tf.exp(self.wv) + (self.w - self.wmu0)**2) / self.wvar0)
        term0b = -0.5*self.dim2
        term1b = 0.5*tf.reduce_sum(np.log(self.bvar0) - self.bv)
        term2b = 0.5*tf.reduce_sum((tf.exp(self.bv) + (self.b - self.bmu0)**2) / self.bvar0)
        sumkl = term1 + term2 + term0 + term1b + term2b + term0b
        self.add_loss(sumkl / self.kldiv)
        
        return tempout
    def compute_output_shape(self, inshape):
        return (inshape[0], inshape[1], inshape[2], self.dim2)

class zlayershare(layers.Layer):
    def __init__(self, dims, priors, trainables, idx, kldiv,name='layername'):
        super(zlayershare, self).__init__(name=name)
        self.dim1 = dims[0][idx]
        self.dim2 = dims[1][idx]
        self.priora = priors[idx][0]
        self.priorb = priors[idx][1]
        self.kldiv = kldiv
        self.trainable = trainables[idx]
        self.temperature = 1
    def build(self, inshape): 
        self.nbatch = inshape[0][0]
        a_init = tf.keras.initializers.Constant(5./40)
        b_init = tf.keras.initializers.Constant(1.)
        z_init = tf.keras.initializers.Constant(0.)
        self.a = self.add_weight("a",trainable=self.trainable,
                                 initializer=a_init,
                                  shape=[self.dim2],)
        self.b = self.add_weight("b",trainable=self.trainable,
                                 initializer=b_init,
                                  shape=[self.dim2],)
        self.pienk = self.add_weight("pienk",trainable=self.trainable,
                                 initializer=z_init,
                                  shape=[1,self.dim2],)
    def call(self, xs):
        x = xs[0]
        
        alpha = tf.nn.softplus(self.a)
        beta = tf.nn.softplus(self.b)
        epsab = tf.random.uniform(shape=tf.shape(alpha), minval=0.1, maxval=0.9)
        kuma = tf.math.pow(1 - tf.math.pow(epsab, 1/beta), 1/alpha)
        priorlogpie = tf.math.log(kuma)
        
        pienk = tf.sigmoid(self.pienk)
        epspie = tf.random.uniform(shape=tf.shape(pienk), minval=0.1, maxval=0.9)
        temp2 = tf.math.log(epspie)-tf.math.log(1-epspie)
        logpienk = tf.math.log(pienk+small) - tf.math.log(1-pienk+small)
        softbernoulli = 1/(1 + tf.exp(-(logpienk+temp2)/self.temperature))
        
        ab = alpha+beta
        term0 = tf.math.lgamma(self.priora)+tf.math.lgamma(self.priorb)-tf.math.lgamma(self.priora+self.priorb)
        term1 = -tf.math.lgamma(alpha)-tf.math.lgamma(beta)+tf.math.lgamma(ab)
        term2 = tf.multiply(alpha-self.priora, tf.math.digamma(alpha))
        term3 = tf.multiply(beta-self.priorb, tf.math.digamma(beta))
        term4 = tf.multiply(-ab+self.priora+self.priorb, tf.math.digamma(ab))
        sumklbeta = tf.reduce_sum(term0+term1+term2+term3+term4)
        
        self.add_loss( sumklbeta / (self.kldiv) )
        
        temp = pienk
        term11 = tf.multiply(temp, tf.math.log(temp + small) - tf.math.log(tf.exp(priorlogpie) + small))
        term12 = tf.multiply(1-temp, tf.math.log(1-temp + small) - tf.math.log(1-tf.exp(priorlogpie) + small))
        sumklber = tf.reduce_sum(term11+term12)
        
        self.add_loss( sumklber / (self.kldiv) )
        
        result = tf.multiply(x, softbernoulli)
        return result
    def compute_output_shape(self, inshape):
        return (inshape[0], inshape[1], inshape[2], self.dim2)

def convblockbn(inputs, dims, priors, trainables, kldiv, mmt, bn, idx, prefix):
    b1a = ConvBayes(dims, priors, trainables, idx, kldiv,3,name=prefix+'')(inputs)
    b1ad = BatchNormalization(momentum=mmt, trainable=bn, name=prefix+'d')(b1a, training=bn) 
    b1ar = ReLU(name=prefix+'r')(b1ad)
    b1az = zlayershare(dims, priors, trainables, idx, kldiv,name=prefix+'z')([b1ar, ])
    return b1az
def convblock(inputs, dims, priors, trainables, kldiv, mmt, bn, idx, prefix):
    b1a = ConvBayes(dims, priors, trainables, idx, kldiv,3,name=prefix+'')(inputs)
    b1ar = ReLU(name=prefix+'r')(b1a)
    b1az = zlayershare(dims, priors, trainables, idx, kldiv,name=prefix+'z')([b1ar, ])
    return b1az


def myNNZ_pretrain(datas, dims, priors, trainables, kldiv, mmt, bn):

    b1a = convblock(datas[0], dims, priors, trainables, kldiv, mmt, bn, 0, 'b1a')
    b1b = convblock(b1a, dims, priors, trainables, kldiv, mmt, bn, 1, 'b1b')
    p1 = MaxPooling2D((2, 2), strides=(2, 2), name='p1')(b1b)
    
    b2a = convblock(p1, dims, priors, trainables, kldiv, mmt, bn, 2, 'b2a')
    b2b = convblock(b2a, dims, priors, trainables, kldiv, mmt, bn, 3, 'b2b')
    p2 = MaxPooling2D((2, 2), strides=(2, 2), name='p2')(b2b)
    
    b3a = convblock(p2, dims, priors, trainables, kldiv, mmt, bn, 4, 'b3a')
    b3b = convblock(b3a, dims, priors, trainables, kldiv, mmt, bn, 5, 'b3b')
    p3 = MaxPooling2D((2, 2), strides=(2, 2), name='p3')(b3b)
    
    b4a = convblock(p3, dims, priors, trainables, kldiv, mmt, bn, 6, 'b4a')
    b4b = convblock(b4a, dims, priors, trainables, kldiv, mmt, bn, 7, 'b4b')
    p4 = UpSampling2D(size=(2, 2), name='p4')(b4b)
    
    b5e = Concatenate(name='b5e')([p4, b3b])
    b5a = convblock(b5e, dims, priors, trainables, kldiv, mmt, bn, 8, 'b5a')
    b5b = convblock(b5a, dims, priors, trainables, kldiv, mmt, bn, 9, 'b5b')
    p5 = UpSampling2D(size=(2, 2), name='p5')(b5b)
    
    b6e = Concatenate(name='b6e')([p5, b2b])
    b6a = convblock(b6e, dims, priors, trainables, kldiv, mmt, bn, 10, 'b6a')
    b6b = convblock(b6a, dims, priors, trainables, kldiv, mmt, bn, 11, 'b6b')
    p6 = UpSampling2D(size=(2, 2), name='p6')(b6b)
    
    b7e = Concatenate(name='b7e')([p6, b1b])
    b7a = convblock(b7e, dims, priors, trainables, kldiv, mmt, bn, 12, 'b7a')
    b7b = convblock(b7a, dims, priors, trainables, kldiv, mmt, bn, 13, 'b7b')
    
    lnow = ConvBayes(dims, priors, trainables, 14, kldiv, 1, name='lnow')(b7b)
    lmean = Lambda(lambda x: x, name='lmean') (lnow)
    
    return [lnow, lmean]


class maskConv(layers.Layer):
    def __init__(self, masks, idx, active, name='name'):
        super(maskConv, self).__init__(name=name)
        self.mask = masks[idx][active]
    def call(self, x):
        output = tf.multiply(x, self.mask)
        return output
    def compute_output_shape(self, inshape):
        return inshape

class ConvBayesReluFrz(layers.Layer):
    def __init__(self, dims, priors, trainables, idx, kldiv, wh, name='name'):
        super(ConvBayesReluFrz, self).__init__(name=name)
        self.dim1 = dims[0][idx]
        self.dim2 = dims[1][idx]
        self.trainable = False
        self.wmu0 = priors[idx][2]
        self.wvar0 = priors[idx][3]
        self.bmu0 = priors[idx][4]
        self.bvar0 = priors[idx][5]
        self.wh = wh
        self.kldiv = kldiv
    def build(self, inshape):
        w_init = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
        v_init = tf.keras.initializers.Constant(initialvar)
        self.w = self.add_weight("w", trainable=self.trainable,
                                 initializer=w_init, shape=[self.wh,self.wh,self.dim1, self.dim2])
        self.wv = self.add_weight("wv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.wh,self.wh,self.dim1, self.dim2])
        self.b = self.add_weight("b", trainable=self.trainable,
                                 initializer=w_init, shape=[self.dim2])
        self.bv = self.add_weight("bv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.dim2])
    def call(self, x):
        rngw = tf.random.truncated_normal([self.wh,self.wh,self.dim1,self.dim2])
        rngb = tf.random.truncated_normal([self.dim2])
        wnow = self.w + tf.math.multiply(rngw, tf.exp(0.5*self.wv))
        bnow = self.b + tf.math.multiply(rngb, tf.exp(0.5*self.bv))
        tempout = tf.add(tf.nn.conv2d(x, wnow, [1,1,1,1], padding='SAME'), bnow)
        output = tf.nn.relu(tempout)
        
        return output
    def compute_output_shape(self, inshape):
        return (inshape[0], inshape[1], inshape[2], self.dim2)
class ConvBayesFrz(layers.Layer):
    def __init__(self, dims, priors, trainables, idx, kldiv, wh, name='name'):
        super(ConvBayesFrz, self).__init__(name=name)
        self.dim1 = dims[0][idx]
        self.dim2 = dims[1][idx]
        self.trainable = False
        self.wmu0 = priors[idx][2]
        self.wvar0 = priors[idx][3]
        self.bmu0 = priors[idx][4]
        self.bvar0 = priors[idx][5]
        self.wh = wh
        self.kldiv = kldiv
    def build(self, inshape):
        w_init = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
        v_init = tf.keras.initializers.Constant(initialvar)
        self.w = self.add_weight("w", trainable=self.trainable,
                                 initializer=w_init, shape=[self.wh,self.wh,self.dim1, self.dim2])
        self.wv = self.add_weight("wv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.wh,self.wh,self.dim1, self.dim2])
        self.b = self.add_weight("b", trainable=self.trainable,
                                 initializer=w_init, shape=[self.dim2])
        self.bv = self.add_weight("bv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.dim2])
    def call(self, x):
        rngw = tf.random.truncated_normal([self.wh,self.wh,self.dim1,self.dim2])
        rngb = tf.random.truncated_normal([self.dim2])
        wnow = self.w + tf.math.multiply(rngw, tf.exp(0.5*self.wv))
        bnow = self.b + tf.math.multiply(rngb, tf.exp(0.5*self.bv))
        output = tf.add(tf.nn.conv2d(x, wnow, [1,1,1,1], padding='SAME'), bnow)
        return output
    def compute_output_shape(self, inshape):
        return (inshape[0], inshape[1], inshape[2], self.dim2)
class ConvBayesnobias(layers.Layer):
    def __init__(self, dims, priors, trainables, idx, kldiv, wh, name='name'):
        super(ConvBayesnobias, self).__init__(name=name)
        self.dim1 = dims[0][idx]
        self.dim2 = dims[1][idx]
        self.trainable = trainables[idx]
        self.wmu0 = priors[idx][2]
        self.wvar0 = priors[idx][3]
        self.bmu0 = priors[idx][4]
        self.bvar0 = priors[idx][5]
        self.wh = wh
        self.kldiv = kldiv
    def build(self, inshape):
        w_init = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
        v_init = tf.keras.initializers.Constant(initialvar)
        self.w = self.add_weight("w", trainable=self.trainable,
                                 initializer=w_init, shape=[self.wh,self.wh,self.dim1, self.dim2])
        self.wv = self.add_weight("wv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.wh,self.wh,self.dim1, self.dim2])
        self.b = self.add_weight("b", trainable=False,
                                 initializer=w_init, shape=[self.dim2])
        self.bv = self.add_weight("bv", trainable=False,
                                 initializer=v_init, shape=[self.dim2])
    def call(self, x):
        rngw = tf.random.truncated_normal([self.wh,self.wh,self.dim1,self.dim2])
        wnow = self.w + tf.math.multiply(rngw, tf.exp(0.5*self.wv))
        tempout = tf.nn.conv2d(x, wnow, [1,1,1,1], padding='SAME')
        
        term0 = -0.5*self.wh*self.wh*self.dim1*self.dim2
        term1 = 0.5*tf.reduce_sum(np.log(self.wvar0) - self.wv)
        term2 = 0.5*tf.reduce_sum((tf.exp(self.wv) + (self.w - self.wmu0)**2) / self.wvar0)
        sumkl = term1 + term2 + term0 
        self.add_loss(sumkl / self.kldiv)
        
        return tempout
    def compute_output_shape(self, inshape):
        return (inshape[0], inshape[1], inshape[2], self.dim2)
class ConvBayesFrznobias(layers.Layer):
    def __init__(self, dims, priors, trainables, idx, kldiv, wh, name='name'):
        super(ConvBayesFrznobias, self).__init__(name=name)
        self.dim1 = dims[0][idx]
        self.dim2 = dims[1][idx]
        self.trainable = False
        self.wmu0 = priors[idx][2]
        self.wvar0 = priors[idx][3]
        self.bmu0 = priors[idx][4]
        self.bvar0 = priors[idx][5]
        self.wh = wh
        self.kldiv = kldiv
    def build(self, inshape):
        w_init = tf.keras.initializers.TruncatedNormal(mean=0., stddev=0.1)
        v_init = tf.keras.initializers.Constant(initialvar)
        self.w = self.add_weight("w", trainable=self.trainable,
                                 initializer=w_init, shape=[self.wh,self.wh,self.dim1, self.dim2])
        self.wv = self.add_weight("wv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.wh,self.wh,self.dim1, self.dim2])
        self.b = self.add_weight("b", trainable=self.trainable,
                                 initializer=w_init, shape=[self.dim2])
        self.bv = self.add_weight("bv", trainable=self.trainable,
                                 initializer=v_init, shape=[self.dim2])
    def call(self, x):
        rngw = tf.random.truncated_normal([self.wh,self.wh,self.dim1,self.dim2])
        wnow = self.w + tf.math.multiply(rngw, tf.exp(0.5*self.wv))
        tempout = tf.nn.conv2d(x, wnow, [1,1,1,1], padding='SAME')
        return tempout
    def compute_output_shape(self, inshape):
        return (inshape[0], inshape[1], inshape[2], self.dim2)



def maskblock(inputs, dims, priors, trainables, kldiv, ndata, masks, idx, prefix):
    a = ConvBayesRelu(dims, priors, trainables, idx, kldiv,3,name=prefix+'a')(inputs)
    ma = maskConv(masks, idx,0, name=prefix+'ma')(a)
    za = zlayershare(dims, priors, trainables, idx, kldiv,name=prefix+'za')([ma, ])
    f = ConvBayesReluFrz(dims, priors, trainables, idx, kldiv,3,name=prefix+'f')(inputs)
    mf = maskConv(masks, idx,1, name=prefix+'mf')(f)
    zf = zlayershare(dims, priors, trainables, idx, kldiv,name=prefix+'zf')([mf, ])
    add = Lambda(lambda x: x[0]+x[1], name=prefix+'add')([za, zf])
    return add
def maskblockfinal(inputs, dims, priors, trainables, kldiv, ndata, masks, idx, prefix):
    ma = maskConv(masks, idx-1,0, name=prefix+'ma')(inputs)
    a = ConvBayesnobias(dims, priors, trainables, idx, kldiv,1,name=prefix+'a')(ma)
    mf = maskConv(masks, idx-1,1, name=prefix+'mf')(inputs)
    f = ConvBayesFrz(dims, priors, trainables, idx, kldiv,1,name=prefix+'f')(mf)
    add = Lambda(lambda x: x[0]+x[1], name=prefix)([a, f])
    return add

def myNNZ_masked(datas, dims, priors, trainables, kldiv, ndata, masks):

    b1a = maskblock(datas[0], dims, priors, trainables, kldiv, ndata, masks, 0, 'b1a')
    b1b = maskblock(b1a, dims, priors, trainables, kldiv, ndata, masks, 1, 'b1b')
    p1 = MaxPooling2D((2, 2), strides=(2, 2), name='p1')(b1b)
    
    b2a = maskblock(p1, dims, priors, trainables, kldiv, ndata, masks, 2, 'b2a')
    b2b = maskblock(b2a, dims, priors, trainables, kldiv, ndata, masks, 3, 'b2b')
    p2 = MaxPooling2D((2, 2), strides=(2, 2), name='p2')(b2b)
    
    b3a = maskblock(p2, dims, priors, trainables, kldiv, ndata, masks, 4, 'b3a')
    b3b = maskblock(b3a, dims, priors, trainables, kldiv, ndata, masks, 5, 'b3b')
    p3 = MaxPooling2D((2, 2), strides=(2, 2), name='p3')(b3b)
    
    b4a = maskblock(p3, dims, priors, trainables, kldiv, ndata, masks, 6, 'b4a')
    b4b = maskblock(b4a, dims, priors, trainables, kldiv, ndata, masks, 7, 'b4b')
    p4 = UpSampling2D(size=(2, 2), name='p4')(b4b)
    
    b5e = Concatenate(name='b5e')([p4, b3b])
    b5a = maskblock(b5e, dims, priors, trainables, kldiv, ndata, masks, 8, 'b5a')
    b5b = maskblock(b5a, dims, priors, trainables, kldiv, ndata, masks, 9, 'b5b')
    p5 = UpSampling2D(size=(2, 2), name='p5')(b5b)
    
    b6e = Concatenate(name='b6e')([p5, b2b])
    b6a = maskblock(b6e, dims, priors, trainables, kldiv, ndata, masks, 10, 'b6a')
    b6b = maskblock(b6a, dims, priors, trainables, kldiv, ndata, masks, 11, 'b6b')
    p6 = UpSampling2D(size=(2, 2), name='p6')(b6b)
    
    b7e = Concatenate(name='b7e')([p6, b1b])
    b7a = maskblock(b7e, dims, priors, trainables, kldiv, ndata, masks, 12, 'b7a')
    b7b = maskblock(b7a, dims, priors, trainables, kldiv, ndata, masks, 13, 'b7b')
    
    lnow = maskblockfinal(b7b, dims, priors, trainables, kldiv, ndata, masks, 14, 'lnow')
    lmean = Lambda(lambda x: x, name='lmean') (lnow)
    
    return [lnow, lmean]


def maskblockfinalallowbias(inputs, dims, priors, trainables, kldiv, ndata, masks, idx, prefix):
    ma = maskConv(masks, idx-1,0, name=prefix+'ma')(inputs)
    a = ConvBayes(dims, priors, trainables, idx, kldiv,1,name=prefix+'a')(ma)
    mf = maskConv(masks, idx-1,1, name=prefix+'mf')(inputs)
    f = ConvBayesFrznobias(dims, priors, trainables, idx, kldiv,1,name=prefix+'f')(mf)
    add = Lambda(lambda x: x[0]+x[1], name=prefix)([a, f])
    return add

def myNNZ_maskedallowbias(datas, dims, priors, trainables, kldiv, ndata, masks):

    b1a = maskblock(datas[0], dims, priors, trainables, kldiv, ndata, masks, 0, 'b1a')
    b1b = maskblock(b1a, dims, priors, trainables, kldiv, ndata, masks, 1, 'b1b')
    p1 = MaxPooling2D((2, 2), strides=(2, 2), name='p1')(b1b)
    
    b2a = maskblock(p1, dims, priors, trainables, kldiv, ndata, masks, 2, 'b2a')
    b2b = maskblock(b2a, dims, priors, trainables, kldiv, ndata, masks, 3, 'b2b')
    p2 = MaxPooling2D((2, 2), strides=(2, 2), name='p2')(b2b)
    
    b3a = maskblock(p2, dims, priors, trainables, kldiv, ndata, masks, 4, 'b3a')
    b3b = maskblock(b3a, dims, priors, trainables, kldiv, ndata, masks, 5, 'b3b')
    p3 = MaxPooling2D((2, 2), strides=(2, 2), name='p3')(b3b)
    
    b4a = maskblock(p3, dims, priors, trainables, kldiv, ndata, masks, 6, 'b4a')
    b4b = maskblock(b4a, dims, priors, trainables, kldiv, ndata, masks, 7, 'b4b')
    p4 = UpSampling2D(size=(2, 2), name='p4')(b4b)
    
    b5e = Concatenate(name='b5e')([p4, b3b])
    b5a = maskblock(b5e, dims, priors, trainables, kldiv, ndata, masks, 8, 'b5a')
    b5b = maskblock(b5a, dims, priors, trainables, kldiv, ndata, masks, 9, 'b5b')
    p5 = UpSampling2D(size=(2, 2), name='p5')(b5b)
    
    b6e = Concatenate(name='b6e')([p5, b2b])
    b6a = maskblock(b6e, dims, priors, trainables, kldiv, ndata, masks, 10, 'b6a')
    b6b = maskblock(b6a, dims, priors, trainables, kldiv, ndata, masks, 11, 'b6b')
    p6 = UpSampling2D(size=(2, 2), name='p6')(b6b)
    
    b7e = Concatenate(name='b7e')([p6, b1b])
    b7a = maskblock(b7e, dims, priors, trainables, kldiv, ndata, masks, 12, 'b7a')
    b7b = maskblock(b7a, dims, priors, trainables, kldiv, ndata, masks, 13, 'b7b')
    
    lnow = maskblockfinalallowbias(b7b, dims, priors, trainables, kldiv, ndata, masks, 14, 'lnow')
    lmean = Lambda(lambda x: x, name='lmean') (lnow)
    
    return [lnow, lmean]
