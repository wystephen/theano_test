#-*- coding:utf-8 -*-
'''
程序说明（Introduce）：
引用（References）：
创建时间：’2015/8/17 19:39'
编写与修改：

'''
__author__ = 'WangYan'

import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import conv

import pylab
import numpy as np
import numpy


import sys,os,cPickle

class LogisticRegression(object):##看起来是针对数字的，实际上针对答案是一个（多种可能的）结果的分散成各个神经元的都可以。
    def __init__(self,input,n_in,n_out):
        self.W = theano.shared(
            value=np.zeros((n_in,n_out),
            dtype = theano.config.floatX),
            name = 'W',
            borrow=True
        )
        self.b = theano.shared(
            value=np.zeros(
                (n_out,),
                dtype=theano.config.floatX)
            ,
            name = 'b',
            borrow=True
            )

        self.p_y_give_x = T.nnet.softmax(T.dot(input,self.W) + self.b)
        self.input = input
        self.y_pred = T.argmax(self.p_y_give_x, axis=1)
        self.params = [self.W,self.b]

    def negative_log_likelihood(self,y):
        return -T.mean(T.log(self.p_y_give_x)[T.arange(y.shape[0]),y])

    def errors(self,y):
        if y.ndim != self.y_pred.ndim:
            raise TypeError(
                'y should have the same shape as self.y_pred',
                ('y',y.type,'y_pred',self.y_pred.type)
            )
        if y.dtype.startswith('int'):
            return T.mean(T.neq(self.y_pred,y))#'''返回两者是否相等,然后去平均值，这样就得到了正确率'''
        else:
            raise NotImplementedError()

    def predict(self):
        pp = T.nnet.softmax(T.dot(self.input,self.W)+self.b)
        return pp

class HiddenLayer(object):
    def __init__(self,rng,input,n_in,n_out,W = None,b = None,activation = T.tanh):
        self.input = input
        if W is None:
            W_values = np.asarray(
                rng.uniform(
                    low = -np.sqrt(6./(n_in+n_out)),
                    high=np.sqrt(6./(n_in + n_out)),
                    size = (n_in,n_out)
                ),
                dtype = theano.config.floatX
            )

            if activation == theano.tensor.nnet.sigmoid:
                W_values*=4

            W = theano.shared(value=W_values,name='W',borrow = True)
        if b is None:
            b_values = np.zeros((n_out,),dtype=theano.config.floatX)
            b = theano.shared(value=b_values,name='b',borrow = True)

        self.W = W
        self.b = b

        lin_output = T.dot(input,self.W) + self.b
        self.output = (
            lin_output if activation is None
            else activation(lin_output)
        )

        self.params = [self.W,self.b]

class LeNetConvPoolLayer(object):

    def __init__(self,rng,input,filter_shape,image_shape,poolsize=(2,2)):
        '''

        :param rng:
        :param input:
        :param filter_shape:
        :param image_shape:
        :param poolsize:
        :return:
        '''

        assert image_shape[1] == filter_shape[1]
        self.input = input

        fan_in = numpy.prod(filter_shape[1:])

        fan_out = (filter_shape[0] * numpy.prod(filter_shape[2:]) /numpy.prod(poolsize))
        W_bound = numpy.sqrt(6. / (fan_in+fan_out))
        self.W = theano.shared(
            numpy.asarray(
                rng.uniform(low = -W_bound, high=W_bound,size = filter_shape),
                dtype=theano.config.floatX

            ),
            borrow = True
        )

        b_values = numpy.zeros((filter_shape[0],),dtype=theano.config.floatX)
        self.b = theano.shared(value=b_values,borrow = True)

        conv_out = conv.conv2d(
            input = input,
            filters = self.W,
            filter_shape = filter_shape,
            image_shape = image_shape
        )

        pooled_out = downsample.max_pool_2d(
            input = conv_out,
            ds = poolsize,
            ignore_border=True
        )

        self.output = T.tanh(pooled_out+self.b.dimshuffle('x',0,'x','x'))

        self.params = [self.W ,self.b]

        self.input = input
class NeuralNetworks():
    def __init__(self):


