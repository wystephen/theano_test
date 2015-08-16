#-*- coding:utf-8 -*-
'''
程序说明（Introduce）：
引用（References）：
创建时间：’2015/8/15 16:35'
编写与修改：

'''
__author__ = 'WangYan'
import theano
import theano.tensor as T
from theano.tensor.signal import downsample
from theano.tensor.nnet import  conv

from logistic import LogisticRegression, load_data
from MLP import HiddenLayer

from LeNet import  LeNetConvPoolLayer
import numpy
import scipy

import pylab

import matplotlib.pyplot as plt

import numpy as np

import timeit
import sys,os,cPickle
import MLP,load_data_test,logistic
class CNN():
    def __init__(self,rng,input,nkerns,batch_size):
        self.layer0_input= input.reshape((batch_size,1,28,28))

        self.layer0 = LeNetConvPoolLayer(
            rng,
            input=self.layer0_input,
            image_shape=(batch_size,1,28,28),
            filter_shape=(nkerns[0],1,5,5),
            poolsize = (2,2)
        )

        self.layer1 = LeNetConvPoolLayer(
            rng,
            input = self.layer0.output,
            image_shape=(batch_size,nkerns[0],12,12),
            filter_shape = (nkerns[0],5,5),
            poolsize = (2,2)
        )

        self.layer2_input = self.layer1.output.flatten(2)

        self.layer2 = HiddenLayer(
            rng,
            input = self.layer2_input,
            n_in = nkerns[1] *4*4,
            n_out = 500,
            activation = T.tanh
        )

        self.layer3  = HiddenLayer(
            rng,
            input = self.layer2.output,
            n_in = 500,
            n_out = 50,
            activation = T.tanh
        )

        self.layer4 = LogisticRegression(
            input = self.layer3.output,
            n_in = 50,
            n_out = 10
        )

        self.errors = self.layer4.errors

        self.params = self.layer4.params + self.layer3.params+ self.layer2.params + self.layer1.params + self.layer0.params

    def __getstate__(self):
        weights = [p.get_value() for p in self.params]
        return weights

    def __setstate__(self,weights):
        i = iter(weights)
        for p in self.params:
            p.set_value(i.__next__())

def train(learning_rate = 0.1, n_epochs = 200,
          nkerns = [20,50],batch_size = 1000):
    rng = np.random.RandomState()
    ##第一步，准备数据
    from csvread import read_csv
    train_x , train_y, test_x = read_csv()
    train_set_x = theano.shared(np.asarray(train_x[:len(train_x)*2/3,:],type = theano.config.floatX),borrow = True)
    train_set_y = theano.shared(np.asarray(train_y[:len(train_y)*2/3,:],type=theano.config.floatX),borrow = True)
    valid_set_x = theano.shared(np.asarray(train_x[len(train_x)*2/3:len(train_x)*5/6,:],type = theano.config.floatX),borrow = True)
    valid_set_y = theano.shared(np.asarray(train_y[len(train_y)*2/3:len(train_x)*5/6,:],type=theano.config.floatX),borrow = True)
    test_set_x = theano.shared(np.asarray(train_x[len(train_x)*5/6:len(train_x),:],type = theano.config.floatX),borrow = True)
    test_set_y = theano.shared(np.asarray(train_y[len(train_x)*5/6:len(train_y),:],type=theano.config.floatX),borrow = True)

    n_train_batches = train_set_x.get_value(borrow= True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow = True).shape[0]
    n_test_batches = test_set_x.get_value(borrow = True).shape[0]

    n_train_batches /=batch_size
    n_valid_batches /=batch_size
    n_test_batches /=batch_size

    index = T.dscalar()
    x = T.dmatrix('x')
    y = T.dvector('y')

    ##构建三个模型
    layer0_input =  x.reshape((batch_size,1,28,28))

    digit_Net = CNN(
        rng=rng,
        input = layer0_input,
        nkerns = nkerns,
        batch_size = batch_size
    )

    cost = CNN.layer4.negative_log_likelihood(y)

    test_model = theano.function(
        [index],
        CNN.layer4.errors(y),
        givens={
            x:test_set_x[index*batch_size:(index+1)*batch_size],
            y:test_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    valid_model = theano.function(
        [index],
        CNN.layer4.errors(y),
        givens={
            x:valid_set_x[index*batch_size:(index+1)*batch_size],
            y:valid_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    grads = T.grad(cost,CNN.params)

    updates = [
        (param_i , param_i - learning_rate * grad_i)
        for param_i,grad_i in zip(CNN.params,grads)
    ]

    train_model = theano.function(
        [index],
        cost,
        updates = updates,
        givens={
            x:train_set_x[index * batch_size:(index+1)*batch_size],
            y:train_set_y[index*batch_size:(index+1) * batch_size]
        }
    )

    ##开始计算，训练过程
































