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

import sys,os,cPickle


class NeuralNetworks():
    def __init__(self):


