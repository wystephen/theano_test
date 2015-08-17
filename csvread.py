#-*- coding:utf-8 -*-
'''
程序说明（Introduce）：
引用（References）：
创建时间：’2015/8/15 15:46'
编写与修改：

'''
__author__ = 'WangYan'

import numpy as np
import csv
import os


def read_csv():
    train_set =np.genfromtxt('train.csv',delimiter=',')
    test_set = np.genfromtxt('test.csv',delimiter=',')
    train_y  = train_set[:,0]
    train_x = train_set[:,1:]
    test_x = test_set[:,:]
    mean = np.mean(train_x)
    std = np.std(train_x)
    train_x = (train_x - mean) /std
    test_x = (test_x - mean )/ std
    return train_x,train_y,test_x

if __name__ == '__main__':
    train_x,train_y,test_x = read_csv()
    np.savetxt('s',train_x)
