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
            filter_shape = (nkerns[1],nkerns[0],5,5),
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

        #self.layer3  = HiddenLayer(
        #    rng,
        #    input = self.layer2.output,
        #    n_in = 800,
        #    n_out = 200,
        #    activation = T.tanh
        #)

        self.layer4 = LogisticRegression(
            input = self.layer2.output,
            n_in = 500,
            n_out = 10
        )

        self.errors = self.layer4.errors

        self.params = self.layer4.params + self.layer2.params + self.layer1.params + self.layer0.params

    def __getstate__(self):
        weights = [p.get_value() for p in self.params]
        return weights

    def __setstate__(self,weights):
        i = iter(weights)
        for p in self.params:
            p.set_value(i.next())

def train(learning_rate = 0.1, n_epochs = 300,
          nkerns = [20,50],batch_size = 1000):
    rng = np.random.RandomState()
    ##第一步，准备数据
    from csvread import read_csv
    train_x , train_y, test_x = read_csv()
    train_set_x = theano.shared(np.asarray(train_x[0:len(train_x)*4/5,:],dtype = theano.config.floatX),borrow = True)
    train_set_y = T.cast(theano.shared(np.asarray(train_y[0:len(train_y)*4/5],dtype=theano.config.floatX),borrow = True),'int32')
    valid_set_x = theano.shared(np.asarray(train_x[len(train_x)*4/5:len(train_x)*5/6,:],dtype = theano.config.floatX),borrow = True)
    valid_set_y = T.cast(theano.shared(np.asarray(train_y[len(train_y)*4/5:len(train_x)*5/6],dtype=theano.config.floatX),borrow = True),'int32')
    test_set_x = theano.shared(np.asarray(train_x[len(train_x)*5/6:len(train_x),:],dtype = theano.config.floatX),borrow = True)
    test_set_y = T.cast(theano.shared(np.asarray(train_y[len(train_x)*5/6:len(train_y)],dtype=theano.config.floatX),borrow = True),'int32')

    n_train_batches = train_set_x.get_value(borrow= True).shape[0]
    n_valid_batches = valid_set_x.get_value(borrow = True).shape[0]
    n_test_batches = test_set_x.get_value(borrow = True).shape[0]

    n_train_batches /=batch_size
    n_valid_batches /=batch_size
    n_test_batches /=batch_size

    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')

    ##构建三个模型

    layer0_input =  x.reshape((batch_size,1,28,28))

    digit_Net = CNN(
        rng=rng,
        input = layer0_input,
        nkerns = nkerns,
        batch_size = batch_size
    )

    cost = digit_Net.layer4.negative_log_likelihood(y)

    test_model = theano.function(
        [index],
        digit_Net.layer4.errors(y),
        givens={
            x:test_set_x[index*batch_size:(index+1)*batch_size],
            y:test_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    valid_model = theano.function(
        [index],
        digit_Net.layer4.errors(y),
        givens={
            x:valid_set_x[index*batch_size:(index+1)*batch_size],
            y:valid_set_y[index*batch_size:(index+1)*batch_size]
        }
    )

    grads = T.grad(cost,digit_Net.params)

    updates = [
        (param_i , param_i - learning_rate * grad_i)
        for param_i,grad_i in zip(digit_Net.params,grads)
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

    print '---training'

    patience = 10000
    patience_increase =2

    improvement_threshold = 0.995

    validation_frequency = min(n_train_batches,patience/2)

    best_validation_loss = numpy.inf
    best_iter =0
    test_score = 0.
    start_time = timeit.default_timer()

    epoch = 0
    done_looping = False

    while(epoch<n_epochs) and (not done_looping):
        epoch = epoch + 1
        for minibatch_index in xrange(n_train_batches):
            iter = (epoch -1 ) * n_train_batches + minibatch_index

            if iter % 100 ==1:
                print  'training @ iter = ',iter
            cost_ij = train_model(minibatch_index)
            if(iter + 1 ) % validation_frequency == 0:
                validation_losses = [valid_model(i)for i in xrange(n_valid_batches)]
                this_validation_loss = np.mean(validation_losses)

                print ('epoch %i , minibatch %i/%i,validation error %f %%'%
                       (
                           epoch,minibatch_index+1,n_train_batches,this_validation_loss*100.
                       ))
                if this_validation_loss < best_validation_loss:
                    if this_validation_loss < best_validation_loss*improvement_threshold:
                        patience = max(patience,iter*patience_increase)

                    best_validation_loss = this_validation_loss
                    best_iter = iter

                    test_losses = [
                        test_model(i)
                        for i in xrange(n_test_batches)
                    ]

                    test_score = np.mean(test_losses)
                    print (' epoch %i , minibatch %i/%i, test error of ,'
                            'with the best performance %f %%'%
                           (
                               epoch,best_validation_loss * 100.,best_iter + 1,test_score * 100.
                           )
                            )

            if patience <= iter :
                done_looping = True
                break

    end_time = timeit.default_timer()
    print ('Optimization complete')
    print ('best validation score of %f %% obtained at iteration %i,'
           'with test performance %f %%'%
           (best_validation_loss * 100., best_iter + 1, test_score * 100.))

    print >> sys.stderr, ('the code for file ' +
                         os.path.split(__file__)[1] +
                         'ran for %.2fm' %((end_time - start_time)/60.))

    f = open('params','wb')
    cPickle.dump(digit_Net.__getstate__(),f,protocol=cPickle.HIGHEST_PROTOCOL)
    f.close()

def predict():
    index = T.lscalar()
    x = T.matrix('x')
    y = T.ivector('y')
    batch_size = 1000
    rng = np.random.RandomState()
    digit_Net = CNN(
        rng=rng,
        input = x,
        nkerns = [20,50],
        batch_size = batch_size
    )
    print 'predict ====='
    f = open('params','rb')
    #print cPickle.load(f)
    digit_Net.__setstate__(cPickle.load(f))
    f.close()

    RET = []
    from csvread import read_csv
    train_x , train_y, test_x = read_csv()
    print 'test_X:',len(test_x),test_x.shape

    test_data = theano.shared(np.asarray(test_x,dtype = theano.config.floatX),borrow = True)
    n_test_batches = test_data.get_value(borrow=True).shape[0]
    n_test_batches /= batch_size

    predict_model = theano.function([index],outputs=digit_Net.layer4.predict(),
                                    givens={
                                        x:test_data[index*batch_size:(index+1)*batch_size]
                                    })

    for it in xrange(n_test_batches):
        p = predict_model(it)
        p = np.argmax(p,axis=1)
        p = p.astype(int)
        for k in range(len(p)):
            RET.append(p[k])

    #for it in range(len(test_x)/2):
    #    test_data = test_x[it:it+1]
    #    N = len(test_data)
    #    print 'N:',N
    #    test_data = theano.shared(np.asarray(test_data,dtype=theano.config.floatX))
    #
    #
    #    test_labels = T.cast(theano.shared(np.asarray(np.zeros(batch_size),dtype=theano.config.floatX)),'int32')
    #
    #
    #    ppm = theano.function([index],outputs=digit_Net.layer4.predict(),
    #                          givens={
    #                              x:test_data[index:index+1],
    #                              y:test_labels
    #                          },
    #                          on_unused_input = 'warn')
    #
    #    p = [ppm(0)]
    #
    #    p = np.argmax(p,axis = 1)
    #
    #    p = p.astype(int)
    #    RET.append(p)


    print RET
    subm = np.empty((len(RET),2))
    subm[:,0] = np.arange(1,len(RET)+1)
    subm[:,1] = RET[:]

    np.savetxt('submission.csv',subm,fmt = '%d',delimiter=',',header = 'ImageId,Label',comments='')







if __name__ == '__main__':
    train()
    predict()






    

































