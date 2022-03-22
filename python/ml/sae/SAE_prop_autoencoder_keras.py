#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:31:14 2020

@author: mccikpc2
"""

"""
    Keras deep convolutional autoencoder
    https://medium.com/analytics-vidhya/building-a-convolutional-autoencoder-using-keras-using-conv2dtranspose-ca403c8d144e
"""

#import tensorflow
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Model, Sequential
from keras.layers import Dense, Conv2D, \
    Dropout, BatchNormalization, Input, \
    Reshape, Flatten, Deconvolution2D, \
    Conv2DTranspose, MaxPooling2D, UpSampling2D
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import SGD 
from keras.optimizers import adam 
from keras.initializers import VarianceScaling
import h5py
import tensorflow as tf
# try
# https://stackoverflow.com/questions/46421258/limit-number-of-cores-used-in-keras
# NUM_WORKERS=32
# from keras import backend as K
# K.set_session(tf.Session(config=tf.ConfigProto( \
#     intra_op_parallelism_threads=NUM_WORKERS, \
#     inter_op_parallelism_threads=NUM_WORKERS)))

loadData=True
defineModel=1
runFit=True
outputs='/models/mccikpc2/CPI-analysis/sae/model_epochs_50_sae_prop05a_10'

init = VarianceScaling(scale=1. / 3., mode='fan_in',
                           distribution='uniform')
init='glorot_uniform' # try different initialiser
if defineModel==1:
    # symmetric fully connected autoencoder - see https://arxiv.org/pdf/1511.06335.pdf
    # Encoder Layers
#     
    inputs=Input(name='input', shape=(14,))

    x=Dense(500, activation='relu', kernel_initializer=init, \
        name='encoder_0')(inputs)

    x=Dense(500, activation='relu', kernel_initializer=init, \
        name='encoder_1')(x)

    x=Dense(2000, activation='relu', kernel_initializer=init, \
        name='encoder_2')(x)



    # features are extracted from here
    x=Dense(10, activation='relu', kernel_initializer=init, \
        name='encoder_3')(x)

    x=Dense(2000, activation='relu', kernel_initializer=init, \
        name='decoder_3')(x)

    x=Dense(500, activation='relu', kernel_initializer=init, \
        name='decoder_2')(x)

    x=Dense(500, activation='relu', kernel_initializer=init, \
        name='decoder_1')(x)

    x=Dense(14, activation='relu', kernel_initializer=init, \
        name='decoder_0')(x)
        
    autoencoder=Model(inputs=inputs, outputs=x, name='AE')

        
    autoencoder.summary()

    autoencoder.compile(optimizer='adam', loss='mse',metrics=['mse'])
    #autoencoder.compile(optimizer=SGD(1.,0.9), loss='mse',metrics=['mse'])




if loadData:
    print('Loading data...')
    # load images
    h5f = h5py.File('/models/mccikpc2/CPI-analysis/postProcessed_t5_l50.h5','r')
    #images=h5f['images'][:]
    #images=np.expand_dims(images,axis=3)
    lens  =h5f['lens'][:]
    times =h5f['times'][:]
    diams=h5f['diams'][:] 
    radii=h5f['radii'][:]
    rounds=h5f['rounds'][:] 
    l2ws=h5f['l2ws'][:]
   
    h5f.close()
    
    i1 = len(lens)
    input1=[None]*i1
    for i in range(i1):
        input1[i]= np.append(diams[i],[rounds[i],l2ws[i]]) #,np.std(diams[i]),np.min(diams[i])])

    input1=np.expand_dims(input1,axis=2)
    i11=60000
    i22=10000
    indices = np.random.permutation(i1)
    #split1=int(0.8*i1)
    training_idx, test_idx = indices[:i11], indices[i11:i11+i22]
    
    x_train, x_test = input1[training_idx,:,:], input1[test_idx,:,:]
    del input1
    x_train=x_train.reshape((x_train.shape[0],-1))
    x_test=x_test.reshape((x_test.shape[0],-1))

    #x_train=x_train.astype('float32')/255.
    #x_test=x_test.astype('float32')/255.

    lens_train,lens_test = lens[training_idx], lens[test_idx]
    times_train,times_test = times[training_idx], times[test_idx]

    print('data is loaded')

if runFit:
    # train the model
    autoencoder.fit(x_train, x_train, epochs=50, batch_size=256, \
                    validation_data=(x_test,x_test),verbose=1)



    model_json = autoencoder.to_json()
    with open(outputs + '.json','w') as json_file:
        json_file.write(model_json)
    autoencoder.save_weights(outputs + '.h5')
    
    # update this code to save the indices on which the training and validation was done
    h5faux = h5py.File(outputs + '_aux.h5' , 'w')
    h5faux.create_dataset('training_idx', data=training_idx)
    h5faux.create_dataset('test_idx', data=test_idx)
    h5faux.close()
    
