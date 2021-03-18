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
from keras.datasets import mnist
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
outputs='/models/mccikpc2/CPI-analysis/sae_mnist/model_epochs_300_sae_mnist'

init = VarianceScaling(scale=1. / 3., mode='fan_in',
                           distribution='uniform')

if defineModel==1:
    # symmetric fully connected autoencoder - see https://arxiv.org/pdf/1511.06335.pdf
    # Encoder Layers
#     
    inputs=Input(name='input', shape=(28**2,))

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

    x=Dense(28**2, activation='relu', kernel_initializer=init, \
        name='decoder_0')(x)
        
    autoencoder=Model(inputs=inputs, outputs=x, name='AE')

        
    autoencoder.summary()

#     autoencoder.compile(optimizer='adam', loss='mse',metrics=['mse'])
    autoencoder.compile(optimizer=SGD(1,0.9), loss='mse',metrics=['mse'])




if loadData:
    print('Loading data...')
    # load images
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    x_train=X_train.reshape((X_train.shape[0],-1))
    x_test=X_test.reshape((X_test.shape[0],-1))

    x_train=x_train.astype('float32')/255.
    x_test=x_test.astype('float32')/255.

    print('data is loaded')

if runFit:
    # train the model
    autoencoder.fit(x_train, x_train, epochs=300, batch_size=256, \
                    validation_data=(x_test,x_test),verbose=1)



    model_json = autoencoder.to_json()
    with open(outputs + '.json','w') as json_file:
        json_file.write(model_json)
    autoencoder.save_weights(outputs + '.h5')

