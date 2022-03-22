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
from keras.optimizers import adam
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
outputs='/models/mccikpc2/CPI-analysis/cnn/model_t5_epochs_50_dense64_2'



if defineModel==1:
    # 64 in dense layer bottleneck
    inputs=Input(name='input', shape=(128,128,1))

    # Encoder Layers
    x=Conv2D(32, 5, strides=2, activation='relu',padding='same')(inputs)
    x=Conv2D(64, 5, strides=2, activation='relu',padding='same')(x)
    x=Conv2D(128, 3, strides=2, activation='relu',padding='same')(x)
    x=Conv2D(256, 3, strides=2, activation='relu',padding='same')(x)



    
    # Flatten encoding for visualization
    x=Flatten()(x)
    x=Dense(64, activation='relu')(x) # 64 habits?
    x=Dense(8*8*256,activation='relu')(x)
    #autoencoder.add(Dense(64, activation='relu')) # 64 habits? maybe try relu here
    x=Reshape((8, 8, 256))(x)
    
    
    # Decoder Layers
    x=Conv2DTranspose(128, 3, strides=2, activation='relu', padding='same')(x)
    x=Conv2DTranspose(64, 3, strides=2, activation='relu', padding='same')(x)
    x=Conv2DTranspose(32, 5, strides=2, activation='relu', padding='same')(x)
    #x=Conv2DTranspose(32, 5, strides=2, activation='relu', padding='same')(x)

    x=Conv2DTranspose(1, 5, strides=2, padding='same')(x)
        
    autoencoder=Model(inputs=inputs, outputs=x, name='AE')
    autoencoder.summary()

    autoencoder.compile(optimizer='adam', loss='mse',metrics=['mse'])
elif defineModel==2:
    # 16 in dense layer bottleneck
    inputs=Input(name='input', shape=(128,128,1))

    # Encoder Layers
    x=Conv2D(32, (3, 3), activation='relu')(inputs)
    x=MaxPooling2D((2, 2))(x) # could add some dropout after pooling layers
                                          # might help with overfitting
    x=Conv2D(64, (3, 3), activation='relu')(x)
    x=MaxPooling2D((2, 2))(x)
    x=Conv2D(64, (3, 3), activation='relu')(x)
    x=MaxPooling2D((2, 2))(x)
    x=Conv2D(64, (3, 3), activation='relu')(x)
    x=MaxPooling2D((2, 2))(x)
    x=Conv2D(64, (3, 3), activation='relu')(x)



    
    # Flatten encoding for visualization
    x=Flatten()(x)
    x=Dense(16, activation='softmax')(x) # 64 habits?
    #autoencoder.add(Dense(64, activation='relu')) # 64 habits? maybe try relu here
    x=Reshape((4, 4, 1))(x)
    
    
    # Decoder Layers
    x=Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    x=BatchNormalization()(x)
    x=Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    x=BatchNormalization()(x)
    x=Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    x=BatchNormalization()(x)
    x=Conv2DTranspose(64, (3, 3), strides=2, activation='relu', padding='same')(x)
    x=BatchNormalization()(x)
    x=Conv2DTranspose(32, (3, 3), strides=2, activation='relu', padding='same')(x)

    x=Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)
        
    autoencoder=Model(inputs=inputs, outputs=x, name='AE')
    autoencoder.summary()

    autoencoder.compile(optimizer='adam', loss='mse',metrics=['mse'])



if loadData:
    print('Loading data...')
    # load images
    h5f = h5py.File('/models/mccikpc2/CPI-analysis/postProcessed_t5_l50.h5','r')
    images=h5f['images'][:]
    images=np.expand_dims(images,axis=3)
    lens  =h5f['lens'][:]
    times =h5f['times'][:]
    h5f.close()
    
    i1 = len(lens)
    i11=60000
    i22=10000
    indices = np.random.permutation(i1)
    #split1=int(0.8*i1)
    training_idx, test_idx = indices[:i11], indices[i11:i11+i22]
    
    x_train, x_test = images[training_idx,:,:], images[test_idx,:,:]
    del images
    x_train=x_train.astype('float32')/255.
    x_test=x_test.astype('float32')/255.

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
    # make a copy of the first 11 layers - encoder part
#     encoder=autoencoder.layers[0:11]
#     # add softmax layer: https://radicalrafi.github.io/blog/autoencoders-as-classifiers/
#     encoder.add(Dense(10,activation='softmax'))
#     encoder.compile(optimizer'adam',loss='categorical_crossentropy', )
