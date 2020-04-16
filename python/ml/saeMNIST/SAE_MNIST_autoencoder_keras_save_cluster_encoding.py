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
from keras.models import Model, model_from_json, Sequential
import h5py
from SAE_MNIST_autoencoder_keras_with_clustering import ClusteringLayer 


loadData=True
inputs='/models/mccikpc2/CPI-analysis/sae_mnist/model_epochs_50_sae_mnist_final'




if loadData:
    print('Loading data...')
    # load images
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    x_train=X_train.reshape((X_train.shape[0],-1))
    x_test=X_test.reshape((X_test.shape[0],-1))

    x_train=x_train.astype('float32')/255.
    x_test=x_test.astype('float32')/255.

    print('data is loaded')




"""
    load the model++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
print('Loading model...')
json_file = open(inputs + '.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json, custom_objects={'ClusteringLayer':ClusteringLayer})
loaded_model.load_weights(inputs + '.h5')
print('model is loaded')
"""
    ----------------------------------------------------------------------------------
"""





# calculate the 'finger prints' for cluster analysis
finger_print=loaded_model.predict(x_test)
    
h5f = h5py.File(inputs + '_encoding.h5', 'w')
h5f.create_dataset('encoding', data=finger_print)
h5f.close()


