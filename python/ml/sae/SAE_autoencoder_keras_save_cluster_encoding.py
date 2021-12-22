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
from keras.models import Model, model_from_json, Sequential
import h5py
from SAE_autoencoder_keras_with_clustering import ClusteringLayer 


loadData=True
inputs='/models/mccikpc2/CPI-analysis/sae/model_epochs_50_sae05_50_final'




if loadData:
    print('Loading data...')
    # load images
    h5f = h5py.File('/models/mccikpc2/CPI-analysis/postProcessed_t5_l50.h5','r')
    images=h5f['images'][:]
    images=np.expand_dims(images,axis=3)
    lens  =h5f['lens'][:]
    times =h5f['times'][:]
    h5f.close()
    
    images=images.astype('float16')/255.
    
    images=images.reshape((images.shape[0],-1))

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
finger_print=loaded_model.predict(images)
    
h5f = h5py.File(inputs + '_encoding.h5', 'w')
h5f.create_dataset('encoding', data=finger_print)
h5f.create_dataset('lens', data=lens)
h5f.close()


