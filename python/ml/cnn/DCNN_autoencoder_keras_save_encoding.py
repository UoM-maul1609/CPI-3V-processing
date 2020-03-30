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


loadData=True
inputs='/models/mccikpc2/CPI-analysis/cnn/model_epochs_50_dense64'




if loadData:
    print('Loading data...')
    # load images
    h5f = h5py.File('/models/mccikpc2/CPI-analysis/postProcessed_l50.h5','r')
    images=h5f['images'][:]
    images=np.expand_dims(images,axis=3)
    lens  =h5f['lens'][:]
    times =h5f['times'][:]
    h5f.close()
    
    images=images.astype('float16')/255.

    print('data is loaded')




"""
    load the model++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
print('Loading model...')
json_file = open(inputs + '.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(inputs + '.h5')
print('model is loaded')
"""
    ----------------------------------------------------------------------------------
"""



# see https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer
layer_name='dense_1'
intermediate_layer_model = Model(inputs=loaded_model.input, \
                         outputs=loaded_model.get_layer(layer_name).output)

"""
# https://stackoverflow.com/questions/53843573/extracting-encoding-decoding-models-from-keras-autoencoder-using-sequential-api
# find the dense layer
for i in range(len(loaded_model.layers)): 
    if loaded_model.layers[i].name == 'dense_1': 
        break 
ei=i+1
intermediate_layer_model = Sequential()
for i in range(0,ei):
    intermediate_layer_model.add(loaded_model.layers[i])
"""

# calculate the 'finger prints' for cluster analysis
# finger_print=np.zeros((len(images),64))
# for i in range(len(images)):
#     finger_print[i,:]=intermediate_layer_model.predict(images[i:i+1,:,:,0:1])
finger_print=intermediate_layer_model.predict(images)
    
h5f = h5py.File(inputs + '_encoding.h5', 'w')
h5f.create_dataset('encoding', data=finger_print)
h5f.create_dataset('lens', data=lens)
h5f.close()


