#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 22 10:31:14 2020

@author: mccikpc2
"""

"""
    Paper of relevance: https://arxiv.org/pdf/1511.06335.pdf
    
    1. load image data
    2. load encoder model and build up encoder layer
    3. add clustering layer
    4. retrain the encoder + cluster layer
    
"""

#import tensorflow
import keras
import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Model, model_from_json, Sequential
import h5py
from sklearn.cluster import KMeans
from keras.engine.topology import Layer, InputSpec
from keras.layers import Dense, Input
from keras.optimizers import SGD
from keras.optimizers import adam
from keras import callbacks
from keras.initializers import VarianceScaling
import keras.backend as K
import metrics





"""
    Auxiliary target distribution
"""
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T
   
    
class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label, i.e. a vector that represents the probability of the
    sample belonging to each cluster. The probability is calculated with student's t-distribution.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), \
            initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.
         Measure the similarity between embedded point z_i and centroid µ_j.
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))          






if __name__ == "__main__":

    n_clusters=10
    batch_size=256
    loadData=True
    inputs='/models/mccikpc2/CPI-analysis/sae_mnist/model_epochs_50_sae_mnist'
    #inputs='/tmp/model_epochs_50_dense64'






    """
        1. Load image data++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
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
        ----------------------------------------------------------------------------------
    """





    """
        load the full model+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
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




    """
        2. Load encoder model+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    # see https://keras.io/getting-started/faq/#how-can-i-obtain-the-output-of-an-intermediate-layer
    layer_name='encoder_3'
    encoder_model = Model(inputs=loaded_model.input, \
                             outputs=loaded_model.get_layer(layer_name).output)

    """
    # https://stackoverflow.com/questions/53843573/extracting-encoding-decoding-models-from-keras-autoencoder-using-sequential-api
    # find the dense layer
    for i in range(len(loaded_model.layers)): 
        if loaded_model.layers[i].name == 'dense_1': 
            break 
    ei=i+1
    encoder_model = Sequential()
    for i in range(0,ei):
        encoder_model.add(loaded_model.layers[i])
    """
    """
        ----------------------------------------------------------------------------------
    """

#     for i in range(0,ei):
#         encoder_model.layers[i].trainable = False

    """
        3. Add clustering layer+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder_model.output)
    new_model = Model(inputs=encoder_model.input, outputs=clustering_layer)
    new_model.summary()
#     new_model.compile(optimizer=SGD(0.01,0.9), loss='kld')
    new_model.compile(optimizer='adam', loss='kld')
#     new_model.compile(optimizer='adam', loss='categorical_crossentropy')
    """
        ----------------------------------------------------------------------------------
    """




    """
        initialise cluster centers using k-means++++++++++++++++++++++++++++++++++++++++++
    """
    print('First k-means...')
    kmeans = KMeans(n_clusters=n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(encoder_model.predict(x_train))
    y_pred_last = np.copy(y_pred)
    new_model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])
    print('...done k-means')
    """
        ----------------------------------------------------------------------------------
    """





    """
        4. retrain the encoder + cluster layer++++++++++++++++++++++++++++++++++++++++++++
    """
    loss = 0
    index = 0
    maxiter = 16000
    update_interval = 140
    index_array = np.arange(x_train.shape[0])
    tol = 0.001 # tolerance threshold to stop training


    for ite in range(int(maxiter)):
        print(".",end="") # print . without newline
        if ite % update_interval == 0:
            q = new_model.predict(x_train, verbose=1)
            p = target_distribution(q)  # update the auxiliary target distribution p

            # evaluate the clustering performance
            y_pred = q.argmax(1)


            if y_test is not None:
                acc = np.round(metrics.acc(y_test, y_pred), 5)
                nmi = np.round(metrics.nmi(y_test, y_pred), 5)
                ari = np.round(metrics.ari(y_test, y_pred), 5)
                loss = np.round(loss, 5)
                print('Iter %d: acc = %.5f, nmi = %.5f, ari = %.5f' \
                    % (ite, acc, nmi, ari), ' ; loss=', loss)
                    

            # check stop criterion - model convergence
            delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
            y_pred_last = np.copy(y_pred)
            if ite > 0:
                print('delta_label=', delta_label,' and tol=',tol)
    
            if ite > 0 and delta_label < tol:
                print('delta_label ', delta_label, '< tol ', tol)
                print('Reached tolerance threshold. Stopping training.')
                break
        idx = index_array[index * batch_size: min((index+1) * batch_size, x_train.shape[0])]
        loss = new_model.train_on_batch(x=x_train[idx], y=p[idx])
        index = index + 1 if (index + 1) * batch_size <= x_train.shape[0] else 0


    model_json = new_model.to_json()
    with open(inputs + '_final.json','w') as json_file:
        json_file.write(model_json)
    new_model.save_weights(inputs + '_final.h5')

    """
        ----------------------------------------------------------------------------------
    """




