import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import keras
from keras.models import Model, Sequential, Input
from keras.models import model_from_json
from DCNN_autoencoder_keras_with_clustering import target_distribution
from mpl_toolkits.axes_grid1 import ImageGrid 

mo='/tmp/model_epochs_50_dense64'


# load the encoded data
h5f = h5py.File(mo + '_encoding.h5','r')
encoded=h5f['encoding'][:]
h5f.close()
# load the clustering data
h5f = h5py.File(mo + '_final_encoding.h5','r')
clustered=h5f['encoding'][:]
lens=h5f['lens'][:]
h5f.close()


# perform k-means
print('Performing k-means')
y_pred = KMeans(n_clusters=10).fit_predict(clustered)
print('done k-means')

# new prediction
p = target_distribution(clustered)
y_pred1 = clustered.argmax(1)


# load the keras model
print('Loading model...')
json_file = open(mo + '.json','r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights(mo + '.h5')
print('model is loaded')

# find the dense layer, we will build model from there...
for i in range(len(loaded_model.layers)): 
    if loaded_model.layers[i].name == 'dense_1': 
        break 
ei=i+1



# now I just want the decoder part of the model
decoder = Sequential()
for i in range(ei,len(loaded_model.layers)):
    decoder.add(loaded_model.get_layer(index=i))
decoder.build(input_shape=loaded_model.layers[ei].input_shape) # (None,64)

decoder.summary()



y_pred=y_pred1
ims=[np.zeros((1,128,128,1))]*10
tit=[None]*10
ind0,=np.where(y_pred==0)
ind1,=np.where(y_pred==1)
ind2,=np.where(y_pred==2)
ind3,=np.where(y_pred==3)
ind4,=np.where(y_pred==4)
ind5,=np.where(y_pred==5)
ind6,=np.where(y_pred==6)
ind7,=np.where(y_pred==7)
ind8,=np.where(y_pred==8)
ind9,=np.where(y_pred==9)

fig = plt.figure(figsize=(10.,1.)) 
for i in range(100): 
    grid = ImageGrid(fig,111,nrows_ncols=(1,10),axes_pad=0.1) 
    try:
        ims[0]=decoder.predict(np.expand_dims(encoded[ind0[i],:],axis=0)) 
        tit[0]='class: ' + str(y_pred[ind0[i]])
    except:
        pass
    try:
        ims[1]=decoder.predict(np.expand_dims(encoded[ind1[i],:],axis=0)) 
        tit[1]='class: ' + str(y_pred[ind1[i]])
    except:
        pass
    try:
        ims[2]=decoder.predict(np.expand_dims(encoded[ind2[i],:],axis=0)) 
        tit[2]='class: ' + str(y_pred[ind2[i]])
    except:
        pass
    try:
        ims[3]=decoder.predict(np.expand_dims(encoded[ind3[i],:],axis=0)) 
        tit[3]='class: ' + str(y_pred[ind3[i]])
    except:
        pass
    try:
        ims[4]=decoder.predict(np.expand_dims(encoded[ind4[i],:],axis=0)) 
        tit[4]='class: ' + str(y_pred[ind4[i]])
    except:
        pass
    try:
        ims[5]=decoder.predict(np.expand_dims(encoded[ind5[i],:],axis=0)) 
        tit[5]='class: ' + str(y_pred[ind5[i]])
    except:
        pass
    try:
        ims[6]=decoder.predict(np.expand_dims(encoded[ind6[i],:],axis=0)) 
        tit[6]='class: ' + str(y_pred[ind6[i]])
    except:
        pass
    try:
        ims[7]=decoder.predict(np.expand_dims(encoded[ind7[i],:],axis=0)) 
        tit[7]='class: ' + str(y_pred[ind7[i]])
    except:
        pass
    try:
        ims[8]=decoder.predict(np.expand_dims(encoded[ind8[i],:],axis=0)) 
        tit[8]='class: ' + str(y_pred[ind8[i]])
    except:
        pass
    try:
        ims[9]=decoder.predict(np.expand_dims(encoded[ind9[i],:],axis=0)) 
        tit[9]='class: ' + str(y_pred[ind9[i]])
    except:
        pass
    

    for ax, im, ti in zip(grid, ims, tit): # ims is a list of images 
        # Iterating over the grid returns the Axes. 
        ax.imshow(im[0,:,:,0]) 
        ax.set_title(ti)
    plt.show() 
    plt.pause(0.5) 
    plt.clf() 
plt.close() 
    
