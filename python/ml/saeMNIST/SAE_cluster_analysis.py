import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import keras
from keras.models import Model, Sequential, Input
from keras.models import model_from_json
from SAE_MNIST_autoencoder_keras_with_clustering import target_distribution
from mpl_toolkits.axes_grid1 import ImageGrid 
from keras.datasets import mnist


mo='/models/mccikpc2/CPI-analysis/sae_mnist/model_epochs_300_sae_mnist'
loadData=True

if loadData:
    """
        Load the image data+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
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
    



# load the encoded data
h5f = h5py.File(mo + '_encoding.h5','r')
encoded=h5f['encoding'][:]
h5f.close()
# load the clustering data
h5f = h5py.File(mo + '_final_encoding.h5','r')
clustered=h5f['encoding'][:]
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
    if loaded_model.layers[i].name == 'encoder_3': 
        break 
ei=i+1



# now I just want the decoder part of the model
decoder = Sequential()
for i in range(ei,len(loaded_model.layers)):
    decoder.add(loaded_model.get_layer(index=i))
decoder.build(input_shape=loaded_model.layers[ei].input_shape) # (None,64)

decoder.summary()



y_pred=y_pred1



import seaborn as sns
import sklearn.metrics
import matplotlib.pyplot as plt
# sns.set(font_scale=3)
confusion_matrix = sklearn.metrics.confusion_matrix(y_test, y_pred)

plt.ion()
plt.figure(figsize=(16, 14))
sns.heatmap(confusion_matrix, annot=True, fmt="d", annot_kws={"size": 20});
plt.title("Confusion matrix", fontsize=30)
plt.ylabel('True label', fontsize=25)
plt.xlabel('Clustering label', fontsize=25)
plt.show()
plt.savefig(mo + '_confusion.png')
plt.close()


ims=[np.zeros((1,28,28,1))]*10
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
plt.ion()
fig = plt.figure(figsize=(10.,1.1)) 
for i in range(10): 
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
        ax.imshow(im.reshape((28,28))) 
        ax.set_title(ti)
    plt.show() 
    plt.savefig(mo + '_eval' + str(i).zfill(2) + '.png')

    plt.pause(0.5) 
    plt.clf() 
plt.close() 
    
