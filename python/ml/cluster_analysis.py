import h5py
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import keras
from keras.models import Model, Sequential, Input
from keras.models import model_from_json

mo='/tmp/model_epochs_50_dense64'


# load the encoded data
h5f = h5py.File(mo + '_encoding.h5','r')
encoded=h5f['encoding'][:]



# perform k-means
print('Performing k-means')
y_pred = KMeans(n_clusters=64).fit_predict(encoded)
print('done k-means')





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

plt.ion()
for i in range(0,100,1): 
    img=decoder.predict(np.expand_dims(encoded[i,:],axis=0))  
    plt.imshow(img[0,:,:,0]*255)  
    plt.title('class: ' + str(y_pred[i])) 
    plt.pause(1) 
    