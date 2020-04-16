import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import h5py
from keras.models import model_from_json
from keras.datasets import mnist

loadData=True

inputs=['/models/mccikpc2/CPI-analysis/sae_mnist/model_epochs_50_sae_mnist']
    
if loadData:
    """
        Load the image data+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    print('Loading data...')
    # load images
    (X_train, y_train), (X_test, y_test) = mnist.load_data()

    x_train=X_train.reshape((x_train.shape[0],-1))
    x_test=X_test.reshape((x_test.shape[0],-1))

    x_train=x_train.astype('float32')/255.
    x_test=x_test.astype('float32')/255.

    print('data is loaded')
    """
        ----------------------------------------------------------------------------------
    """
    
    
digits=[0,1,2,3,4,5,6,7,8,9]
ims=[None]*100
for mo in inputs:
    """
        load the model++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    print('Loading model...')
    json_file = open(mo + '.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(mo + '.h5')
    print('model is loaded')
    """
        ----------------------------------------------------------------------------------
    """

    # Save images ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    for i in range(len(digits)):
        inds,=np.where((y_test==digits[i]) )
        # do 5 images
        for j in range(5):
            k=inds[j]
            # save raw image
            ims[i*10+j]=x_test[k,:].reshape((1,28,28,1))
            # encode - decode
            ed=loaded_model.predict(np.expand_dims(x_test[k,:],axis=0))
            # save processed image
            ims[i*10+j+5]=ed.reshape((1,28,28,1))
    #-------------------------------------------------------------------------------------
        


    """
        do plot on grids++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    plt.ion()
    fig = plt.figure(figsize=(10., 10.))
    grid = ImageGrid(fig, 111,  # similar to subplot(111)
                     nrows_ncols=(10, 10),  # creates 10x10 grid of axes
                     axes_pad=0.1,  # pad between axes in inch.
                     )
    for ax, im in zip(grid, ims): # ims is a list of images
        # Iterating over the grid returns the Axes.
        ax.imshow(im[0,:,:,0])

    plt.show()
    fig.savefig(mo + '.png')
    plt.close()
    """
        ----------------------------------------------------------------------------------
    """
