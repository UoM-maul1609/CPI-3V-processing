import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np
import h5py
from keras.models import model_from_json

loadData=True
auxLoad=False

inputs=['/models/mccikpc2/CPI-analysis/cnn/model_epochs_5_dense16' , \
    '/models/mccikpc2/CPI-analysis/cnn/model_epochs_50_dense16' , \
    '/models/mccikpc2/CPI-analysis/cnn/model_epochs_5_dense64' , \
    '/models/mccikpc2/CPI-analysis/cnn/model_epochs_15_dense64', \
    '/models/mccikpc2/CPI-analysis/cnn/model_epochs_50_dense64',\
    '/models/mccikpc2/CPI-analysis/cnn/model_t2_epochs_50_dense64',\
    '/models/mccikpc2/CPI-analysis/cnn/model_t4_epochs_50_dense64']
    
    
dataFiles=['/models/mccikpc2/CPI-analysis/postProcessed_l50.h5', \
        '/models/mccikpc2/CPI-analysis/postProcessed_l50.h5', \
        '/models/mccikpc2/CPI-analysis/postProcessed_l50.h5', \
        '/models/mccikpc2/CPI-analysis/postProcessed_l50.h5', \
        '/models/mccikpc2/CPI-analysis/postProcessed_l50.h5', \
        '/models/mccikpc2/CPI-analysis/postProcessed_t2_l50.h5',\
        '/models/mccikpc2/CPI-analysis/postProcessed_t4_l50.h5']

inputs=['/models/mccikpc2/DCMEX/CPI-analysis/cnn/model_t5_epochs_100_dense64_3a']
dataFiles=['/models/mccikpc2/DCMEX/CPI-analysis/postProcessed_t5_l50.h5']

size_bins=[0,100,200,300,400,500,600,700,800,900,1000]
ims=[None]*100
ii=0
dataFileOld=''
for mo in inputs:

    if loadData:
        """
            Load the image data+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        print('Loading image data...')
        # load images
        dataFile=dataFiles[ii]
        if dataFile != dataFileOld:
            dataFileOld = dataFile
            
            h5f = h5py.File(dataFile,'r')
            images=h5f['images'][:]
            images=np.expand_dims(images,axis=3)
            lens  =h5f['lens'][:]
            times =h5f['times'][:]
            h5f.close()
            
            i1 = len(lens)
                
            #split1=int(0.8*i1)
            if ii == 0:
                if ~auxLoad:
                    i11=100000
                    i22=10000
                    indices = np.random.permutation(i1)
                    #split1=int(0.8*i1)
                    training_idx, test_idx = indices[:i11], indices[i11:i11+i22]
                else:
                    h5faux = h5py.File(inputs + '_aux.h5','r')
                    training_idx=h5faux['training_idx'][:]
                    test_idx =h5faux['test_idx'][:]
                    h5faux.close()
    
            x_train, x_test = images[training_idx,:,:], images[test_idx,:,:]
            del images
            x_train=x_train.astype('float32')/255.
            x_test=x_test.astype('float32')/255.

            lens_train,lens_test = lens[training_idx], lens[test_idx]
            times_train,times_test = times[training_idx], times[test_idx]

        print('image data is loaded')
        ii += 1
        """
            ------------------------------------------------------------------------------
        """
    
    
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
    for i in range(len(size_bins)-1):
        inds,=np.where((lens_train>size_bins[i]) & (lens_train<=size_bins[i+1]))
        # do 5 images
        for j in range(5):
            k=inds[j]
            # save raw image
            ims[i*10+j]=x_train[k:k+1,:,:,0:1]
            # encode - decode
            ed=loaded_model.predict(x_train[k:k+1,:,:,0:1])
            # save processed image
            ims[i*10+j+5]=ed
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
