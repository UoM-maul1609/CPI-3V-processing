import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import ImageGrid
import numpy as np


loadData=True

inputs='/models/mccikpc2/CPI-analysis/model_epochs_15_dense64'

if loadData:
    """
        Load the image data+++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    print('Loading image data...')
    # load images
    h5f = h5py.File('/models/mccikpc2/CPI-analysis/postProcessed_l50.h5','r')
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

    print('image data is loaded')
    """
        ----------------------------------------------------------------------------------
    """
    
    
    """
        load the model++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    print('Loading model...')
    json_file = open(inputs + '.json','r')
    loaded_model_hson = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    loaded_model.load_weights(inputs + '.h5')
    print('model is loaded')
    """
        ----------------------------------------------------------------------------------
    """

# Save images ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
size_bins=[0,100,250,500,750,1000,1250,1500,1750,2000,2250]
ims=[None]*100
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
#-----------------------------------------------------------------------------------------
        


"""
    do plot on grids
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