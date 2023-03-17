import matplotlib.pyplot as plt
import matplotlib as mpl  
import numpy as np
import h5py 

from sklearn.cluster import KMeans

from mpl_toolkits.axes_grid1 import ImageGrid
from sklearn.cluster import MiniBatchKMeans   
from sklearn.mixture import GaussianMixture
from yellowbrick.cluster import KElbowVisualizer

mpl.rcParams.update(mpl.rcParamsDefault) 
readIn=True

if readIn:
    print('Loading images....')
    h5f = h5py.File('/models/mccikpc2/DCMEX/CPI-analysis/postProcessed_t5_l50.h5','r')
    images=h5f['images'][:]
    h5f.close()
    print('Images loaded')
    
h5f = h5py.File('/models/mccikpc2/DCMEX/CPI-analysis/cnn/model_t5_epochs_100_dense64_3a_freeze_final_encoding.h5','r') 
cod2=h5f['encoding'][:]
h5f.close()

y_pred = cod2.argmax(1)
#y_pred = kmeans.predict(cod2)


print (np.unique(y_pred))
cat1=7;
ind,=np.where(y_pred==cat1) 
np.random.shuffle(ind)

ims=100*[None]
for j in range(100): 
    ims[j]=(images[ind[j]]) 

plt.ion()

fig=plt.figure(figsize=(10,10)) 
grid=ImageGrid(fig,111,nrows_ncols=(8,8),axes_pad=0.0,label_mode=None) 
i=0
for ax, im in zip(grid,ims):
    #ax.tick_params(labelbottom=False,labelleft=False) 
    ax.imshow(im[:,:]) 
    ax.set_xticks([-1])
    ax.set_yticks([-1])
    rounded=[round(num1,2) for num1 in cod2[ind[i]]]   
    ax.text(0.05,0.9,str(rounded[0:3]),transform=ax.transAxes,fontsize=5,color=[1, 1, 1]) 
    ax.text(0.1,0.8,str(rounded[3:6]),transform=ax.transAxes,fontsize=5,color=[1, 1, 1]) 
    ax.text(0.1,0.7,str(rounded[6:]),transform=ax.transAxes,fontsize=5,color=[1, 1, 1]) 
    i += 1
    
plt.savefig('/tmp/images' + str(cat1).zfill(2) + '.png')

