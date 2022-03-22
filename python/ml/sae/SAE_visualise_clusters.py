from sklearn.manifold import TSNE 
import h5py
from matplotlib import cm
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans


# project based on autoencoder encoding - see what natural variation there is spatially
#h5f = h5py.File('/models/mccikpc2/CPI-analysis/sae/model_epochs_50_sae_prop05_10_encoding.h5','r')
h5f = h5py.File('/models/mccikpc2/CPI-analysis/sae/model_epochs_50_sae05_50_final_encoding.h5','r')

cod1=h5f['encoding'][:]
#test_predictions = KMeans(n_clusters=5).fit_predict(cod1)

i1=len(cod1)
i2=1000
indices = np.random.permutation(i1)
cod1=cod1[indices[:i2]]

tsne = TSNE(2, verbose=1)
tsne_proj = tsne.fit_transform(cod1)

h5f.close()


# colour based on cluster layer encoding
#h5f = h5py.File('/models/mccikpc2/CPI-analysis/sae/model_epochs_50_sae_prop05_10_final_encoding.h5','r')
h5f = h5py.File('/models/mccikpc2/CPI-analysis/sae/model_epochs_50_sae05_50_final_encoding.h5','r')
cod1=h5f['encoding'][:]
#cod1=cod1[indices[:i2]]
h5f.close()

test_predictions = cod1.argmax(1)
#inds,=np.where(indices[:i1])
inds=indices[:i2] 
test_predictions1=test_predictions[inds]

plt.ion()

cmap = cm.get_cmap('tab20',5) 
fig, ax = plt.subplots(figsize=(8,8)) 
num_categories = 5
col1=['r','g','b','k','m']
for lab in range(num_categories): 
    indices = test_predictions1==lab 
    ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=col1[lab],
    label = lab ,alpha=0.5) 
ax.legend(fontsize='large', markerscale=2) 
plt.show()                                                                                    
