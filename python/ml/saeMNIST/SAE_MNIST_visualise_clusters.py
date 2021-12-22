from sklearn.manifold import TSNE 
import h5py
from matplotlib import cm
import matplotlib.pyplot as plt

h5f = h5py.File('/models/mccikpc2/CPI-analysis/sae_mnist/model_epochs_300_sae_mnist_final_encoding.h5','r')

cod1=h5f['encoding'][:]
tsne = TSNE(2, verbose=1)
tsne_proj = tsne.fit_transform(cod1)
test_predictions = cod1.argmax(1)


plt.ion()

cmap = cm.get_cmap('tab20') 
fig, ax = plt.subplots(figsize=(8,8)) 
num_categories = 10 
for lab in range(num_categories): 
    indices = test_predictions==lab 
    ax.scatter(tsne_proj[indices,0],tsne_proj[indices,1], c=np.array(cmap(lab)).reshape(1,4),
    label = lab ,alpha=0.5) 
ax.legend(fontsize='large', markerscale=2) 
plt.show()                                                                                    
