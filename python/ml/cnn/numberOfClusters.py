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

    
h5f = h5py.File('/models/mccikpc2/CPI-analysis/cnn/model_t5_epochs_50_dense64_3a_encoding.h5','r') 
cod1=h5f['encoding'][:]
h5f.close()


h5f = h5py.File('/models/mccikpc2/CPI-analysis/cnn/model_t5_epochs_50_dense64_3a_aux.h5','r') 
test_idx=h5f['test_idx'][:] 
h5f.close()

model = KMeans() 
plt.ion()
plt.show()

# elbow plot
plt.figure()
visualizer = KElbowVisualizer(model,k=(2,30),timings=True,verbose=1)
visualizer.fit(cod1[test_idx]) 


# silouette plot
plt.figure()
visualizer = KElbowVisualizer(model,k=(2,30), metric='silhouette',timings=True,verbose=1)
visualizer.fit(cod1[test_idx]) 


# gaussian mixtures BIC & AIC
# https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
plt.figure()
n_components = np.arange(1, 21)
models = [GaussianMixture(n, covariance_type='full', random_state=0).fit(cod1[test_idx])
          for n in n_components]

plt.plot(n_components, [m.bic(cod1[test_idx]) for m in models], label='BIC')
plt.plot(n_components, [m.aic(cod1[test_idx]) for m in models], label='AIC')
plt.legend(loc='best')
plt.xlabel('n_components');
