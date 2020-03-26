from sklearn.cluster import KMeans

loadData=True

if loadData:
    print('Loading data...')
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



# 10 clusters
n_clusters = 10
# Runs in parallel 4 CPUs
kmeans = KMeans(n_clusters=n_clusters, n_init=20, n_jobs=4)
# Train K-Means.
y_pred_kmeans = kmeans.fit_predict(x)
# Evaluate the K-Means clustering accuracy.
metrics.acc(y, y_pred_kmeans)