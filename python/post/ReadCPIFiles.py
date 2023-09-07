import numpy as np
import matplotlib.pyplot as plt
import datetime as dt
import glob
import scipy.io as sio
from tqdm import tqdm

"""
    This code loads times, focus, x, y, images and boundaries for each image
    sorts them by time, and returns them
"""
def ReadMAT(pathName='/Users/mccikpc2/Downloads/CPI_cals/ICEAnalogues/CPI'):
    times = np.zeros(0)
    foc   = np.zeros(0)
    len1  = np.zeros(0)
    x     = np.zeros(0)
    y     = np.zeros(0)
    im1   = np.zeros(0) # bin for the images
    boundaries   = np.zeros(0) # 
    
    files=glob.glob(pathName + '/*.mat')
    # remove timeseries.mat
    files = [ x1 for x1 in files if not("timeseries.mat") in x1 ] 
    # remove full_backgrounds.mat
    files = [ x1 for x1 in files if not("full_backgrounds.mat") in x1 ]

    for i in tqdm(range(len(files))):
        #print(files[i])
        dataload=sio.loadmat(files[i],
                           variable_names=['ROI_N','HOUSE','IMAGE1','BG','dat'])

        foc=np.concatenate([foc,np.squeeze(np.concatenate(dataload['dat']['foc'][0,0]['focus'][0]))])   
        times = np.concatenate([times,dataload['dat']['Time'][0][0][0,:]])
        len1=np.concatenate([len1,np.squeeze(dataload['dat']['len'][0][0][:,0])  ])
        #print(np.shape(dataload['ROI_N']['StartX'][0][0][:]))
        x=np.concatenate([x,0.5*(dataload['ROI_N']['StartX'][0][0][0,:] + \
             dataload['ROI_N']['EndX'][0][0][0,:])])
        y=np.concatenate([y,0.5*(dataload['ROI_N']['StartY'][0][0][0,:] + \
            dataload['ROI_N']['EndY'][0][0][0,:]) ])
            
        # images
        im1=np.concatenate([im1,dataload['ROI_N'][0,0]['IMAGE'][0,:]])
        boundaries=np.concatenate([boundaries,dataload['dat']['foc'][0,0]['boundaries'][0]])
    
    ind=np.argsort(times)
    times=times[ind]
    foc=foc[ind]
    len1=len1[ind]
    x=x[ind]
    y=y[ind]
    im1=im1[ind]
    boundaries=boundaries[ind]
    
    return (times,foc,len1,im1,boundaries,x,y)


if __name__=="__main__":
    (times,foc,len1,im1,boundaries,x,y)=ReadMAT()
