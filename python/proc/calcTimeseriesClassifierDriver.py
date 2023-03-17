#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:05:02 2018

@author: mccikpc2
"""
import scipy.io as sio
import numpy as np
import os
from scipy.interpolate import interp1d
from os import path

import keras
from keras.models import Model, model_from_json, Sequential
import h5py
from tqdm import tqdm

import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '../ml/cnn')
sys.path.insert(1, '../ml')
from DCNN_autoencoder_keras_with_clustering import ClusteringLayer 



def calcTimeseriesClassifierDriver(path1,filename1,foc_crit,dt,ds,vel,outputfile,cpiv1,classifierFile,minClassSize):
    
    """
        load the model++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    """
    print('Loading model...')
    json_file = open(classifierFile + '.json','r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json, custom_objects={'ClusteringLayer':ClusteringLayer})
    loaded_model.load_weights(classifierFile + '.h5')
    print('model is loaded')
    """
        ----------------------------------------------------------------------------------
    """
    
    
    """ 
        extract the images for classifying
    """
    print('extract images...')
    import postProcessImages05 
    imagePP=[]
    lensPP=[]
    indsPP=[]
    tot2=0
    for i in tqdm(range(len(filename1))):
        f=filename1[i]
        (imagePP1,lensPP1,timesPP1,diamPP1,roundPP1,l2wPP1,radiusPP1,indsPP1,tot1)=postProcessImages05.postProcessing(\
                                    path.join(path1,f.replace('.roi','.mat')), \
                                    path.join(path1,'full_backgrounds.mat'),\
                                    foc_crit,minClassSize)
        indsPP1=indsPP1+tot2
        imagePP.extend(imagePP1)
        lensPP.extend(lensPP1)
        indsPP.extend(indsPP1)  
        tot2 += tot1  
        
    imagePP=np.stack(imagePP,axis=0)
    indsPP=np.stack(indsPP,axis=0)
    print('extracted')    
    """
        ----------------------------------------------------------------------------------
    """
     
    """
        classify the images using model
    """
    print('classifying...')
    imagePP=np.expand_dims(imagePP,axis=3)
    imagePP = imagePP.astype('float16') / 255.
    cod2=loaded_model.predict(imagePP)
    y_pred=cod2.argmax(1)
    class1=-np.ones((tot2)) # all -1 to start with
    test=loaded_model.get_config()
    n_clusters=test['layers'][7]['config']['n_clusters']  
    
    for i in range(n_clusters):
        ind,=np.where(y_pred==i)
        class1[indsPP[ind]]=i
    print('classified')    
    """
        ----------------------------------------------------------------------------------
    """
    
    
    # note, numpy.histogramdd might be better
    save_files=True
    sa=1280.*1024./np.sqrt(2.)*2.3e-6**2  # sample area of image perp to flow
    sv=sa*np.sqrt(2.)*3e-3             # sample volume of one image

    dataload=sio.loadmat("{0}{1}".format(path1, 'full_backgrounds.mat'),\
                         variable_names=['t_range'])
    t_range=dataload['t_range']

    print('====================calculating timeseries=========================')


    print('Set-up arrays')
    dt2=dt/86400.
    Time=np.mgrid[t_range[0,0]:t_range[0,1]+dt2:dt2]
    size1=np.mgrid[0:2300+ds:ds]
    size1a=np.mgrid[0:2300+ds*2.:ds]
    ar1=np.mgrid[0:1.2:0.2]
    nt=len(Time)
    nl=len(size1)
    na=len(ar1)
    na=n_clusters+1 # 4 drops in one, 5 actual ice, 1 unclassified 
    
    
    timeser={'Time':Time,
             'size1':size1,
             'size2':size1+ds,
             'midsize':(size1+size1+ds)/2.,
             'ar1':ar1,
             'ar2':ar1+0.2,
             'conc2':np.zeros((nt,nl)),
             'conc':np.zeros((nt,1)),
             'deadtimes':np.zeros((nt,1)),
             'nimages':np.zeros((nt,1)),
             'conc2ar':np.zeros((nt,nl,na))}
    array=np.zeros((nt,nl))




    test1=0
    ilast=0
    for i in range(len(filename1)):
        # load from file
        print('Loading from file...')
        dataload=sio.loadmat("{0}{1}".format(path1, filename1[i].replace('.roi','.mat')),
                           variable_names=['ROI_N','HOUSE','IMAGE1','BG','dat'])
        ROI_N=dataload['ROI_N']
        HOUSE=dataload['HOUSE']
        IMAGE1=dataload['IMAGE1']
        BG=dataload['BG']
        dat=dataload['dat']
        print('done')
        test1=test1+len(dat['foc'][0,0]['focus'][0,:])
        indc=np.mgrid[0:len(dat['foc'][0,0]['focus'][0,:])]+ilast
        ilast=ilast+len(indc)
        class2=class1[indc]
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        
        
        # Calculate timeseries ++++++++++++++++++++++++++++++++++++++++++++++++
        print('Calculate timeseries...')

        if cpiv1:
            ind,=np.where((ROI_N['imageType'][0,0][:,0] == 33857) & \
                      (dat['foc'][0,0]['focus'][0,:] >foc_crit))
        else:
            ind,=np.where((ROI_N['imageType'][0,0][:,0] == 89) & \
                      (dat['foc'][0,0]['focus'][0,:] >foc_crit))
                      
        
        if(len(ind)==0):
            continue
        tmin=np.min(ROI_N['Time'][0,0][ind,0])
        tmax=np.max(ROI_N['Time'][0,0][ind,0])
        ilow,=np.where(timeser['Time']>=tmin)
        ilow=ilow[0]
        ihigh,=np.where(timeser['Time']<tmax)
        ihigh=ihigh[-1]
        
        
        
        
        #=======time loop======================================================
        f=interp1d(IMAGE1['Time1'][0,0][:,0], \
                   IMAGE1['Time'][0,0][0,:].T, \
                   kind='linear',fill_value='extrapolate')
        
        for j in range(ilow,ihigh+1):
            # dead-time: find imge data that are in the time-window
                        # find corresponding funny time of tiem-windown
                        # add up dead-times
            indim,=np.where((IMAGE1['Time1'][0,0][:,0]>=(timeser['Time'][j]-dt2/2.)) & \
                      (IMAGE1['Time1'][0,0][:,0]<(timeser['Time'][j]+dt2/2.)))
            # interpolation:
            try:
                if not cpiv1:
                    twin1=np.array([timeser['Time'][j]-dt2/2., timeser['Time'][j]+dt2/2.])
                    twin=f(np.array([timeser['Time'][j]-dt2/2., timeser['Time'][j]+dt2/2.]))
                    tclock = (np.max(IMAGE1['Time'][0,0][0,:])-np.min(IMAGE1['Time'][0,0][0,:]))/ \
                        (np.max(IMAGE1['Time1'][0,0][:,0])-np.min(IMAGE1['Time1'][0,0][:,0]))* \
                            (twin1 - np.min(IMAGE1['Time1'][0,0][:,0])) + np.min(IMAGE1['Time'][0,0][0,:])
                    indho,=np.where((HOUSE['Time'][0,0][0,:]  >=twin[0]) & \
                                    (HOUSE['Time'][0,0][0,:]< twin[1]))
                    timeser['deadtimes'][j]=np.nansum(HOUSE['deadtime'][0,0][indho,0])
                else:
                    indho,=np.where(\
                        (HOUSE['Time'][0,0][0,:]/86400.-\
                         np.floor(HOUSE['Time'][0,0][0,:]/86400.)+\
                             np.floor(IMAGE1['Time1'][0,0][0,:]) \
                             >=(timeser['Time'][j]-dt2/2.)) & \
                        (HOUSE['Time'][0,0][0,:]/86400.-\
                         np.floor(HOUSE['Time'][0,0][0,:]/86400.)+\
                             np.floor(IMAGE1['Time1'][0,0][0,:]) \
                             < (timeser['Time'][j]+dt2/2.)))
                    timeser['deadtimes'][j]=np.nansum(HOUSE['deadtime'][0,0][indho,0])
            except:
                timeser['deadtimes'][j]=0.0

            timeser['nimages'][j]=len(indim)
            
            
            
            # particles:
            ind2,=np.where((dat['Time'][0,0][0,ind]>=(timeser['Time'][j]-dt2/2.)) & \
                (dat['Time'][0,0][0,ind]<(timeser['Time'][j]+dt2/2)))
            timeser['conc'][j]=timeser['conc'][j]+len(ind2)
            
            
            
            # size bins========================================================
            (N,X)=np.histogram(dat['len'][0,0][ind[ind2],0],size1a)
            timeser['conc2'][j,:]=timeser['conc2'][j,:]+N
            #------------------------------------------------------------------
            
            # size and roundness bins==========================================
            for k in range(0,na):
                ind3=ind[ind2]
                ind4,=np.where(class2[ind3] == k-1  ) #  so the first one k, is equal to -1 , which is unclassified. The rest are in order
                    
                
                (N,X)=np.histogram(dat['len'][0,0][ind3[ind4],0],size1a)
                timeser['conc2ar'][j,:,k]=timeser['conc2ar'][j,:,k]+N     
            #------------------------------------------------------------------
            
        #----------------------------------------------------------------------
            
            
            
            
    # scale by sample volume
    dead=np.transpose(np.tile(timeser['deadtimes'].T,(nl,na,1)), (2,0,1))
    nimages=np.transpose(np.tile(timeser['nimages'].T,(nl,na,1)), (2,0,1))
    timeser['conc2ar']=timeser['conc2ar']/((dt-dead)*vel*sa+nimages*sv)
    
    timeser['conc2']=np.nansum(timeser['conc2ar'],axis=2)
    timeser['conc']=np.nansum(timeser['conc2'],axis=1)
    
    
    print(str(test1) + ' ' + str(tot2))
    
    print('done')
    #----------------------------------------------------------------------
    
    
    if save_files:
        # save to file
        print('Saving to file...')
        if os.path.exists("{0}{1}".format(path1, outputfile)):
            # save / append
            #print("should be appending")
            sio.savemat("{0}{1}".format(path1, outputfile),{'timeser':timeser})  
        else:
            sio.savemat("{0}{1}".format(path1, outputfile),{'timeser':timeser})  
            
        print('done')

    return
    
    
    
