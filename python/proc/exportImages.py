#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:05:02 2018

@author: mccikpc2
"""
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import time
import os
from datetime import datetime
from datetime import timedelta
from tqdm import tqdm
import gc

def exportImages(pathname,filenames,foc_crit,size_thresh,MAP,cpiv1,classifier, \
    classifierFile,minClassSize, dropBins=[1,2,3,8],iceBins=[0,4,5,6,7],unclass=[-1]):

    if classifier == True:
        import keras
        from keras.models import Model, model_from_json, Sequential
        import h5py
        import sys
        from os import path
        # insert at 1, 0 is the script path (or '' in REPL)
        sys.path.insert(1, '../ml/cnn')
        sys.path.insert(1, '../ml')
        from DCNN_autoencoder_keras_with_clustering import ClusteringLayer 
    
        """
            load the model++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        print('Loading model...')
        json_file = open(classifierFile + '.json','r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json, \
            custom_objects={'ClusteringLayer':ClusteringLayer})
        loaded_model.load_weights(classifierFile + '.h5')
        print('model is loaded')
        """
            ------------------------------------------------------------------------------
        """
    
    
        """ 
            extract the images for classifying++++++++++++++++++++++++++++++++++++++++++++
        """
        print('extract images...')
        import postProcessImages05 
        imagePP=[]
        lensPP=[]
        indsPP=[]
        tot2=0
        for i in tqdm(range(len(filenames))):
            f=filenames[i]
            (imagePP1,lensPP1,timesPP1,diamPP1,roundPP1,l2wPP1,radiusPP1,indsPP1,tot1)= \
                postProcessImages05.postProcessing(\
                                        path.join(pathname,f.replace('.roi','.mat')), \
                                        path.join(pathname,'full_backgrounds.mat'),\
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
            ------------------------------------------------------------------------------
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
            ------------------------------------------------------------------------------
        """

    prefix='_pytgt'
    interxy=0.01
    maxx=0.
    maxy=1.
    j=1

    plt.ioff()
    plt.figure(figsize=(1024/200, 1280/200))
    #plt.ion()
    #plt.show()

    runx=0.
    runy=1.
    

    ilast=0    
    for l in range(len(filenames)):

   
        dataload=sio.loadmat("{0}{1}".format(pathname, filenames[l].replace('.roi','.mat')),
                           variable_names=['ROI_N','dat'])
        ROI_N=dataload['ROI_N']
        dat=dataload['dat']
        del dataload
        
        """
            get the classes of all this file++++++++++++++++++++++++++++++++++++++++++++++
        """
        if classifier==True:
            indc=np.mgrid[0:len(dat['foc'][0,0]['focus'][0,:])]+ilast
            ilast=ilast+len(indc)
            class2=class1[indc]
        """
            ------------------------------------------------------------------------------
        """

        if classifier==True:
            ind,=np.where(np.isin(class2,np.append(np.append(dropBins,iceBins),unclass)) \
                & (dat['len'][0,0][:,0]>size_thresh) & \
                          (dat['foc'][0,0]['focus'][0,:] >foc_crit))
        elif classifier==False:
            ind,=np.where( (dat['len'][0,0][:,0]>size_thresh) & \
                          (dat['foc'][0,0]['focus'][0,:] >foc_crit) )
        
        if(len(ind)==0):
            continue
        
        
        # loop over all the images in this file
        i=0
        #https://stackoverflow.com/questions/45808140/using-tqdm-progress-bar-in-a-while-loop
        tqdm.monitor_interval = 0
        pbar=tqdm(total=len(dat['foc'][0,0]['focus'][0]))
        while(i<len(dat['foc'][0,0]['focus'][0])):
            if np.mod(i+1,10)==0:
                pbar.update(10)
            elif i+1==len(dat['foc'][0,0]['focus'][0]):
                pbar.n=len(dat['foc'][0,0]['focus'][0])
                pbar.update()
                
            
            # check to see if criteria are met
            if not np.isin(i,ind):
                continue
#             if ((dat['len'][0,0][i,0]<size_thresh) or 
#                 (dat['foc'][0,0]['focus'][0,i].item() <=foc_crit) ):
#                 i=i+1
#                 continue
            
            if(j==1):
                time1=ROI_N['Time'][0,0][i,0]
                dayold=np.floor(time1)
                pytime=datetime.fromordinal(int(time1)) + \
                     timedelta(days=time1%1) - \
                     timedelta(days = 366)
                str1=str(pytime.time())
                hour1=pytime.hour
                filename1="{0}{1}{2}{3}{4}{5}".format(datetime.strptime( \
                    str(pytime.date()), '%Y-%m-%d').strftime('%m_%d_%y'), \
                    '_', str(pytime.time())[0:8].replace(':','_'), '.', \
                    str('%03d' % int(pytime.microsecond/1000)), '.png')
                


            time1=ROI_N['Time'][0,0][i,0]
            pytime=datetime.fromordinal(int(time1)) + \
                 timedelta(days=time1%1) - \
                 timedelta(days = 366)
            str1=str(pytime.time())
            hour1=pytime.hour
            daynew=np.floor(time1)



            (r,c)=np.shape(ROI_N['IMAGE'][0,0][0,i]['IM'][0,0])
            h=plt.axes([runx, runy-r/(1280), c/(1024), r/(1280)])
            runx=runx+c/1024+interxy
            maxy=np.min([maxy,runy-r/1280-interxy])
            
            
            if ((maxy<-interxy) or (daynew != dayold)):
                h.remove()
                #time.sleep(1)
                maxx=0.
                maxy=1.
                runx=0.
                runy=1.   
                
                if classifier==True: 
                    # file output
                    if not os.path.exists("{0}{1}{2}{3}".format(pathname, filename1[0:8],'_class','')):
                        os.makedirs("{0}{1}{2}{3}".format(pathname, filename1[0:8],'_class',''))
                    plt.savefig("{0}{1}{2}{3}{4}{5}".format(pathname, filename1[0:8],'_class', \
                                '', '/', filename1),dpi=300)
                elif classifier==False: 
                    # file output
                    if not os.path.exists("{0}{1}{2}{3}".format(pathname, filename1[0:8],prefix,str(size_thresh))):
                        os.makedirs("{0}{1}{2}{3}".format(pathname, filename1[0:8],prefix,str(size_thresh)))
                    plt.savefig("{0}{1}{2}{3}{4}{5}".format(pathname, filename1[0:8],prefix, \
                                str(size_thresh), '/', filename1),dpi=300)
    
                plt.close()
                plt.figure(figsize=(1024/200, 1280/200))
                j=1
                daynew=dayold
                continue
            
            
            
            if runx<= (1.+interxy):
                if classifier==True:
                    # drops
                    if(np.isin(class2[i],dropBins)):
                        h.imshow(ROI_N['IMAGE'][0,0][0,i]['IM'][0,0],cmap='Greys_r')
                    # ice
                    elif(np.isin(class2[i],iceBins)):
                        h.imshow(ROI_N['IMAGE'][0,0][0,i]['IM'][0,0],cmap='Blues_r')
                    # unclassified
                    elif(np.isin(class2[i],unclass)):
                        h.imshow(ROI_N['IMAGE'][0,0][0,i]['IM'][0,0],cmap='Reds_r')
                elif classifier==False:
                    h.imshow(ROI_N['IMAGE'][0,0][0,i]['IM'][0,0],cmap='Blues_r')
                h.axis('off')
                
                # this plots the boundary on the image
                #h.plot(dat[0,0]['foc'][0,i]['boundaries'][:,1], \
                #    dat[0,0]['foc'][0,i]['boundaries'][:,0],'r') 
                
                time1=ROI_N['Time'][0,0][i,0]
                pytime=datetime.fromordinal(int(time1)) + \
                     timedelta(days=time1%1) - \
                     timedelta(days = 366)
                str1=str(pytime.time())
                hour1=pytime.hour
                h.text(0,0.95,str1[0:12],ha='left',va='center', \
                       transform=h.transAxes,fontsize=2)
                h.text(0,0.05,"{0}{1}".format(str('%d' % dat['len'][0,0][i,0]), 'um'), \
                       ha='left',va='center',transform=h.transAxes,fontsize=3)

                i=i+1
                j=j+1
            elif(runx>(1+interxy)):
                h.remove()
                runx=0.
                runy=maxy
                maxy=1.

        del ROI_N, dat
        # Garbage collection:
        gc.collect()
        del gc.garbage[:]
    
        pbar.close()
        del pbar
       
    if classifier==True: 
        # file output
        if not os.path.exists("{0}{1}{2}{3}".format(pathname, filename1[0:8],'_class','')):
            os.makedirs("{0}{1}{2}{3}".format(pathname, filename1[0:8],'_class',''))
        plt.savefig("{0}{1}{2}{3}{4}{5}".format(pathname, filename1[0:8],'_class', \
                    '', '/', filename1),dpi=300)
    elif classifier==False:
        # file output
        if not os.path.exists("{0}{1}{2}{3}".format(pathname, filename1[0:8],prefix,str(size_thresh))):
            os.makedirs("{0}{1}{2}{3}".format(pathname, filename1[0:8],prefix,str(size_thresh)))
        plt.savefig("{0}{1}{2}{3}{4}{5}".format(pathname, filename1[0:8],prefix, \
                    str(size_thresh), '/', filename1),dpi=300)
    plt.close()
            

    
    return
    
