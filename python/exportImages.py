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

def exportImages(pathname,filenames,foc_crit,size_thresh,MAP):

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
    
    tqdm.monitor_interval = 0

    
    for l in range(len(filenames)):

   
        dataload=sio.loadmat("{0}{1}".format(pathname, filenames[l].replace('.roi','.mat')),
                           variable_names=['ROI_N','dat'])
        ROI_N=dataload['ROI_N']
        dat=dataload['dat']
        del dataload

        ind,=np.where( (dat['len'][0,0][:,0]>size_thresh) & \
                      (dat['foc'][0,0]['focus'][0,:] >foc_crit) )
        if(len(ind)==0):
            continue
        
        
        # loop over all the images in this file
        i=0
        #https://stackoverflow.com/questions/45808140/using-tqdm-progress-bar-in-a-while-loop
        pbar=tqdm(total=len(dat['foc'][0,0]['focus'][0]))
        
        while(i<len(dat['foc'][0,0]['focus'][0])):
            pbar.update(1)
            # check to see if criteria are met
            if ((dat['len'][0,0][i,0]<size_thresh) or 
                (np.asscalar(dat['foc'][0,0]['focus'][0,i]) <=foc_crit) ):
                i=i+1
                continue
            
            if(j==1):
                time1=ROI_N['Time'][0,0][i,0]
                dayold=np.floor(time1)
                pytime=datetime.fromordinal(int(time1)) + \
                     timedelta(days=time1%1) - \
                     timedelta(days = 366)
                str1=str(pytime.time())
                hour1=pytime.hour
                filename1="{0}{1}{2}{3}{4}{5}".format(datetime.strptime(str(pytime.date()), '%Y-%m-%d').strftime('%m_%d_%y'), \
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
                
                # file output
                if not os.path.exists("{0}{1}{2}{3}".format(pathname, filename1[0:8],'_pygt',str(size_thresh))):
                    os.makedirs("{0}{1}{2}{3}".format(pathname, filename1[0:8],'_pygt',str(size_thresh)))
                plt.savefig("{0}{1}{2}{3}{4}{5}".format(pathname, filename1[0:8],'_pygt', \
                            str(size_thresh), '/', filename1),dpi=300)

                plt.close()
                plt.figure(figsize=(1024/200, 1280/200))
                j=1
                daynew=dayold
                continue
            
            
            
            if runx<= (1.+interxy):
                h.imshow(ROI_N['IMAGE'][0,0][0,i]['IM'][0,0],cmap='Blues_r')
                h.axis('off')
                
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
    
        if((l+1)==len(filenames)):
            # file output
            if not os.path.exists("{0}{1}{2}{3}".format(pathname, filename1[0:8],'_pygt',str(size_thresh))):
                os.makedirs("{0}{1}{2}{3}".format(pathname, filename1[0:8],'_pygt',str(size_thresh)))
            plt.savefig("{0}{1}{2}{3}{4}{5}".format(pathname, filename1[0:8],'_pygt', \
                        str(size_thresh), '/', filename1),dpi=300)
            plt.close()
            
            pbar.close()

    
    return
    
