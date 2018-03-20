#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:05:02 2018

@author: mccikpc2
"""
import scipy.io as sio
from imageStats import imageStats
import gc
from multiprocessing import Pool, cpu_count
import time
import numpy as np

results = []

def imageStatsDriver(path1,filename1,find_particle_edges,num_cores=cpu_count()):
    print('====================particle properties===========================')
    #https://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments
    """for i in range(len(filename1)):
        p=Pool(processes=1)

        p.apply_async(mult_job,(path1,filename1[i],find_particle_edges))

        p.close()
        p.join()
        del p
    """
    
    lf=len(filename1)
    #nc=cpu_count()
    nc=num_cores
    fpc=np.ceil(lf/nc).astype(int)
    
    for i in range(fpc): # number of chunks
        p=Pool(processes=nc)
        
        # farm out to processors:
        for j in range(nc): # number of files in a chunk
            if (i+1)*(j+1) > lf:
                continue
            fn=filename1[j+i*nc]
            p.apply_async(mult_job,args=(path1,fn,find_particle_edges), 
                          callback=collect_results)

        p.close()
        p.join()
        del p
    
    
    return

#https://sedeh.github.io/python-pandas-multiprocessing-workaround.html
def collect_results(result):
    """Uses apply_async's callback to setup up a separate Queue for each process"""
    results.extend(result)


def mult_job(path1,filename1,find_particle_edges):
    # load from file
    print('Loading from file...')
    dataload=sio.loadmat("{0}{1}".format(path1, filename1.replace('.roi','.mat')),
                       variable_names=['ROI_N','HOUSE','IMAGE1','BG'])
    ROI_N=dataload['ROI_N']
    HOUSE=dataload['HOUSE']
    IMAGE1=dataload['IMAGE1']
    BG=dataload['BG']
    print('done')

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    
    
    # Particle properties +++++++++++++++++++++++++++++++++++++++++++++++++
    print("{0}{1}".format('calculating particle properties...',filename1))
    dat=imageStats(ROI_N,BG,find_particle_edges)
    print('done')
    #----------------------------------------------------------------------
    
    
    # save to file
    print('Saving to file...')
    #with open(path1 + filename1[i].replace('.roi','.mat'),'ab') as f:
    #    sio.savemat(f, {'dat':dat})
    sio.savemat("{0}{1}".format(path1, filename1.replace('.roi','.mat')),
     {'ROI_N':ROI_N, 'HOUSE':HOUSE,'IMAGE1':IMAGE1,'BG':BG,'dat':dat})

    del dat, ROI_N, HOUSE, IMAGE1, BG, dataload

    # Garbage collection:
    gc.collect()
    del gc.garbage[:]

    print('done')
    
    return