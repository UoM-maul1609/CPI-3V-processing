#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:05:02 2018

@author: mccikpc2
"""
#import gc
#from multiprocessing import Pool, cpu_count, Lock, Manager
#import time
#import numpy as np
#from tqdm import tqdm
#import os
#import globalLock

#def init():
#    globalLock.lock = Lock()

from joblib import Parallel, delayed, parallel_backend
from multiprocessing import Pool, cpu_count, Lock, Manager
import numpy as np
import time


def imageStatsDriver(path1,filename1,find_particle_edges,num_cores,p,m,l):
    print('====================particle properties===========================')
    #https://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments

    
    lf=len(filename1)
    #nc=cpu_count()
    nc=min([num_cores, lf])
    
    
    #ctx = set_start_method('forkserver',force=True)
    
    #p=Pool(processes=nc,maxtasksperchild=None)
    #m = Manager()
    #l = m.Lock()

    results=[]
    for i in range(lf): 
        # farm out to processors:
        #time.sleep(0.1)

        p.apply_async(mult_job, \
                        args=(path1,filename1[i],find_particle_edges,i,l)) # ,\
                        #callback=good,error_callback=bad) #np.mod(i,nc))))
        #results.append(r)
        
        #time.sleep(0.1)
    """
    for r in results:
        r.wait()
       
    """
    """
    # build the list / transpose    
    #https://stackoverflow.com/questions/6473679/transpose-list-of-lists
    list1=[[path1]*len(filename1),filename1,[find_particle_edges]*len(filename1),np.arange(lf)]
    list1=list(map(list,zip(*list1)))

    results = Parallel(n_jobs=nc)(delayed(mult_job)(i,j,k,n) for i,j,k,n in list1)
    """
    """
    output=[]
    for r in results:
        output.append(r.get())
        wait = time.time()+0.01
        while time.time() < wait:
            pass
    """
    
    p.close()
    p.join()
    
    

    """if os.path.exists("{0}{1}".format(path1,'output1.txt')):
        os.remove("{0}{1}".format(path1,'output1.txt'))
    """
    
    
    """
    p=Pool(processes=nc,maxtasksperchild=1)
    m = Manager()
    

    # build the list / transpose    
    #https://stackoverflow.com/questions/6473679/transpose-list-of-lists
    list1=[[path1]*len(filename1),filename1,[find_particle_edges]*len(filename1),np.arange(lf)]
    list1=list(map(list,zip(*list1)))
    
    # farm out to processors:
    p.map_async(mult_job,iterable=list1)

    p.close()
    p.join()
    """
    
    return



#def mult_job(list1): # path1, filename1, find_particle_edges
def mult_job(path1, filename1, find_particle_edges,position,lock):
    import scipy.io as sio
    from imageStats import imageStats
    #import sys
    """   
    path1=list1[0]
    filename1=list1[1]
    find_particle_edges=list1[2]
    position=list1[3]
    manager=list1[4]
    """
    
    # load from file
    #print("{0}{1}".format('Loading from file...',filename1))
    #sys.stdout.flush()
    dataload=sio.loadmat("{0}{1}".format(path1, filename1.replace('.roi','.mat')),
                       variable_names=['ROI_N','HOUSE','IMAGE1','BG'])
    ROI_N=dataload['ROI_N']
    HOUSE=dataload['HOUSE']
    IMAGE1=dataload['IMAGE1']
    BG=dataload['BG']
    #print("{0}{1}".format('Loaded ',filename1))
    #sys.stdout.flush()
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    
    # Particle properties +++++++++++++++++++++++++++++++++++++++++++++++++
    #print("{0}{1}".format('calculating particle properties...',filename1))
    #sys.stdout.flush()
    dat=imageStats(ROI_N,BG,find_particle_edges,position,filename1,lock)
    #print('done')
    #sys.stdout.flush()
    #----------------------------------------------------------------------
    
    # save to file
    #print("{0}{1}".format('Saving to file... ',filename1))
    #sys.stdout.flush()
    #with open(path1 + filename1[i].replace('.roi','.mat'),'ab') as f:
    #    sio.savemat(f, {'dat':dat})
    sio.savemat("{0}{1}".format(path1, filename1.replace('.roi','.mat')),
     {'ROI_N':ROI_N, 'HOUSE':HOUSE,'IMAGE1':IMAGE1,'BG':BG,'dat':dat})

    del dat, ROI_N, HOUSE, IMAGE1, BG, dataload, imageStats, sio

    # Garbage collection:
    #gc.collect()
    #del gc.garbage[:]

    #print("{0}{1}".format('Saved ',filename1))
    #sys.stdout.flush()
    
    return 1

def good(result):
    #globalLock.lock.acquire()
    #print("good" + str(result))
    #globalLock.lock.release()
    return
def bad(result):
    #globalLock.lock.acquire()
    #print("bad" + str(result))
    #globalLock.lock.release()
    return

