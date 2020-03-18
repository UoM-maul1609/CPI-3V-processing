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
"""
from os import environ
environ["OMP_NUM_THREADS"]="1"
environ["OPENBLAS_NUM_THREADS"]="1"
environ["MKL_NUM_THREADS"]="1"
environ["VECLIB_MAXIMUM_THREADS"]="1"
environ["NUMEXPR_NUM_THREADS"]="1"
from joblib import Parallel, delayed, parallel_backend
"""
from multiprocessing import TimeoutError, Pool, Process, cpu_count, Lock, Manager
#import numpy as np
from math import ceil
from time import sleep
from tqdm import tqdm

def imageStatsDriver(path1,filename1,find_particle_edges,num_cores):
    import subprocess
    import sys
    import os
    import time
    print('====================particle properties===========================')
    #https://stackoverflow.com/questions/5442910/python-multiprocessing-pool-map-for-multiple-arguments

    
    lf=len(filename1)
    #nc=cpu_count()
    nc=min([num_cores, lf])
    

    """
        submit dummy jobs using subprocess
    """
    p=[None]*nc
    log=[None]*nc
    submission=[None]*nc
    for i in range(nc):
        p[i] = subprocess.Popen('', shell=True)
    time.sleep(1)

    """
       main loop 
    """
    i = 0
    iter1=0
    while True:
        runsProcessing = False
        # go through all cores
        for j in range(nc):
            # submit a new run
            if p[j].poll() == 0 and i < lf:
                #if iter1 > 0:
                    # close file
                #    log[j].close()
                
                #log[j] = open('/tmp/' + filename1[i].replace('.roi','.out'),'wb')
                #log[j].flush()
                # run command
                runs=sys.executable + ' ' + os.path.abspath(__file__) + ' ' + path1 + ' ' + \
                    filename1[i] + ' ' + str(find_particle_edges) + ' ' + str(i)  
                submission[j]=runs
                print(runs)
                # submit the job
                #p[j]=subprocess.Popen(submission[j], shell=True,stdout=log[j],stderr=log[j])                
                p[j]=subprocess.Popen(runs, shell=True)
                i += 1

        
            # resubmit if it didn't work
            if (p[j].poll() != 0) and (p[j].poll() != None):
                print('resubmitting')
                p[j].terminate()
                #p[j]=subprocess.Popen(submission[j], shell=True,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,stdout=log[j],stderr=log[j])
                p[j]=subprocess.Popen(submission[j], shell=True)

            # check if runs are still going
            runsProcessing = runsProcessing or (p[j].poll() != 0)

        # break out if the runs are not processing and we are past the last file
        if not(runsProcessing) and (i>=lf):
            break
        iter1 += 1
        time.sleep(10)
            

            


    
    """
    m = Manager()
    l = m.RLock()

    
    nchunks=1
    nelem=ceil(lf / nchunks)
    fns = [filename1[i * nelem:(i + 1) * nelem] for i in range((len(filename1) + nelem - 1) // nelem )] 
    p=[]
    k=0
    for n in range(nchunks):
        ncc = int(nc / nchunks)
        p.append(Pool(processes=ncc,maxtasksperchild=1))

        results=[]
        for i in range(len(fns[n])): 
            # farm out to processors:
            r=p[n].apply_async(mult_job, \
                        args=(path1,fns[n][i],find_particle_edges,k,l),callback=good,error_callback=bad) # ,\
         
            k += 1
            results.append(r)
    
    
    ###########################################################################
    notReady=[True]*lf
    while True:
        for i in range(len(results)):
            try:
                results[i].get(timeout=0.01)
                if notReady[i] == True:
                    fp = open('/tmp/pyout','a')
                    fp.write(filename1[i] + 'done')
                    fp.close()
                notReady[i]=False
            except TimeoutError:
                notReady[i]=True
                
        # to break out we want all elements to be false
        # if the are all false then any will be false
        if not(any(notReady)):
            break

        sleep(10)
    ###########################################################################
    """

    """
    for n in range(nchunks):
        p[n].close()
        p[n].join()
    """


    """
    for n in range(nchunks):
        p[n].close()
        p[n].join()

    """
    
    """
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
    """
    from os import environ
    environ["OMP_NUM_THREADS"]="1"
    environ["OPENBLAS_NUM_THREADS"]="1"
    environ["MKL_NUM_THREADS"]="1"
    environ["VECLIB_MAXIMUM_THREADS"]="1"
    environ["NUMEXPR_NUM_THREADS"]="1"
    """
    import scipy.io as sio
    from imageStats import imageStats
    #import sys
    import gc
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
    del dataload
    gc.collect()
    del gc.garbage[:]

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
            {'ROI_N':ROI_N,'HOUSE':HOUSE,'IMAGE1':IMAGE1,'BG':BG,'dat':dat})

    del dat, ROI_N, BG, imageStats, HOUSE, IMAGE1, sio

    
    # Garbage collection:
    gc.collect()
    del gc.garbage[:]

    #print("{0}{1}".format('Saved ',filename1))
    #sys.stdout.flush()
    
    return position

def good(result):
    #globalLock.lock.acquire()
    #print("good" + str(result))
    #globalLock.lock.release()
    #ok[result]=1
    return
def bad(result):
    #globalLock.lock.acquire()
    #print("bad" + str(result))
    #globalLock.lock.release()
    return

class NullContextManager(object):
    def __init__(self, dummy_resource=None):
        self.dummy_resource = dummy_resource
    def __enter__(self):
        return self.dummy_resource
    def __exit__(self, *args):
        pass


if __name__ == "__main__":
    """
        Call the mult_job function using commandline arguemnts
    """
    import sys
    path1=sys.argv[1]
    filename1=sys.argv[2]

    print(path1 + ' ' + filename1)
    dummy1=NullContextManager()
    find_particle_edges = eval(sys.argv[3])
    position = int(sys.argv[4])
    lock = NullContextManager()
    mult_job(path1, filename1, find_particle_edges,position,lock)

