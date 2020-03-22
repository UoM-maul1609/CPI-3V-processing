#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 09:28:54 2020

@author: mccikpc2
"""
import scipy.io as sio
from postProcessImages import postProcessing
from os import listdir
from os import path
from tqdm import tqdm
import hdf5storage
import h5py
import pickle
import numpy as np
import json

def postProcessingDriver(path1,outputfile,foc_crit,min_len):
    """
        1. loop over all paths
        2. extract all images from each file & append
        3. save to output file
    """
    
    """
        1. loop over all paths ++++++++++++++++++++++++++++++++++++++++++++++++
    """
    imagePP=[]
    lensPP=[]
    timesPP=[]
    for p in path1:
        print(p)
        filename1 = [f for f in listdir(p) if f.endswith(".roi")]
        filename1.sort()
    
        """
            2. extract all images from each file & append
        """
        for i in tqdm(range(len(filename1))):
            f=filename1[i]
            (imagePP1,lensPP1,timesPP1)=postProcessing(\
                                path.join(p,f.replace('.roi','.mat')), \
                                path.join(p,'full_backgrounds.mat'),\
                                foc_crit,min_len)
            
            """
                & append
            """
            imagePP.extend(imagePP1)
            lensPP.extend(lensPP1)
            timesPP.extend(timesPP1)
            
    """
        -----------------------------------------------------------------------
    """
    
    
    """
        3. save output file
    """
    #h5py.get_config().default_file_mode='w'
#    hdf5storage.savemat(outputfile,\
#                {'imagePP':imagePP,'lensPP':lensPP,'timesPP':timesPP})    
#    l=np.stack(imagePP,axis=0)
    mydict={'imagePP':imagePP,'lensPP':lensPP,'timesPP':timesPP}
    with open(outputfile,'wb') as out:
        json.dump(mydict, out)