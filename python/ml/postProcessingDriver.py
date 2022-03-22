#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 21 09:28:54 2020

@author: mccikpc2
"""
import scipy.io as sio
import postProcessImages01
import postProcessImages02
import postProcessImages03
import postProcessImages04
import postProcessImages05
from os import listdir
from os import path
from tqdm import tqdm
import hdf5storage
import h5py
import pickle
import numpy as np
import json

def postProcessingDriver(path1,outputfile,foc_crit,min_len, type1):
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
    diamPP=[]
    roundPP=[]
    l2wPP=[]
    radiusPP=[]
    for p in path1:
        print(p)
        filename1 = [f for f in listdir(p) if f.endswith(".roi")]
        filename1.sort()
    
        """
            2. extract all images from each file & append
        """
        for i in tqdm(range(len(filename1))):
            f=filename1[i]
            if type1==1:
                (imagePP1,lensPP1,timesPP1)=postProcessImages01.postProcessing(\
                                    path.join(p,f.replace('.roi','.mat')), \
                                    path.join(p,'full_backgrounds.mat'),\
                                    foc_crit,min_len)
            elif type1==2:
                (imagePP1,lensPP1,timesPP1)=postProcessImages02.postProcessing(\
                                    path.join(p,f.replace('.roi','.mat')), \
                                    path.join(p,'full_backgrounds.mat'),\
                                    foc_crit,min_len)
            elif type1==3:
                (imagePP1,lensPP1,timesPP1)=postProcessImages03.postProcessing(\
                                    path.join(p,f.replace('.roi','.mat')), \
                                    path.join(p,'full_backgrounds.mat'),\
                                    foc_crit,min_len)
            elif type1==4:
                (imagePP1,lensPP1,timesPP1)=postProcessImages04.postProcessing(\
                                    path.join(p,f.replace('.roi','.mat')), \
                                    path.join(p,'full_backgrounds.mat'),\
                                    foc_crit,min_len)
            elif type1==5:
                (imagePP1,lensPP1,timesPP1,diamPP1,roundPP1,l2wPP1,radiusPP1,indsPP1,tot1)=postProcessImages05.postProcessing(\
                                    path.join(p,f.replace('.roi','.mat')), \
                                    path.join(p,'full_backgrounds.mat'),\
                                    foc_crit,min_len)
            
            """
                & append
            """
            imagePP.extend(imagePP1)
            lensPP.extend(lensPP1)
            timesPP.extend(timesPP1)
            
            if type1==5:
                diamPP.extend(diamPP1)
                roundPP.extend(roundPP1)
                l2wPP.extend(l2wPP1)
                radiusPP.extend(radiusPP1)
            
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
#    mydict={'imagePP':imagePP,'lensPP':lensPP,'timesPP':timesPP}
#    with open(outputfile,'wb') as out:
#        pickle.dump(mydict, out, protocol=pickle.HIGHEST_PROTOCOL)
        
    # https://stackoverflow.com/questions/20928136/input-and-output-numpy-arrays-to-h5py
    i=np.stack(imagePP,axis=0)
    l=np.stack( lensPP,axis=0)
    t=np.stack(timesPP,axis=0)
    h5f = h5py.File(outputfile, 'w')
    h5f.create_dataset('images', data=i)
    h5f.create_dataset('lens', data=l)
    h5f.create_dataset('times', data=t)
    if type1==5:
        d=np.stack(diamPP,axis=0)
        r=np.stack( roundPP,axis=0)
        l2=np.stack(l2wPP,axis=0)
        ri=np.stack(radiusPP,axis=0)
        h5f.create_dataset('diams', data=d)
        h5f.create_dataset('rounds', data=r)
        h5f.create_dataset('l2ws', data=l2)
        h5f.create_dataset('radii', data=ri)
        
    h5f.close()
    
    
