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
    timesPP=[]
    for p in path1:
        
        filename1 = [f for f in listdir(p) if f.endswith(".roi")]
        filename1.sort()
    
        """
            2. extract all images from each file & append
        """
        for f in filename1:
            
            (imagePP1,timesPP1)=postProcessing(f.replace('.roi','.mat'), \
                                path.join(p,'full_backgrounds.mat'),foc_crit)
            
            """
                & append
            """
            imagePP.append(imagePP1)
            timesPP.append(timesPP1)
            
    """
        -----------------------------------------------------------------------
    """
    
    
    """
        3. save output file
    """
    sio.savemat(outputfile,{'imagePP':imagePP,'timesPP':timesPP})