#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:05:02 2018

@author: mccikpc2
"""
import scipy.io as sio
from exportImages import exportImages

def exportImagesDriver(path1,filename1,foc_crit,min_len):
    
    print('====================exporting images =============================')
    dataload=sio.loadmat('cmap',variable_names=['MAP2'])
    MAP2=dataload['MAP2']
        
    # Exporting all images ++++++++++++++++++++++++++++++++++++++++++++++++++++
    print('exporting all images...')
    exportImages(path1,filename1,foc_crit,min_len,MAP2)
    print('done')
    #--------------------------------------------------------------------------
        

    return
    
