#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:05:02 2018

@author: mccikpc2
"""
import scipy.io as sio
from imageStats import imageStats

def imageStatsDriver(path1,filename1,find_particle_edges):
    
    print('====================particle properties===========================')
    for i in range(len(filename1)):
        # load from file
        print('Loading from file...')
        dataload=sio.loadmat(path1 + filename1[i].replace('.roi','.mat'),
                           variable_names=['ROI_N','HOUSE','IMAGE1','BG'])
        ROI_N=dataload['ROI_N']
        HOUSE=dataload['HOUSE']
        IMAGE1=dataload['IMAGE1']
        BG=dataload['BG']
        print('done')
    
        #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        
        
        
        # Particle properties +++++++++++++++++++++++++++++++++++++++++++++++++
        print('calculating particle properties...')
        dat=imageStats(ROI_N,BG,find_particle_edges)
        print('done')
        #----------------------------------------------------------------------
        
        
        # save to file
        print('Saving to file...')
        #with open(path1 + filename1[i].replace('.roi','.mat'),'ab') as f:
        #    sio.savemat(f, {'dat':dat})
        sio.savemat(path1 + filename1[i].replace('.roi','.mat'),
         {'ROI_N':ROI_N, 'HOUSE':HOUSE,'IMAGE1':IMAGE1,'BG':BG,'dat':dat})
        print('done')
    
    
    return