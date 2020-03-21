#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 08:09:34 2020

@author: mccikpc2
"""


#import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import cv2

def postProcessing(filename1='/tmp/20180213071742.mat',\
                   background_file='/tmp/full_backgrounds.mat',\
                   foc_crit=12):
    """
        loop through all images full filling criteria
        process them so they are all the same size and rotated to same 
        orientation
    """
    
    
    databg=sio.loadmat(background_file, \
        variable_names=['FULL_BG'])
    
    
    data=sio.loadmat(filename1, \
        variable_names=['ROI_N','HOUSE','IMAGE1','BG','dat'])
    
    (maxrow,maxcol)=databg['FULL_BG'][0,0]['IMAGE'][0,0][0,0]['BG'].shape
#    (minrow,mincol)=(20,20)
    
    
    # find all data with a focus greater than foc_crit
    ind,=np.where(data['dat'][0,0]['foc'][0]['focus'] > foc_crit)
    
#    l1=len(ind)
    
    #postP={'images': np.zeros((l1,130,130),dtype='uint8'),
    #     'times': np.zeros((l1,1),dtype='float')}
    
    imagePP=[]
    timesPP=[]
    for j in range(len(ind)):
        i=ind[j].astype(int)
        
        
        # subtract background and change type to uint8
        image=data['ROI_N']['IMAGE'][0,0][0,i]['IM'][0,0].astype('int32')- \
            data['BG'][0,i][0,0][0].astype('int32')
        image=image-np.min(np.min(image))
        image=image.astype('uint8')
        
        """++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
            Need to do the following:
            1. Make into a square by adding a border
            2. Add an extra border of a sin(theta)cos(theta) before rotation
            3. Rotate by angle theta
            4. take pixels off all sides: 
                acos(theta)sin(theta)x(cos(theta)+sin(theta))
            5. resize to common size
           ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        
        
        
        #theta=0.
        theta=-data['dat'][0,0]['orientation'][i,0]*180./np.pi
        theta_r=theta*np.pi/180.
        
        # find the maximum dimension
        (imrow,imcol)=image.shape
        maxdim=np.max((imrow,imcol)) # i.e. "a"
        
        """
            1. Make into a square by adding a border+++++++++++++++++++++++++++++++++++
            2. Add an extra border 
        """
        # scale up border to a square 
        scale=1.
        
        topBottom=np.ceil((np.abs(np.ceil((maxdim-imrow)/2)+ \
            maxdim*np.abs(np.sin(theta_r))*np.abs(np.cos(theta_r))))).astype(int)
        leftRight=((2*topBottom+imrow-imcol)/2).astype(int)
        
        #leftRight=np.ceil((np.abs(np.ceil((maxdim-imcol)/2)+ \
        #    maxdim*np.sin(theta_r)*np.cos(theta_r)))).astype(int)
        outputImage = cv2.copyMakeBorder(
                         image, 
                         topBottom, # left
                         topBottom, # right
                         leftRight, # top
                         leftRight, # bottom
                         cv2.BORDER_REPLICATE
                      )
        image1=outputImage
        
        maxdimr,maxdimc=image1.shape
        total1=maxdim*(np.abs(np.sin(theta_r))+np.abs(np.cos(theta_r)))
        total1=np.ceil(total1).astype(int)
        
        total2=maxdim*(np.abs(np.sin(theta_r)) + np.abs(np.cos(theta_r)))* \
            (np.abs(np.cos(theta_r)) + np.abs(np.sin(theta_r)))
        total2=np.ceil(total2).astype(int)
        
        total3=total2*(np.abs(np.cos(theta_r))+np.abs(np.sin(theta_r)))
        total3=np.ceil(total3).astype(int)
        
        
        
        
        """
        -------------------------------------------------------------------------------
        """
        
        
        
        """
            3. Rotate by angle theta+++++++++++++++++++++++++++++++++++++++++++++++++++
        """
        # rotate by some angle
        rows,cols = image1.shape
        M = cv2.getRotationMatrix2D(\
            ((imrow/2+topBottom-0.5),\
             (imcol/2+leftRight-0.5)),theta,scale)
        res1=cv2.warpAffine(image1,M,(cols,rows))
        """
        -------------------------------------------------------------------------------
        """
        
        
        
        
        (rows1,cols1)=res1.shape
        
        """
            4. take pixels off all sides
        """
        #theta_r=theta_r+np.pi/2.
        #remove=np.abs(np.ceil((maxdim*np.abs(np.cos(theta_r))*np.abs(np.sin(theta_r)))*\
        #    (np.abs(np.cos(theta_r))+np.abs(np.cos(theta_r))))*scale)
        #remove=(np.ceil((maxdim*np.abs(np.cos(theta_r))*np.abs(np.sin(theta_r)))*\
        #    (np.abs(np.cos(theta_r))+np.abs(np.cos(theta_r)))))* \
        #                total1/maxdimr
        
        remove=(total3-total2)/2*total2/total3
        
        remove=np.ceil(remove).astype(int)+1
        
        
        
        (newrow,newcol)=res1.shape
        res2=res1[remove:newrow-1-remove,remove:newrow-1-remove]
        """
        -------------------------------------------------------------------------------
        """
        
        
        
        """
            5. resize to common size
        """
        #res1=image
        # resize to constant size
        res=cv2.resize(res2, dsize=(130,130), interpolation=cv2.INTER_CUBIC)
        """
        -------------------------------------------------------------------------------
        """
        
        """
            save the data
        """
    #    postP['images'][j]=res
    #    postP['times'][j]=data['dat']['Time'][0,0][0,j]
        imagePP.append(res)
        timesPP.append(data['dat']['Time'][0,0][0,j])
        
    return (imagePP,timesPP)