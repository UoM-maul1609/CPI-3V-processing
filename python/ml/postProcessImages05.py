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
# import matplotlib.pyplot as plt 

def postProcessing(filename1='/tmp/20180213071742.mat',\
                   background_file='/tmp/full_backgrounds.mat',\
                   foc_crit=12,min_len=50,selective=False):
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
    
    
    
    # extract centroid
    centroid=data['dat']['centroid'][0,0] 
    tot1=len(data['dat']['len'][0,0][:,0])
    # find all data with a focus greater than foc_crit
    boundary=data['dat']['foc'][0,0]['boundaries'][0,:]
        
    newboundary = [None]*len(centroid)
    radius1 = [None]*len(centroid)
    radius2 = [None]*len(centroid)
    diam2 = [None]*len(centroid)
    angle1 = [None]*len(centroid)
    angle2 = [None]*len(centroid)
    for i in range(len(boundary)):
        # move to centroid
        newboundary[i] = np.zeros(np.shape(boundary[i]))
        newboundary[i][:,1]=boundary[i][:,1]-centroid[i][1] # x
        newboundary[i][:,0]=boundary[i][:,0]-centroid[i][0] # y
        # apply coordinate rotation to boundary array
        theta=-data['dat'][0,0]['orientation'][i,0]*180./np.pi
        theta_r=theta*np.pi/180. +np.pi
        xdash = newboundary[i][:,1]*np.cos(theta_r) + newboundary[i][:,0]*np.sin(theta_r)
        ydash = -newboundary[i][:,1]*np.sin(theta_r) + newboundary[i][:,0]*np.cos(theta_r)
        # calculate the angle from the boundary points
        angle1[i]=np.arctan2(ydash,xdash)+np.pi
        #ind=angle1.argsort()
        #angle1=angle1[ind]

        radius1[i] = np.zeros((len(boundary[i])))
        radius1[i]=np.sqrt(xdash*xdash+ydash*ydash)
        #radius1[i]=radius1[i][ind]

        # interpolate new boundary, adjusted to start from 0
        #angle2[i]=np.mgrid[0:360:30j]*np.pi/180.
        angle2[i]=np.mgrid[0:360:15]*np.pi/180.
        radius2[i]=np.interp(angle2[i],angle1[i],radius1[i],period=2.*np.pi)
        diam2[i]=radius2[i][0:12]+radius2[i][12:]
        diam2[i] = diam2[i] / np.max(diam2[i])
        radius2[i] = radius2[i] /np.max(radius2[i])
        
        if(diam2[i][0] < diam2[i][-1]):
            diam2[i]=np.flip(diam2[i])
        if(radius2[i][0] < radius2[i][-1]):
            radius2[i]=np.flip(radius2[i])
        
        newboundary[i][:,1]=xdash
        newboundary[i][:,0]=ydash
        
    
        
    
    notOnBoundary=[None]*len(boundary)
    for i in range(len(boundary)):
        # extract boundary for this particle
        boundpart= boundary[i]
        
        # getting the size of this image
        (xsize,ysize)=np.shape(data['ROI_N']['IMAGE'][0,0][0,i]['IM'][0,0])
        
        # extract xs and ys
        xs=boundpart[:,1]
        ys=boundpart[:,0]

        # check if any of boundary is on the image edge
        if any(xs==0) or any(ys==0) or any(xs==(xsize-1)) or any (ys==(ysize-1)):
            notOnBoundary[i]=False
        else:
            notOnBoundary[i]=True
            
            
        # check whether particle goes to edge based on darkness
        if notOnBoundary[i] == True:
            image1=data['ROI_N']['IMAGE'][0,0][0,i]['IM'][0,0]
            if any(image1[0,:]<60) or any(image1[-1,:]<60) or any(image1[:,0]<60) or any(image1[:,-1]<60):
                notOnBoundary[i]=False
                
            
        # additionally, if centroid is not within 25% of image centre remove 
        if notOnBoundary[i]==True:
            if (np.abs((centroid[i,1]-ysize*0.5) / ysize) > 0.25) or (np.abs((centroid[i,0]-xsize*0.5) / xsize) > 0.25):
                notOnBoundary[i]=False
                
            

    if selective==False:
        ind1=[(data['dat'][0,0]['foc'][0,i]['focus'][0][0] > foc_crit) \
              and (data['dat']['len'][0,0][i,0] >min_len)  \
              for i in range(len(data['dat']['len'][0,0][:,0]))]
    else:
        ind1=[i for i in range(len(data['dat']['len'][0,0][:,0]))]
    
    for i in range(len(ind1)):
        ind1[i] = ind1[i] and notOnBoundary[i]

    ind,=np.where(ind1)
    indsPP=ind.astype(int)
#    l1=len(ind)
    
    #postP={'images': np.zeros((l1,130,130),dtype='uint8'),
    #     'times': np.zeros((l1,1),dtype='float')}
    
    imagePP=[]
    lensPP=[]
    timesPP=[]
    diamPP=[]
    radiusPP=[]
    roundPP=[]
    l2wPP=[]
    for j in range(len(ind)):
        i=ind[j].astype(int)
        
        
        # subtract background and change type to uint8
        image=data['ROI_N']['IMAGE'][0,0][0,i]['IM'][0,0].astype('int32')- \
            data['BG'][0,i][0,0][0].astype('int32')
        image=image-np.min(np.min(image))
        image=image.astype('uint8')
        
        len1=data['dat']['len'][0,0][i,0]
        bound=data['dat'][0,0]['foc'][0]['boundaries'][i]
        centroid=data['dat']['centroid'][0,0][i,:]
        
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
        (imcol,imrow)=image.shape
        maxdim=np.max((imrow,imcol)) # i.e. "a"
        
        
        
        
        
        """
            0. Make image centre++++++++++++++++++++++++++++++++++++++++++++++
            apply a warp / translation so that particle is at image centre
        """
        x_trans=((imcol)/2-(centroid[0]+1))
        y_trans=((imrow)/2-(centroid[1]+1))
        image_o=image
        M = np.float32([[1,0,y_trans], \
                        [0,1,x_trans]])
        imax,jmax=np.ceil(np.abs(y_trans)).astype(int), np.ceil(np.abs(x_trans)).astype(int)
        #image = cv2.warpAffine(image,M,(imrow,imcol)) [jmax:imcol-(jmax),imax:imrow-(imax)]   
        
        
        # find the maximum dimension - again
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
        
        # scale so pad is same
        rows,cols = res2.shape
        pad=(2.3*rows-len1) / 2
        
        # let the pad be 50% of len
        row_new=(len1*0.5*2+len1)/2.3
        scale=row_new/rows
        
        topBottom=((row_new-rows)/2).astype(int)
        leftRight=topBottom
        # print(str(scale) + ' ' + str(topBottom))

        if topBottom >= 1:
            res2 = cv2.copyMakeBorder(
                             res2, 
                             topBottom, # left
                             topBottom, # right
                             leftRight, # top
                             leftRight, # bottom
                             cv2.BORDER_REPLICATE
                      )
            rows,cols = res2.shape
            # M = cv2.getRotationMatrix2D(\
            #     (rows/2.,cols/2.),0.,scale)
            # res2=cv2.warpAffine(res2,M,(cols,rows))
        else:
            rem=-topBottom+1
            rows,cols = res2.shape
            res2=res2[rem:rows-rem,rem:cols-rem]
            
        # multiply by 255 / (75th percentile)
        res2=np.minimum(res2.astype('float32') * 255./np.percentile(res2.astype('float32'),75),255.)        
        res2=255-res2.astype('uint8')
        
        
        """
            5. resize to common size
        """
        #res1=image
        # resize to constant size
        res=cv2.resize(res2, dsize=(128,128), interpolation=cv2.INTER_CUBIC)
        """
        -------------------------------------------------------------------------------
        """

        # plt.imshow(res)
        # plt.plot(bound[:,1]+y_trans-imax,bound[:,0]+x_trans-jmax,color='c',linewidth=5)
        # plt.show()
        # plt.pause(1.)
        
        """
            save the data
        """
        imagePP.append(res)
        timesPP.append(data['dat'][0,0]['Time'][0,i])
        lensPP.append(data['dat'][0,0]['len'][i,0])
        diamPP.append(diam2[i])
        radiusPP.append(radius2[i])
        roundPP.append(data['dat'][0,0]['round'][i,0])
        l2wPP.append(data['dat'][0,0]['len'][i,0] /data['dat'][0,0]['wid'][i,0] )
        
    return (imagePP,lensPP,timesPP,diamPP,roundPP,l2wPP,radiusPP,indsPP,tot1)

if __name__ == "__main__":
    postProcessing()
    
