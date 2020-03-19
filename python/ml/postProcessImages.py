#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 18 08:09:34 2020

@author: mccikpc2
"""
import matplotlib.pyplot as plt
import scipy.io as sio
import numpy as np
import cv2

foc_crit=12
filename1='/tmp/20180213071742.mat'

databg=sio.loadmat('/tmp/full_backgrounds.mat', \
    variable_names=['FULL_BG'])


data=sio.loadmat(filename1, \
    variable_names=['ROI_N','HOUSE','IMAGE1','BG','dat'])

(maxrow,maxcol)=databg['FULL_BG'][0,0]['IMAGE'][0,0][0,0]['BG'].shape
(minrow,mincol)=(20,20)


# find all data with a focus greater than foc_crit
ind,=np.where(data['dat'][0,0]['foc'][0]['focus'] > foc_crit)
#ind=np.linspace(0,len(data['dat'][0,0]['foc'][0])-1,len(data['dat'][0,0]['foc'][0]))
i=591
#i,=np.where(data['dat'][0,0]['len'] == np.max(data['dat'][0,0]['len']))
#i=822
#i=600
i=ind[i].astype(int)
#bound=data['dat'][0,0]['foc'][0]['boundaries'][i]




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


# find the maximum dimension
(imrow,imcol)=image.shape
maxdim=np.max((imrow,imcol)) # i.e. "a"

theta=60.
theta_r=theta*np.pi/180.


"""
    1. Make into a square by adding a border+++++++++++++++++++++++++++++++++++
    2. Add an extra border 
"""
# scale up border to a square 

topBottom=(np.ceil((maxdim-imrow)/2)+ \
    maxdim*np.sin(theta_r)*np.cos(theta_r)).astype(int)
leftRight=(np.ceil((maxdim-imcol)/2)+ \
    maxdim*np.sin(theta_r)*np.cos(theta_r)).astype(int)

outputImage = cv2.copyMakeBorder(
                 image, 
                 topBottom, # left
                 topBottom, # right
                 leftRight, # top
                 leftRight, # bottom
                 cv2.BORDER_REPLICATE
              )
image1=outputImage
"""
-------------------------------------------------------------------------------
"""



"""
    3. Rotate by angle theta+++++++++++++++++++++++++++++++++++++++++++++++++++
"""
# rotate by some angle
scale=1./(np.cos(theta_r)+np.sin(theta_r))
rows,cols = image1.shape
M = cv2.getRotationMatrix2D(\
    (imrow/2+topBottom,\
     imcol/2+leftRight),theta,scale)
res1=cv2.warpAffine(image1,M,(cols,rows))
"""
-------------------------------------------------------------------------------
"""

"""
    4. take pixels off all sides
"""
remove=np.ceil(maxdim*np.cos(theta_r)*np.sin(theta_r)*\
    (np.cos(theta_r)+np.cos(theta_r)))
remove=remove.astype(int)
(newrow,newcol)=res1.shape
res2=res1[remove:newrow-remove,remove:newrow-remove]
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


plt.imshow(res)

#plt.plot(bound[:,1],bound[:,0],color='c',linewidth=5)
