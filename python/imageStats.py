#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:05:02 2018

@author: mccikpc2
"""
import numpy as np
import cv2
import scipy.signal as signal
from skimage import measure
from matplotlib import path
#from scipy.interpolate import interp2d
from scipy.interpolate import RectBivariateSpline
from tqdm import tqdm
import sys

def imageStats(ROI_N,BG,b_flag,position,desc):
    
    pix=2.3
    ind,=np.where(ROI_N['imageType'][0,0][:,0] == 89)
    l1=len(ROI_N['IMAGE'][0,0][0,:])
    
    # define data structured array:
    dat={'len': np.zeros((l1,1)),
         'wid': np.zeros((l1,1)),
         'area': np.zeros((l1,1)),
         'round': np.zeros((l1,1)),
         'centroid': np.zeros((l1,2))}
    if not(b_flag):
        foc1=np.zeros(l1, dtype=[('focus', 'float')])
        foc1.fill(np.nan)
    else:
        #ld=3
        foc1=np.zeros(l1, dtype=[('focus', 'float'),
                                 ('boundaries','(30,2)float32') ])
        #foc1=np.zeros(l1, dtype=[('focus', 'float'),
        #                         ('xs','('+str(ld)+',29)float'),
        #                         ('ys','('+str(ld)+',29)float'),
        #                         ('boundaries','(30,2)float') ])
        foc1.fill(np.nan)
    
    dat['foc']=foc1
        
        
    
    dat['Time'] = ROI_N['Time'][0,0][:,0]
    
    tqdm.monitor_interval = 0
    #https://stackoverflow.com/questions/45742888/tqdm-using-multiple-bars
    pbar=tqdm(total=len(ind), position=position)
    pbar.set_description("%s %d" % (desc,position))
    for i in ind:
        pbar.update(1)
        arr=ROI_N['IMAGE'][0,0][0,i]['IM'][0,0].astype(int)-BG[0,i]['BG'][0,0].astype(int)
     
        inda_r,inda_c=np.where(arr<=-15)
        indb_r,indb_c=np.where(arr>-15)
        
        if len(inda_r)<=7:
            continue
        
        arr=(arr-np.min(arr))
        #mx=np.max(arr)
        #arr=arr*255/mx
        #arr=arr/255
    
        (r,c)=np.shape(arr)

        #https://docs.opencv.org/2.4/doc/tutorials/imgproc/threshold/threshold.html        
        #https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=threshold#threshold
        """dx=cv2.Sobel(arr.astype(float),cv2.CV_64F,1,0,ksize=1)
        dy=cv2.Sobel(arr.astype(float),cv2.CV_64F,0,1,ksize=1)
        mag=np.sqrt(dx**2+dy**2)
        (th,level)=cv2.threshold(mag.astype('H'),50,255,cv2.THRESH_BINARY)
        """
        (th,level)=cv2.threshold(arr.astype('H'),38,255,cv2.THRESH_BINARY)
        

        BW2=level.astype('B')
        BW2[1:-1,1:-1]=signal.medfilt2d(level.astype('B'),(3,3))[1:-1,1:-1]
       

        # fairly insensitive to the 0.4 choice
        contours=measure.find_contours((BW2-np.min(BW2))/255,0.4)
        if not(len(contours)):
            continue
        
        # put longest list first
        contours.sort(key=len,reverse=True)
        contour=contours[0]
        

        # https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask
        x, y = np.meshgrid(np.arange(r), np.arange(c))
        x, y = x.flatten(), y.flatten()
        points = np.vstack((x,y)).T
        
        
        p=path.Path(contour)
        mask=p.contains_points(points, radius=2)
        IN = mask.reshape((c,r)).T
        
        #http://scikit-image.org/docs/dev/auto_examples/segmentation/plot_regionprops.html
        stats=measure.regionprops(IN.astype('B'))
        if not(len(stats)):
            continue
        if stats[0].major_axis_length==0:
            continue
        
        if stats[0].eccentricity > 0.9999:
            #print("Eccentricity too high")
            sys.stdout.flush()
            continue
        
        
        dat['len'][i]=stats[0].major_axis_length*pix
        dat['area'][i]=stats[0].filled_area*pix*pix
        if(dat['area'][i] <= pix*pix):
            #print('Problem with this particle in regionprops')
            sys.stdout.flush()
            continue
        dat['wid'][i]=stats[0].minor_axis_length*pix

        
        dat['round'][i]=stats[0].filled_area/(np.pi/4.*stats[0].major_axis_length**2)
        dat['centroid'][i,:]=stats[0].centroid
            
        boundaries=contour
        
        
        foc=calculate_focus(boundaries,arr,b_flag) # maybe focus less than 12?
    
        dat['foc'][i]=foc
        
    
    return dat




def calculate_focus(boundaries,IM,b_flag):
    ld=3
    
    
    if not(b_flag):
        foc=np.zeros(1, dtype=[('focus', 'float')])
    else:
        foc=np.zeros(1, dtype=[('focus', 'float'),
                                 ('boundaries','(30,2)float32') ])
        #foc=np.zeros(1, dtype=[('focus', 'float'),
        #                         ('xs','('+str(ld)+',29)float'),
        #                         ('ys','('+str(ld)+',29)float'),
        #                         ('boundaries','(30,2)float') ])

    foc['focus'][0]=np.nan

    ind1=np.linspace(1,30,len(boundaries[:,0]))
    ind2=np.linspace(1,ind1[-1],30);
    if not(len(ind2)):
        return foc
        
    
    
    
    #if not boundaries2:
        
    boundaries2=np.zeros((len(ind2),2)) 
    (r,c)=np.shape(boundaries2)
    
    xs=np.zeros((ld,r-1))
    ys=np.zeros((ld,r-1))
    focus=np.zeros((r-1,1))
    intensity=np.zeros(ld)
    
    
    
    if len(boundaries[:,0])>9:
        boundaries2[:,0]=np.interp(ind2,ind1,signal.filtfilt([1,1,1],3,boundaries[:,0]))
        boundaries2[:,1]=np.interp(ind2,ind1,signal.filtfilt([1,1,1],3,boundaries[:,1]))
    else:
        boundaries2[:,0]=np.interp(ind2,ind1,boundaries[:,0])
        boundaries2[:,1]=np.interp(ind2,ind1,boundaries[:,1])

    midpointx=(boundaries2[1:len(boundaries2),0]+boundaries2[0:-1,0])/2.
    midpointy=(boundaries2[1:len(boundaries2),1]+boundaries2[0:-1,1])/2.
    
    
    gradient1=(boundaries2[1:len(boundaries2),1]-boundaries2[0:-1,1]) \
        / (1e-10+boundaries2[1:len(boundaries2),0]-boundaries2[0:-1,0])
    
    gradient2=-1./(gradient1+1e-10)
    
    # find start and end point of the lines to calculate the gradients along
    dl=-2. # length 2 pixels
    dx1=dl/np.sqrt(1.+gradient2**2)
    dy1=dl/np.sqrt(1.+1./gradient2**2)
    
    dl=2 # length 2 pixels
    dx2=dl/np.sqrt(1.+gradient2**2)
    dy2=dl/np.sqrt(1.+1./gradient2**2)
    
    deltax=(boundaries2[1:len(boundaries2),0]-boundaries2[0:-1,0])
    deltay=(boundaries2[1:len(boundaries2),1]-boundaries2[0:-1,1])
    
    
    for i in range(r-1):
        xs[:,i]=np.linspace(midpointx[i]+dx1[i],midpointx[i]+dx2[i],ld)
        ys[:,i]=np.linspace(midpointy[i]-dy1[i],midpointy[i]-dy2[i],ld)
        
    
    # correct for direction of gradient
    ind,=np.where((deltay<0) & (deltax>0))
    for i in ind:
        ys[:,i]=np.linspace(midpointy[i]+dy1[i],midpointy[i]+dy2[i],ld)
    
    ind,=np.where((deltay>0) & (deltax<0))
    for i in ind:  
        ys[:,i]=np.linspace(midpointy[i]+dy1[i],midpointy[i]+dy2[i],ld)
    
    
    # now calculate the focus
    (rr,cc)=np.shape(IM)
#    rr,cc=np.meshgrid(np.arange(0,rr),np.arange(0,cc))
    rr,cc=np.arange(0,rr),np.arange(0,cc)
    #rr, cc = np.mgrid[0:rr:1, 0:cc:1]
    try:
        #intensity=griddata(xs[:,i],ys[:,i],(rr,cc),method='nearest')
        #f=interp2d(rr,cc,IM,bounds_error=False,method='linear')
        f=RectBivariateSpline(rr,cc,IM)
        for i in range(0,r-1):
            for j in range(ld):
                intensity[j]=f(xs[j,i],ys[j,i]) 
            focus[i]=np.abs(np.gradient(intensity,edge_order=2)[1])
            foc['focus'][0]=np.nanmean(focus)/2.
            
            
    except:
        foc['focus'][0]=np.nan
    
        
    if b_flag:
       #foc['xs'][0]=xs
       #foc['ys'][0]=ys
       foc['boundaries'][0]=boundaries2
    
    
    return foc

