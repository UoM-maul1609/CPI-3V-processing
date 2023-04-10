#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 12 10:05:02 2018

@author: mccikpc2
"""

"""
from os import environ
environ["OMP_NUM_THREADS"]="1"
environ["OPENBLAS_NUM_THREADS"]="1"
environ["MKL_NUM_THREADS"]="1"
environ["VECLIB_MAXIMUM_THREADS"]="1"
environ["NUMEXPR_NUM_THREADS"]="1"

from cv2 import setNumThreads
from cv2 import threshold
from cv2 import THRESH_BINARY
from skimage import measure
from matplotlib import path
from time import sleep
import gc
from tqdm import tqdm

import numpy as np
from scipy.interpolate import RectBivariateSpline
from scipy import signal
"""
NB=30
ld=3
from numpy import zeros
boundaries2=zeros((NB,2),dtype='float')
xs=zeros((ld,NB-1),dtype='float')
ys=zeros((ld,NB-1),dtype='float')
focus=zeros((NB-1,1),dtype='float')
intensity=zeros(ld,dtype='float')

def imageStats(ROI_N,BG,cpiv1,b_flag,position,desc,globalLock):

    import sys
    """
    from os import environ
    environ["OMP_NUM_THREADS"]="1"
    environ["OPENBLAS_NUM_THREADS"]="1"
    environ["MKL_NUM_THREADS"]="1"
    environ["VECLIB_MAXIMUM_THREADS"]="1"
    environ["NUMEXPR_NUM_THREADS"]="1"
    """
    from cv2 import setNumThreads
    from cv2 import threshold, adaptiveThreshold, cvtColor, COLOR_BGR2GRAY, floodFill
    from cv2 import THRESH_BINARY, THRESH_BINARY_INV, ADAPTIVE_THRESH_GAUSSIAN_C, \
        ADAPTIVE_THRESH_MEAN_C, THRESH_OTSU
    from skimage import measure
    from matplotlib import path
    from time import sleep
    import gc
    from tqdm import tqdm

    import numpy as np
    from scipy.interpolate import RectBivariateSpline
    from scipy import signal

    #setNumThreads(1)
    
    
    global NB, ld
    pix=2.3
    if cpiv1:
        ind,=np.where(ROI_N['imageType'][0,0][:,0] == 33857)
    else:
        ind,=np.where(ROI_N['imageType'][0,0][:,0] == 89)
    l1=len(ROI_N['IMAGE'][0,0][0,:])
    
    # define data structured array:
    dat={'len': np.zeros((l1,1),dtype='float'),
         'wid': np.zeros((l1,1),dtype='float'),
         'area': np.zeros((l1,1),dtype='float'),
         'round': np.zeros((l1,1),dtype='float'),
         'orientation': np.zeros((l1,1),dtype='float'),
         'centroid': np.zeros((l1,2),dtype='float')}
    if not(b_flag):
        foc1=np.zeros(l1, dtype=[('focus', 'float')])
        foc1.fill(np.nan)
    else:
        #ld=3
        foc1=np.zeros(l1, dtype=[('focus', 'float'),
                                 ('boundaries','(30,2)float') ])
        #foc1=np.zeros(l1, dtype=[('focus', 'float'),
        #                         ('xs','('+str(ld)+',29)float'),
        #                         ('ys','('+str(ld)+',29)float'),
        #                         ('boundaries','(30,2)float') ])
        foc1.fill(np.nan)
    
    dat['foc']=foc1
        
     
    
    dat['Time'] = ROI_N['Time'][0,0][:,0]
    
    if len(ind)==0:
        return dat
    
         
    #https://stackoverflow.com/questions/45742888/tqdm-using-multiple-bars
    
    with tqdm(total=len(ind), position=position,leave=True) as pbar:
        with globalLock:
            pbar.monitor_interval = 0
            pbar.set_description("%s %d" % (desc,position))
            sys.stdout.flush()
            sys.stderr.flush()
            sleep(0.1)
    
        j=0
        for i in ind:
            j += 1    
            
            if np.mod(j,50) == 0:
                with globalLock:
                    pbar.n=j-1
                    pbar.update()
                    sys.stdout.flush()
                    sys.stderr.flush()
                    sleep(0.1)

            elif j == len(ind):
                with globalLock:
                    pbar.n=j-1
                    pbar.update()
                    sys.stdout.flush()
                    sys.stderr.flush()
                    sleep(0.1)
            
               
            #pbar.update(1)
            arr=ROI_N['IMAGE'][0,0][0,i]['IM'][0,0].astype(int)-BG[0,i]['BG'][0,0].astype(int)
         
            inda_r,inda_c=np.where(arr<=-15)
            indb_r,indb_c=np.where(arr>-15)
            
            if len(inda_r)<=20:
                continue
                
            
            arr=(arr-np.min(arr))

            
            if cpiv1:
               # this bit of code removes the centre
               h,w=np.shape(arr)
               start1=ROI_N['StartX'][0,0][0,i]
               end1=start1+h
               # fill in the first line
               if((end1 >= 512) and (start1 < 511) ):
                  arr[512-start1-1:512-start1,:]=arr[512-start1-2:512-start1-1,:]

               if((end1 >= 514) and (start1<=512) ):
                  arr[512-start1:512-start1+1,:]=arr[512-start1+1:512-start1+2,:]

            
            for f in range(100):
                arr[1:-1,1:-1]=signal.medfilt2d(arr.astype('B'),(3,3))[1:-1,1:-1]

            (r,c)=np.shape(arr)
            if r <= 15 and c <=15:
                continue
            
            # 10 pixels need to be darker by 15 units or more for a particle
            #indt,=np.where((arr.flatten()-0.5*np.max(arr).astype('H'))>30)
            indt,=np.where((arr.flatten()-np.percentile(arr,90).astype('H'))<-30)
            if len(indt)<=10:
                continue
    
            #https://docs.opencv.org/2.4/doc/tutorials/imgproc/threshold/threshold.html        
            #https://docs.opencv.org/2.4/modules/imgproc/doc/miscellaneous_transformations.html?highlight=threshold#threshold
            """dx=cv2.Sobel(arr.astype(float),cv2.CV_64F,1,0,ksize=1)
            dy=cv2.Sobel(arr.astype(float),cv2.CV_64F,0,1,ksize=1)
            mag=np.sqrt(dx**2+dy**2)
            (th,level)=cv2.threshold(mag.astype('H'),50,255,cv2.THRESH_BINARY)
            """
            # could try np.float32(max(arr(:)).astype('H'))*0.4
            #(th,level)=threshold(np.float32(arr.astype('H')),38,255,THRESH_BINARY)
            
            # think I am going to try 80th percentile *0.7. wihich is probably indicative of 
            # background image value
#             (th,level)=threshold(np.float32(arr.astype('H')), \
#                np.max([np.float32(np.max(arr[:]).astype('H'))*0.5,38]),255,THRESH_BINARY)
#            (th,level)=threshold(np.float32(arr.astype('H')), \
#                np.percentile(arr,18).astype('H'),255,THRESH_BINARY)
            (th,level)=threshold(arr.astype('uint8'), \
                0,255,THRESH_BINARY+THRESH_OTSU)
            """
            (th,level)=threshold(np.float32(arr.astype('H')), \
               (0.5*np.percentile(arr,95)).astype('H'),255,THRESH_BINARY)
            im_floodfill=level.copy()
            h,w=level.shape[:2]
            mask=np.zeros((h+2,w+2),np.uint8)
            floodFill(im_floodfill, mask, (42,51), 255)
            im_floodfill_inv = cv2.bitwise_not(im_floodfill.astype('B'))
            BW2=im_floodfill_inv
            contours=measure.find_contours((BW2-np.min(BW2))/255,0.5)
            
            """

            #print(np.shape(np.stack([arr,arr,arr],axis=2)))           
            #(level)=adaptiveThreshold(arr.astype('uint8'), \
            #     255,ADAPTIVE_THRESH_MEAN_C,THRESH_BINARY,71,50)
    
            BW2=level.astype('B')
            BW2[1:-1,1:-1]=signal.medfilt2d(level.astype('B'),(3,3))[1:-1,1:-1]
            for f in range(10):
                BW2[1:-1,1:-1]=signal.medfilt2d(BW2.astype('B'),(3,3))[1:-1,1:-1]
    
            # fairly insensitive to the 0.4 choice
            #continue
            contours=measure.find_contours((BW2-np.min(BW2))/255,0.4) # this prob doesn't make much difference - as it's a BW image
            
            
            #temp=(BW2-np.min(BW2))
            #contours, hierarchy = cv2.findContours(temp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE) 
            if not(len(contours)):
                continue
            
            # put longest list first
            if len(contours) > 1:
                contours.sort(key=len,reverse=True)
            contour=contours[0]
            
    
            # https://stackoverflow.com/questions/3654289/scipy-create-2d-polygon-mask
            x, y = np.meshgrid(np.arange(r), np.arange(c))
            x, y = x.flatten(), y.flatten()
            points = np.vstack((x,y)).T
            
            if len(contour) <= 5:
                continue

            p=path.Path(contour)
            if len(contour) > 10000:
                continue
            """
            print(' ')
            print(np.shape(contour))
            plt.imshow(arr)
            plt.plot(contour[:,1],contour[:,0])
            plt.savefig('/tmp/test.png')
            """
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
                #sys.stdout.flush()
                continue
            
            
            dat['len'][i]=stats[0].major_axis_length*pix
            dat['area'][i]=stats[0].filled_area*pix*pix
            if(dat['area'][i] <= pix*pix):
                #print('Problem with this particle in regionprops')
                #sys.stdout.flush()
                continue
            dat['wid'][i]=stats[0].minor_axis_length*pix
    
            
            dat['round'][i]=stats[0].filled_area/(np.pi/4.*stats[0].major_axis_length**2)
            dat['centroid'][i,:]=stats[0].centroid
            dat['orientation'][i]=stats[0].orientation
                
            
            #dat['foc'][i]['focus'][0]=np.nan
            
            boundaries = contour
            foc=calculate_focus(boundaries,arr,b_flag) # maybe focus less than 12?
            dat['foc'][i]=foc
            
            del boundaries, contour, contours, IN, BW2, level, mask, stats
            gc.collect()
            del gc.garbage[:]
        
        sleep(0.01)
        
        with globalLock:
            pbar.n=len(ind)-1
            pbar.update()
            sys.stdout.flush()
            sys.stderr.flush()
        
    return dat




def calculate_focus(boundaries,IM,b_flag):
    """
    from os import environ
    environ["OMP_NUM_THREADS"]="1"
    environ["OPENBLAS_NUM_THREADS"]="1"
    environ["MKL_NUM_THREADS"]="1"
    environ["VECLIB_MAXIMUM_THREADS"]="1"
    environ["NUMEXPR_NUM_THREADS"]="1"
    """
    from time import sleep
    import gc

    import numpy as np
    from scipy.interpolate import RectBivariateSpline
    from scipy import signal
    

    global boundaries2, xs, ys, focus, intensity, NB, ld
    
    if not(b_flag):
        foc=np.zeros(1, dtype=[('focus', 'float')])
    else:
        foc=np.zeros(1, dtype=[('focus', 'float'),
                                 ('boundaries','(30,2)float') ])
        #foc=np.zeros(1, dtype=[('focus', 'float'),
        #                         ('xs','('+str(ld)+',29)float'),
        #                         ('ys','('+str(ld)+',29)float'),
        #                         ('boundaries','(30,2)float') ])

    foc['focus'][0]=np.nan
    

    ind1=np.linspace(1,NB,len(boundaries[:,0]))
    ind2=np.linspace(1,ind1[-1],NB);
    if not(len(ind2)):
        return foc
        
    
    
    
        
    #boundaries2=np.zeros((len(NB),2)) 
    #xs=np.zeros((ld,NB-1))
    #ys=np.zeros((ld,NB-1))
    #focus=np.zeros((NB-1,1))
    #intensity=np.zeros(ld)
    
    #boundaries2=np.zeros((NB,2),dtype='float')
    #xs=np.zeros((ld,NB-1),dtype='float16')
    #ys=np.zeros((ld,NB-1),dtype='float16')
    #focus=np.zeros((NB-1,1),dtype='float16')
    #intensity=np.zeros(ld,dtype='float16')
    
    
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
    
    
    for i in range(NB-1):
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
        for i in range(0,NB-1):
            for j in range(ld):
                intensity[j]=f(xs[j,i],ys[j,i]) 
            focus[i]=np.abs(np.gradient(intensity,edge_order=2)[1])
            foc['focus'][0]=np.nanmean(focus)/2.
    except:
        foc['focus'][0]=np.nan
    finally:
        pass
        #del xs, ys, RectBivariateSpline


    if b_flag:
       #foc['xs'][0]=xs
       #foc['ys'][0]=ys
       foc['boundaries'][0]=boundaries2
    
    
    return foc

