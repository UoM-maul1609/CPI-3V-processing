import numpy as np
import cv2 as cv
import copy
import math
import time
import  tkinter as tk
from tkinter import filedialog
import sys
import os
import netCDF4 as nc
import matplotlib.pyplot as plt
import matplotlib as mpl
from cycler import cycler


""" 
    get a video file name+++++++++++++++++++++++++++++++++++++++++++++++++++++
"""
root = tk.Tk()
root.withdraw()
filename1 = filedialog.askopenfilename(title='Select a file...', \
    filetypes = (('Video files', '*.h264'),('All files', '*.*')))
print(filename1)
if not len(filename1):
    sys.exit()
filename2 = filename1[0:-5]
"""
-------------------------------------------------------------------------------
"""



"""
    get 100th frame from video and use to find location of drops+++++++++++++++
"""
vidcap = cv.VideoCapture(filename1)
count = 0
success = True
print('First frame read')

print('Read first 100 frames')
while success:
    success,imgOrig = vidcap.read()
    if count == 100:
#         cv2.imwrite(filename2+'_frame1.jpg',imgOrig)
        break
    count+=1
print('...done')
"""
-------------------------------------------------------------------------------
"""




"""
    some initial variables
"""
plotFP = False
drawing = False # true if mouse is pressed
slide1 = False
grow1 = False
rButtonDown = False
ix,iy = -1,-1
dx,dy = -1,-1

circleStore=dict()
circleStore['radii']=[]
circleStore['x']=[]
circleStore['y']=[]
circleStore['delete']=[]
circleStore['selected']=[]

"""
 check if netcdf data exist and if so load
"""
fn = filename2 + '.nc'
if(os.path.exists(fn)):
    ds = nc.Dataset(fn)
    circleStore['radii']=ds['rads'][:].astype(int).tolist()
    circleStore['x']=ds['xs'][:].astype(int).tolist()
    circleStore['y']=ds['ys'][:].astype(int).tolist()
    circleStore['delete'] = [False]*len(circleStore['radii'])
    circleStore['selected'] = [False]*len(circleStore['radii'])
    ds.close()
 

tempCircle=dict()
tempCircle['radii']=-1
tempCircle['x']=-1
tempCircle['y']=-1
tempCircle['delete']=True


"""
    Read the image in and scale it, create a cache of the original scaled image
"""
# imgOrig=cv.imread('../../20230525-163451_frame1.jpg')
width=1024
height=int(width/imgOrig.shape[1]*imgOrig.shape[0])
dim=(width,height)
img = cv.resize(imgOrig,dim) 
im2 = copy.deepcopy(img)
cache = copy.deepcopy(img)




# mouse callback function
def draw_circle(event,x,y,flags,param):
    global ix,iy,dx,dy,drawing,img, im2,cache, circleStore,rButtonDown

    if event == cv.EVENT_LBUTTONDOWN:
        drawing = True
        ix,iy = x,y

    elif event == cv.EVENT_RBUTTONDOWN:
        drawing = False
        rButtonDown = True
        circleStore['selected']=[False for i in range(len(circleStore['selected']))]
        dx,dy = x,y
        # store which circle
        for i in range(len(circleStore['delete'])):
            dist1=np.sqrt((circleStore['x'][i]-x)**2 + \
                (circleStore['y'][i]-y)**2)
            if(dist1<=circleStore['radii'][i]):
                circleStore['selected'][i]=True
                break
                
    elif event == cv.EVENT_RBUTTONUP:
        drawing = False
        rButtonDown = False
        if slide1 or grow1:
            return
        # perform check to see which circles point is inside
        for i in range(len(circleStore['delete'])):
            dist1=np.sqrt((circleStore['x'][i]-dx)**2 + \
                (circleStore['y'][i]-dy)**2)
            if(dist1<=circleStore['radii'][i]):
                circleStore['delete'][i]=True
        
        ind1,=np.where(np.array(circleStore['delete']) == True)
        for ele in sorted(ind1, reverse=True):
            circleStore['x'].pop(ele)
            circleStore['y'].pop(ele)
            circleStore['radii'].pop(ele)
            circleStore['delete'].pop(ele)
        

    elif event == cv.EVENT_LBUTTONUP:
        if slide1 or grow1:
            return
        drawing = False
        radius = int(math.sqrt(((ix - x) ** 2) + ((iy - y) ** 2)))
        circleStore['x'].append(ix)
        circleStore['y'].append(iy)
        circleStore['radii'].append(radius)
        circleStore['delete'].append(False)
        circleStore['selected'].append(False)

        tempCircle['delete']=True

    elif event == cv.EVENT_MOUSEMOVE:
    
        if slide1 and rButtonDown:
            """
                in here we need to check which circle was right clicked on, and 
                move its position based on current position of mouse
            """
            ind,=np.where(np.array(circleStore['selected']) == True)
            if len(ind):
                circleStore['x'][ind[0]]=x
                circleStore['y'][ind[0]]=y
            return
        if grow1 and rButtonDown:
            """
                in here we need to check which circle was right clicked on, and 
                grow its position based on current position of mouse
            """
            ind,=np.where(np.array(circleStore['selected']) == True)
            if len(ind):
                dist1=np.sqrt((circleStore['x'][ind[0]]-x)**2 + \
                    (circleStore['y'][ind[0]]-y)**2)
                circleStore['radii'][ind[0]]=int(dist1)
            return
            
        if (slide1 or grow1) and not rButtonDown:
            return
        if drawing == True:
            radius = int(math.sqrt(((ix - x) ** 2) + ((iy - y) ** 2)))
            
            tempCircle['x']=ix
            tempCircle['y']=iy
            tempCircle['radii']=radius
            tempCircle['delete']=False
                
                

cv.namedWindow('image',cv.WINDOW_GUI_NORMAL)
cv.setMouseCallback('image',draw_circle)
while(1):
    im2=copy.deepcopy(img)
    for i in range(len(circleStore['x'])):
        cv.circle(im2,(circleStore['x'][i],circleStore['y'][i]), \
            circleStore['radii'][i],(0,0,255),thickness=2)
    if(tempCircle['delete']==False):
        cv.circle(im2,(tempCircle['x'],tempCircle['y']), \
            tempCircle['radii'],(0,255,0),thickness=2)

    cv.imshow('image',im2)
    k = cv.waitKey(1) & 0xFF
    if k == ord('s'):
        grow1=False
        slide1 = not slide1
        if slide1:
            print("We are in the mode to move a circle")
        else:
            print("We are in the standard mode")
    elif k == ord('g'):
        slide1=False
        grow1 = not grow1
        if grow1:
            print("We are in the mode to grow a circle")
        else:
            print("We are in the standard mode")
    elif k == 27:
        break
    elif k == 13:
        break
cv.destroyAllWindows()



"""
    now process
"""
if not len(circleStore['radii']):
    sys.exit()


"""
    save the circle definitions to a file
"""
if k == 27:
    sys.exit()
if k == 13:
    fn = filename2 + '.nc'
    ds = nc.Dataset(fn, 'w',format = 'NETCDF4')
    circledims = ds.createDimension('circles',len(circleStore['x']) )
    xs = ds.createVariable('xs','f4',('circles',))
    ys = ds.createVariable('ys','f4',('circles',))
    rads = ds.createVariable('rads','f4',('circles',))
    xs[:] = np.array(circleStore['x'])
    ys[:] = np.array(circleStore['y'])
    rads[:] = np.array(circleStore['radii'])
    ds.close()
    
    
    
"""
    read video as frames and calculate std
"""
img1=cv.cvtColor(imgOrig,cv.COLOR_BGR2GRAY)
square_scale=np.mean(img1[-100:-1,-100:-1])

print('Processing frames...')
d_total = len(circleStore['x'])
results, frame_count = [], []
vidcap = cv.VideoCapture(filename1)
success,frame = vidcap.read()
frame = cv.resize(frame,dim)
height1, width1, depth1 = frame.shape
count=0
border=1
while success:
    print('Read a new frame: ', success, count)
    success,frame1 = vidcap.read()
    if(not success):
        break
    frame = cv.resize(frame1,dim)

    count += 1    

    roi = np.zeros(frame.shape[:2], np.uint8)
    height, width, depth = img.shape
    for i in range(d_total):
        x,y,r = circleStore['x'][i], circleStore['y'][i], circleStore['radii'][i]
        # writes the circle onto the roi mask
        roi = cv.circle(roi, (x, y), r, 255, cv.FILLED)
        mask = np.ones_like(frame) * 255
        # this gets the part of the image masked by roi and puts on mask
        mask = cv.bitwise_and(mask, frame, mask=roi) + cv.bitwise_and(mask, mask, mask=~roi)
        x1 = max(x-r - border//2, 0)      # eventually  -(border//2+1)
        x2 = min(x+r + border//2, width)  # eventually  +(border//2+1)
        y1 = max(y-r - border//2, 0)      # eventually  -(border//2+1)
        y2 = min(y+r + border//2, height) # eventually  +(bord

        # get the part of the image that corresponds to the drop
        image = mask[y1:y2,x1:x2]
        img1=cv.cvtColor(frame1,cv.COLOR_BGR2GRAY)
        square_scale1=np.mean(img1[-100:-1,-100:-1])

        std=np.nanstd(image*square_scale/square_scale1)

        results.append(std)
        frame_count.append(count)
print('Done')

"""
    Convert to lists to array and save data
"""
frame_count = np.array(frame_count)
results = np.array(results)
f_total = np.ndarray.item(np.array(frame_count[-1:]))
x = frame_count[::d_total]

# put into 2-d array of points. Can access with e.g. plt.plot(np.array(l)[:,14])
l = np.array_split(results, f_total) 
np.savetxt(filename2+'_mean.txt', l)


"""
    determine freezing point
"""
data = np.loadtxt(filename2+'_mean.txt')
if plotFP == True:
    # #   uncomment below to show std of drops against frames
    color = mpl.cm.nipy_spectral(np.linspace(0, 1, d_total))
    mpl.rcParams['axes.prop_cycle'] = cycler('color', color)

    for column in data.T:
        plt.scatter(x, column)
    plt.ylabel('Stdev of drop pixels')
    plt.xlabel('Frame number')
    plt.ion()
    plt.show()

#   To sort when variation large at start of data set, assumes frame 0 is freezing point 
freezing_frame=[]
kernel_size = 10 
kernel = np.ones(kernel_size) / kernel_size 
for column in data.T:
    column = np.convolve(column, kernel, mode='same')
    a_prime = column[9:-kernel_size+1] - column[8:-kernel_size]
    answer = np.argmax(np.abs(a_prime))
    freezing_frame.append(answer+8) # +8 adjusts for missing frames

print(sorted(freezing_frame))
freezing_frame = np.array(freezing_frame)  


"""
    finally calculate temperature of freezing point and save as txt file

"""

#   load text file
a = np.loadtxt(filename2+'.txt', skiprows=2)
x = freezing_frame[0:]/5 # /5 refers to fps

temp = np.interp(x, a[:,0], a[:,2])

np.savetxt(filename2+'_data.txt', temp)

print(sorted(temp))
fno='20230126_22K_500us_QBT_F01_d_ff.png'
fno='Blank_aa_ff.png'
# fno='20230126_22K_500us_QBT_F02_bb_ff.png'
plt.figure()
ax=plt.subplot(111)
plt.hist(temp,density=True,cumulative=-1,bins=np.mgrid[-45:0:2.5])
plt.xlim((-45,0)) 
plt.xlabel('T ($^\circ$C)') 
plt.ylabel('Fraction frozen') 
plt.grid()
plt.text(0.3,0.8,fno.replace('.png',''),horizontalalignment='left',verticalalignment='top',transform=ax.transAxes)
plt.savefig('../../figures/' + fno,dpi=300)
