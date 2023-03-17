import numpy as np
from datetime import datetime
from datetime import timedelta
def postProcess(bytes1,rois,R,H,I,Header,cpiv1):
    # use dictionaries for storing info
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # image times in hk format
    IMAGE1={'Time' : I['ArrivalTime2']*16**8+
            (I['ulTime']/65536.)*16**4+
            I['ArrivalTime1']}
    
    # time in MATLAB format, using msecond, etc
    (Time,Timestr)=calc_datetime(Header['usYear'],
                       Header['ucMonth'],
                       I['day'],
                       I['hour'],
                       I['minute'],
                       I['second'],
                       I['msecond']
                       )

    IMAGE1['Time1']=Time
    IMAGE1['Timestr']=Timestr
    IMAGE1['imageType']=I['ImageType']
    #--------------------------------------------------------------------------
    

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if not cpiv1:
        # housekeeping packet times
        HOUSE={'Time' : H['TimeMSW']*16**8 +
               H['TimeISW']*16**4 +
               H['TimeLSW'] }
        Rdgs=H['Readings1']
        #HOUSE['deadtime']=Rdgs[:,66]*0.000341333
        HOUSE['deadtime']=Rdgs[:,57] /256
    else:
        # housekeeping packet times
        HOUSE={'Time' : H['Time'] }
        Rdgs=H['Readings']
        HOUSE['deadtime']=Rdgs[:,31]*0.000682667
        
    #--------------------------------------------------------------------------

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # map 'IMAGE' to ROI
    orderImage=I['order']
    orderROIs=R['order']
    timeROIs=np.zeros((len(R),1))
    imageTypeROIs=np.zeros((len(R),1))
    imageMeans=np.zeros((len(R),1))
    
    for i in range(len(orderImage)-1):
        ind,=np.where((orderImage[i]<orderROIs) & (orderImage[i+1]>orderROIs)) 
        timeROIs[ind]=Time[i]
        imageTypeROIs[ind]=IMAGE1['imageType'][i]
        imageMeans[ind]=I['ucImgMean'][i]
    
    i=len(orderImage)-1
    ind,=np.where(orderImage[i]<orderROIs )
    timeROIs[ind]=Time[i]
    imageTypeROIs[ind]=IMAGE1['imageType'][i]
    imageMeans[ind]=I['ucImgMean'][i]
    #--------------------------------------------------------------------------
    

    #https://stackoverflow.com/questions/2397754/how-can-i-create-an-array-list-of-dictionaries-in-python
    ROI_N={'Time' : timeROIs,  'IMAGE' : [dict() for x in range(len(rois))] }    
    #indroi=np.argsort(ROI_N['Time'][:,0])
    indroi=np.argsort(R['order'])
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # associate variables with each ROI++++++++++++++++++++++++++++++++++++++++
    for i in range(len(rois)):
        StartX=R['StartX'][indroi[i]]
        StartY=R['StartY'][indroi[i]]
        EndX=R['EndX'][indroi[i]]
        EndY=R['EndY'][indroi[i]]
        
        X=(EndX-StartX+1)
        Y=(EndY-StartY+1)
        numberOfChars=X*Y
    
        chars1=bytearray(
                bytes1[(rois[indroi[i]]+1)*2-1+33*2-14+1:\
                       (rois[indroi[i]]+1)*2-1+33*2-14+1+numberOfChars])
        #IM=np.reshape(chars1,(X,Y))
        ROI_N['IMAGE'][i]['IM']=np.transpose(np.reshape(chars1,(Y,X)))
        
    
    ROI_N['imageType']=imageTypeROIs
    ROI_N['StartX']=R['StartX']
    ROI_N['StartY']=R['StartY']
    ROI_N['EndX']=R['EndX']
    ROI_N['EndY']=R['EndY']
    ROI_N['IM']=imageMeans
    #--------------------------------------------------------------------------

    # sort
    #indroi=np.argsort(ROI_N['Time'][:,0])
    indroi=np.argsort(R['order'])
    ROI_N['Time'][:,:]=ROI_N['Time'][indroi,:]
    ROI_N['imageType'][:]=ROI_N['imageType'][indroi]
    ROI_N['StartX'][:]=ROI_N['StartX'][indroi]
    ROI_N['StartY'][:]=ROI_N['StartY'][indroi]
    ROI_N['EndX'][:]=ROI_N['EndX'][indroi]
    ROI_N['EndY'][:]=ROI_N['EndY'][indroi]
    ROI_N['IM'][:]=ROI_N['IM'][indroi]
    #ROI_N['IMAGE'][:]=ROI_N['IMAGE'][indroi]
    # sort
    #indhouse=np.argsort(HOUSE['Time'])
    indhouse=np.argsort(H['order'])
    HOUSE['Time'][:]=HOUSE['Time'][indhouse]
    HOUSE['deadtime'][:]=HOUSE['deadtime'][indhouse]
    
    # sort
    #indimage=np.argsort(IMAGE1['Time'][:])
    indimage=np.argsort(I['order'])
    IMAGE1['Time'][:]=IMAGE1['Time'][indimage]
    IMAGE1['Time1'][:,:]=IMAGE1['Time1'][indimage,:]
    IMAGE1['Timestr'][:]=IMAGE1['Timestr'][indimage]
    IMAGE1['imageType'][:]=IMAGE1['imageType'][indimage]
    
    return (ROI_N,HOUSE,IMAGE1)





def calc_datetime(year,mon,day,hour,mins,sec,msec):
    mtime=np.zeros((len(hour),1))
    mtimestr=np.chararray(len(hour),24)
    for i in range(len(hour)):
        d=datetime(year,mon,day[i],hour[i],mins[i],sec[i],
                 msec[i]*1000)
        mtime[i]=datetime2matlabdn(d)
        
        # http://strftime.org
        str1="{0}{1}{2}{3}{4}{5}{6}{7}{8}".format(datetime.strftime(d,'%d'), '-', \
            datetime.strftime(d,'%b'), '-', \
            datetime.strftime(d,'%Y'), ' ', \
            datetime.strftime(d,'%X'), '.', \
            datetime.strftime(d,'%f')[0:3])
        mtimestr[i]=str1
        
    return (mtime,mtimestr)

#https://stackoverflow.com/questions/8776414/python-datetime-to-matlab-datenum
def datetime2matlabdn(dt):
    mdn = dt + timedelta(days = 366)
    frac_seconds = (dt-datetime(dt.year,dt.month,dt.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
    frac_microseconds = dt.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
    return mdn.toordinal() + frac_seconds + frac_microseconds
