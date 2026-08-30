import numpy as np
from datetime import datetime
from datetime import timedelta
def postProcess(bytes1,rois,R,H,I,Header,cpiv1):
    # use dictionaries for storing info
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # image times in hk format
    # Reconstruct the instrument time counter using an explicitly wide dtype.
    # NumPy 2.x no longer promotes uint16 arrays when multiplying by Python
    # integers outside the uint16 range (e.g. 16**8 == 2**32), so the legacy
    # expression raises OverflowError.  Use float64 here to preserve the dtype
    # produced by the historical expression (which contained a floating divide).
    image_time = (
        I['ArrivalTime2'].astype(np.float64) * float(16**8)
        + I['ulTime'].astype(np.float64)
        + I['ArrivalTime1'].astype(np.float64)
    )
    IMAGE1={'Time' : image_time}
    
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
        # Housekeeping timestamps are three 16-bit words.  Promote before
        # shifting/multiplication so the arithmetic cannot overflow uint16.
        HOUSE={'Time' : (
               H['TimeMSW'].astype(np.uint64) * np.uint64(16**8)
               + H['TimeISW'].astype(np.uint64) * np.uint64(16**4)
               + H['TimeLSW'].astype(np.uint64))}
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
    
    # Each ROI belongs to the immediately preceding IMAGE block in file order.
    # The legacy implementation scanned every ROI once for every image block;
    # searchsorted gives the same mapping without the O(Nimage*Nroi) loop.
    if len(orderImage):
        parent_image = np.searchsorted(orderImage, orderROIs, side='left') - 1
        valid = parent_image >= 0
        timeROIs[valid, 0] = Time[parent_image[valid], 0]
        imageTypeROIs[valid, 0] = IMAGE1['imageType'][parent_image[valid]]
        imageMeans[valid, 0] = I['ucImgMean'][parent_image[valid]]
    #--------------------------------------------------------------------------
    

    #https://stackoverflow.com/questions/2397754/how-can-i-create-an-array-list-of-dictionaries-in-python
    ROI_N={'Time' : timeROIs,  'IMAGE' : [dict() for x in range(len(rois))] }    
    #indroi=np.argsort(ROI_N['Time'][:,0])
    indroi=np.argsort(R['order'])
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # associate variables with each ROI++++++++++++++++++++++++++++++++++++++++
    for i in range(len(rois)):
        # R stores coordinates as uint16.  Convert to Python ints *before*
        # subtracting/multiplying: under NumPy 2.x scalar uint16 arithmetic
        # remains uint16, so a full 1280 x 1024 image gives
        # 1,310,720 mod 65,536 == 0.
        source_index = int(indroi[i])
        StartX = int(R['StartX'][source_index])
        StartY = int(R['StartY'][source_index])
        EndX = int(R['EndX'][source_index])
        EndY = int(R['EndY'][source_index])

        X = EndX - StartX + 1
        Y = EndY - StartY + 1
        if X <= 0 or Y <= 0:
            raise ValueError(
                f"Invalid ROI dimensions for ROI {i}: "
                f"Start=({StartX},{StartY}), End=({EndX},{EndY})"
            )

        numberOfChars = X * Y
        roi_word = int(rois[source_index])

        # Preserve the legacy byte offset exactly, but calculate it with
        # ordinary Python integers so it cannot wrap in a NumPy dtype.
        pixel_start = (roi_word + 1) * 2 - 1 + 33 * 2 - 14 + 1
        pixel_end = pixel_start + numberOfChars
        chars1 = bytes1[pixel_start:pixel_end]

        if len(chars1) != numberOfChars:
            raise ValueError(
                f"Incomplete ROI pixel data for ROI {i}: expected "
                f"{numberOfChars} bytes for {X}x{Y}, got {len(chars1)}; "
                f"byte range [{pixel_start}:{pixel_end}], "
                f"buffer length {len(bytes1)}"
            )

        ROI_N['IMAGE'][i]['IM'] = np.frombuffer(
            chars1, dtype=np.uint8
        ).reshape(Y, X).T.copy()
        
    
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
    mtimestr = np.empty(len(hour), dtype='<U24')
    for i in range(len(hour)):
        # Convert NumPy integer scalars to Python ints before arithmetic.
        # With NumPy 2.x, np.uint16(999) * 1000 remains uint16 and wraps.
        d=datetime(int(year), int(mon), int(day[i]), int(hour[i]),
                   int(mins[i]), int(sec[i]), int(msec[i]) * 1000)
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
