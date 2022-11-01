import numpy as np
import struct
def convertDataToImageSA(bytes1,ushort,order,images):
    # https://stackoverflow.com/questions/3648442/how-to-define-a-structure-like-in-c
    #https://stackoverflow.com/questions/5824530/python-struct-arrays
    #https://docs.scipy.org/doc/numpy-1.9.3/user/basics.rec.html
    # structured array with dictionary argument
    """    
    ulItemSize   # ulong
    usVersion   # ushort
    usROIsCount   # ushort
    ulTotROISize   # ulong
    day   # uchar
    hour   # uchar 
    minute   # uchar
    second   # uchar
    msecond   # ushort
    ImageType   # ushort
    StartX   # ushort
    StartY   # ushort
    EndX   # ushort
    EndY   # ushort
    BGRate   # ushort
    usBkgPDSThresh   # ushort
    ulFrmsProc   # ulong
    IThreshold   # uchar
    ROIError   # uchar
    ROIMinSize   # ushort
    ROIAspectRatio   # float
    ROIFillRatio   # float
    ROIFCount   # ulong
    ucImgMean   # uchar
    ucBkgMean   # uchar
    Spare1   # short
    ROIXPad   # ushort
    ROIYPad   # ushort
    ulStrobeCount   # ulong
    ulFrmsSaved   # ulong
    ImgMinVal   # uchar
    ImgMaxVal   # uchar
    ulROIsSaved   # ulong
    usPDSChkSumFlag   # ushort
    usPDSHead=np.array(3)    # ushort
    ulTime   # ulong
    ArrivalTime1   # ushort
    ArrivalTime2   # ushort
    TransitTime   # ushort
    usStrobes   # ushort
    PulseInt45   # ushort
    PulseInt90   # ushort
    PDSChkSum   # ushort
    ProbeMode   # ushort
    order       
    """
    
    I=np.zeros(len(images), 
            dtype=[('ulItemSize', 'uint32'),
                    ('usVersion', 'uint16'),
                    ('usROIsCount', 'uint16'),
                    ('ulTotROISize', 'uint32'),
                    ('day', 'B'),
                    ('hour', 'B'),
                    ('minute', 'B'),
                    ('second', 'B'),
                    ('msecond', 'uint16'),
                    ('ImageType', 'uint16'),
                    ('StartX', 'uint16'),
                    ('StartY', 'uint16'),
                    ('EndX', 'uint16'),
                    ('EndY', 'uint16'),
                    ('BGRate', 'uint16'),
                    ('usBkgPDSThresh', 'uint16'),
                    ('ulFrmsProc', 'uint32'),
                    ('IThreshold', 'B'),
                    ('ROIError', 'B'),
                    ('ROIMinSize', 'uint16'),
                    ('ROIAspectRatio', 'float'),
                    ('ROIFillRatio', 'float'),
                    ('ROIFCount', 'uint32'),
                    ('ucImgMean', 'B'),
                    ('ucBkgMean', 'B'),
                    ('Spare1', 'int16'),
                    ('ROIXPad', 'uint16'),
                    ('ROIYPad', 'uint16'),
                    ('ulStrobeCount', 'uint32'),
                    ('ulFrmsSaved', 'uint32'),
                    ('ImgMinVal', 'B'),
                    ('ImgMaxVal', 'B'),
                    ('ulROIsSaved', 'uint32'),
                    ('usPDSChkSumFlag', 'uint16'),
                    ('usPDSHead','(3,1)uint16'),
                    ('ulTime', 'uint32'),
                    ('ArrivalTime1', 'uint16'),
                    ('ArrivalTime2', 'uint16'),
                    ('TransitTime', 'uint16'),
                    ('usStrobes', 'uint16'),
                    ('PulseInt45', 'uint16'),
                    ('PulseInt90', 'uint16'),
                    ('PDSChkSum', 'uint16'),
                    ('ProbeMode', 'uint16'),
                    ('order', 'uint32') ])
    
    
    for i in range(len(images)):
        im=2*images[i] # note, for same indexing as MATLAB this should be (2(images+1)-1)
                       # but this has been taken into account in the indexing
        # although ulong, it appears uint is needed here:
        #https://docs.python.org/2/library/struct.html
        # ulong would be 8 bytes1
        I['ulItemSize'][i],=struct.unpack('I', bytes1[im+2:im+2+4]) 
        I['usVersion'][i]=ushort[images[i]+3]
        
        if((I['usVersion'][i]==40) or (I['usVersion'][i]==25)):
            I['usROIsCount'][i]=ushort[images[i]+4]
            I['ulTotROISize'][i],=struct.unpack('I', bytes1[im+10:im+10+4]) 
            
            I['day'][i]=bytes1[im+14]
            I['hour'][i]=bytes1[im+15]
            I['minute'][i]=bytes1[im+16]
            I['second'][i]=bytes1[im+17]
    
            I['msecond'][i]=ushort[images[i]+9]
            I['ImageType'][i]=ushort[images[i]+10]
            
            I['StartX'][i]=ushort[images[i]+11]
            I['StartY'][i]=ushort[images[i]+12]
            I['EndX'][i]=ushort[images[i]+13]
            I['EndY'][i]=ushort[images[i]+14]
            I['BGRate'][i]=ushort[images[i]+15]
            I['usBkgPDSThresh'][i]=ushort[images[i]+16]
            
            I['ulFrmsProc'][i],=struct.unpack('I', bytes1[im+34:im+34+4]) 
            
            I['IThreshold'][i]=bytes1[im+38]
            I['ROIError'][i]=bytes1[im+39]
            
            I['ROIMinSize'][i]=ushort[images[i]+20]
            
            I['ROIAspectRatio'][i],=struct.unpack('f', bytes1[im+42:im+42+4]) 
            I['ROIFillRatio'][i],=struct.unpack('f', bytes1[im+46:im+46+4]) 
            
            I['ROIFCount'][i],=struct.unpack('I', bytes1[im+50:im+50+4]) 
            
            I['ucImgMean'][i]=bytes1[im+54]
            I['ucBkgMean'][i]=bytes1[im+55]
            
            I['Spare1'][i]=ushort[images[i]+28]
            I['ROIXPad'][i]=ushort[images[i]+29]
            I['ROIYPad'][i]=ushort[images[i]+30]
            
            I['ulStrobeCount'][i],=struct.unpack('I', bytes1[im+62:im+62+4]) 
            I['ulFrmsSaved'][i],=struct.unpack('I', bytes1[im+66:im+66+4]) 
    
            I['ImgMinVal'][i]=bytes1[im+70]
            I['ImgMaxVal'][i]=bytes1[im+71]
        
            I['ulROIsSaved'][i],=struct.unpack('I', bytes1[im+72:im+72+4]) 
    
            I['usPDSChkSumFlag'][i]=ushort[images[i]+38]
            I['usPDSHead'][i]=np.reshape(ushort[images[i]+39:images[i]+41+1],(3,1))
            
            I['ulTime'][i],=struct.unpack('I', bytes1[im+84:im+84+4]) 
            I['ArrivalTime1'][i]=ushort[images[i]+43]
            I['ArrivalTime2'][i]=ushort[images[i]+44]
    
            I['TransitTime'][i]=ushort[images[i]+46]
            I['usStrobes'][i]=ushort[images[i]+47]
            I['PulseInt45'][i]=ushort[images[i]+48]
            I['PulseInt90'][i]=ushort[images[i]+49]
            I['PDSChkSum'][i]=ushort[images[i]+50]
            I['ProbeMode'][i]=ushort[images[i]+51]
    
            
            I['order'][i]=order[images[i]]
       
    
    
    ver=I['usVersion']
    day=I['day']
    ind,=np.where(((ver==40) | (ver==25)) & (day>0)  )
    I=I[ind]
    images=images[ind]

    return (I,images)


