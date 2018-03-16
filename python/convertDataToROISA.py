import numpy as np
import struct
def convertDataToROISA(bytes1,ushort,order,rois):
    # https://stackoverflow.com/questions/3648442/how-to-define-a-structure-like-in-c
    #https://stackoverflow.com/questions/5824530/python-struct-arrays
    #https://docs.scipy.org/doc/numpy-1.9.3/user/basics.rec.html
    # structured array with dictionary argument
    """    
    ulItemSize   # ulong
    usVersion   # ushort
    StartX   # ushort
    StartY   # ushort
    EndX   # ushort
    EndY   # ushort
    PixBytes   # short
    usROIFlags   # ushort
    fLength   # float
    ulStartLen   # ulong
    ulEndLen   # ulong
    fWidth   # float
    Spare   # (18,1)char
    order       
    """
    
    R=np.zeros(len(rois), 
            dtype=[('ulItemSize', 'uint32'),
                    ('usVersion', 'uint16'),
                    ('StartX', 'uint16'),
                    ('StartY', 'uint16'),
                    ('EndX', 'uint16'),
                    ('EndY', 'uint16'),
                    ('PixBytes', 'int16'),
                    ('usROIFlags', 'uint16'),
                    ('fLength', 'float'),
                    ('ulStartLen', 'uint32'),
                    ('ulEndLen', 'uint32'),
                    ('fWidth', 'float'),
                    ('Spare','(18,1)b'),
                    ('order', 'uint32') ])
    
    
    for i in range(len(rois)):
        ro=2*rois[i] # note, for same indexing as MATLAB this should be (2(rois+1)-1)
                     # but this has been taken into account in the indexing
        # although ulong, it appears uint is needed here:
        #https://docs.python.org/2/library/struct.html
        # ulong would be 8 bytes1
        R['ulItemSize'][i],=struct.unpack('I', bytes1[ro+2:ro+2+4]) 
        R['usVersion'][i]=ushort[rois[i]+3]
                
        R['StartX'][i]=ushort[rois[i]+4]
        R['StartY'][i]=ushort[rois[i]+5]
        R['EndX'][i]=ushort[rois[i]+6]
        R['EndY'][i]=ushort[rois[i]+7]
        R['PixBytes'][i],=struct.unpack('h', bytes1[ro+16:ro+16+2]) 
        R['usROIFlags'][i]=ushort[rois[i]+9]
        
        R['fLength'][i],=struct.unpack('f', bytes1[ro+20:ro+20+4]) 

        R['ulStartLen'][i],=struct.unpack('I', bytes1[ro+24:ro+24+4]) 
        R['ulEndLen'][i],=struct.unpack('I', bytes1[ro+28:ro+28+4]) 
        
        R['fWidth'][i],=struct.unpack('f', bytes1[ro+32:ro+32+4]) 

        R['Spare'][i]=np.reshape(bytearray(bytes1[ro+36:ro+36+18]),(18,1))

        
        R['order'][i]=order[rois[i]]
       
    
    
    ver=R['usVersion']
    ind,=np.where(ver==25)
    R=R[ind]
    rois=rois[ind]

    return (R,rois)


