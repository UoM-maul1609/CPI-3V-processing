import numpy as np
import struct
def convertDataToHouseSA(bytes1,ushort,order,house,cpiv1):
    # https://stackoverflow.com/questions/3648442/how-to-define-a-structure-like-in-c
    #https://stackoverflow.com/questions/5824530/python-struct-arrays
    #https://docs.scipy.org/doc/numpy-1.9.3/user/basics.rec.html
    # structured array with dictionary argument
    """  
    BlockNum     # ushort
    ulItemSize   # ulong
    Readings1   # (70,1)ushort
    TimeMSW   # ushort
    TimeISW   # ushort
    TimeLSW   # ushort
    Readings   # (8,1)ushort
    order       
    """
    
    if cpiv1:
        H=np.zeros(len(house), 
            dtype=[('BlockNum', 'uint16'),
                    ('ulItemSize', 'uint32'),
                    ('usVersion', 'uint16'),
                    ('ulXtraData', 'uint32'),
                    ('bNewData', 'uint16'),
                    ('usChkSumFlag', 'uint16'),
                    ('TopReadings', '(3,1)uint16'),
                    ('Time','uint32'),
                    ('Readings','(40,1)uint16'),
                    ('order', 'uint32') ])
                    
        for i in range(len(house)):
            if (ushort[house[i]+1] == 83):
                H['BlockNum'][i]=ushort[house[i]+0]
                H['ulItemSize'][i]=(ushort[house[i]+1] << 16) + ushort[house[i]+2]
                H['usVersion'][i]=ushort[house[i]+3]
                H['ulXtraData'][i]=(ushort[house[i]+4] << 16) + ushort[house[i]+5]
                H['bNewData'][i]=ushort[house[i]+6]
                H['usChkSumFlag'][i]=ushort[house[i]+7]
                H['TopReadings'][i]=np.reshape(ushort[house[i]+8:house[i]+8+3],(3,1))
                H['Time'][i] = (ushort[house[i]+11] << 16) + ushort[house[i]+12]
                H['Readings'][i]=np.reshape(ushort[house[i]+13:house[i]+13+40],(40,1))
                H['order'][i]=order[house[i]]
    
    else:
        H=np.zeros(len(house), 
            dtype=[('BlockNum', 'uint16'),
                    ('ulItemSize', 'uint16'),
                    ('Readings1', '(70,1)uint16'),
                    ('TimeMSW', 'uint16'),
                    ('TimeISW', 'uint16'),
                    ('TimeLSW', 'uint16'),
                    ('Readings', '(8,1)int16'),
                    ('order', 'uint32') ])
    
    
        for i in range(len(house)):
            if (ushort[house[i]+1] == 83):
                H['BlockNum'][i]=ushort[house[i]+0]
                H['ulItemSize'][i]=ushort[house[i]+1]
                H['Readings1'][i]=np.reshape(ushort[house[i]+2:house[i]+2+70],(70,1))
                H['TimeMSW'][i]=ushort[house[i]+72]
                H['TimeISW'][i]=ushort[house[i]+73]
                H['TimeLSW'][i]=ushort[house[i]+74]
                H['Readings'][i]=np.reshape(ushort[house[i]+75:house[i]+75+8],(8,1))   
                H['order'][i]=order[house[i]]
        
    
    
    ver=H['ulItemSize']
    ind,=np.where(ver==83)
    H=H[ind]

    return (H)


