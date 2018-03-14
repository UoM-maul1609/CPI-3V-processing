import numpy as np
import struct
def convertDataToHouseSA(bytes,ushort,order,house):
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


