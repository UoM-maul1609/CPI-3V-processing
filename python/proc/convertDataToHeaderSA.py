import numpy as np

def convertDataToHeaderSA(ushort):

    #https://docs.scipy.org/doc/numpy-1.9.3/user/basics.rec.html
    # structured array with dictionary argument
    """    
        ucVersion=1
        usYear=1
        ucMonth=1
        ImageX=1
        ImageY=1
        szInfo=np.array(70)
    """
    
    Header,=np.zeros(1, 
            dtype=[('ucVersion','uint16'), 
                   ('usYear','uint16'), 
                   ('ucMonth','uint16'), 
                   ('ImageX','uint16'), 
                   ('ImageY','uint16'), 
                   ('szInfo','(70,1)uint16') ])
    
    Header['ucVersion']=ushort[0]
    Header['usYear']=ushort[1]
    Header['ucMonth']=ushort[2]
    Header['ImageX']=ushort[3]
    Header['ImageY']=ushort[4]
    Header['szInfo']=np.reshape(ushort[5:5+70],(70,1))
    
    
    return Header