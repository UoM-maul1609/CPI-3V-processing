import numpy as np
import copy

def fullBackgrounds(ROI_N):

    #https://docs.scipy.org/doc/numpy-1.9.3/user/basics.rec.html
    # structured array with dictionary argument
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # search for backgrounds
    (ind,i1)=np.where(ROI_N['imageType']==19)
    if len(ind) == 0:
        print('no backgrounds in this file')
        FULL_BG={'Time':np.array([]),'IMAGE':np.array([]) }
        return FULL_BG
    
    #--------------------------------------------------------------------------
    
    
    
    # +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # rename elements of array of struct
    # https://blogs.mathworks.com/loren/2010/05/13/rename-a-field-in-a-structure-array/
    FULL_BG = {'Time' : ROI_N['Time'][ind], 'IMAGE' : [dict() for x in range(len(ind))] }
    
    
    #https://stackoverflow.com/questions/16475384/rename-a-dictionary-key
    for i in range(len(FULL_BG['Time'])):
        FULL_BG['IMAGE'][i]['BG']=ROI_N['IMAGE'][i]['IM']
    #--------------------------------------------------------------------------
    
    
    return FULL_BG