import numpy as np
from scipy.interpolate import interp1d

def associateBackgrounds(ROI_N,FULL_BG):

    
    if len(FULL_BG['Time'][0,0][0,:]) == 0:
        print('Error, there are no backgrounds in these files')
        
    ind=np.mgrid[0:len(FULL_BG['Time'][0,0][0,:])]

    #--------------------------------------------------------------------------
    BG = [dict() for x in range(len(ROI_N['IMAGE'][0,0][0,:]))]

    for i in range(len(ROI_N['IMAGE'][0,0][0,:])):
       if len(ind)==1:
           jj=int(ind)
       else:
          Timecur=ROI_N['Time'][0,0][i,0]
          #https://docs.scipy.org/doc/scipy/reference/generated/scipy.interpolate.interp1d.html
          f=interp1d(FULL_BG['Time'][0,0][0,:],ind,kind='nearest',fill_value='extrapolate')
          jj=int(f(Timecur))
       
       
       xl=ROI_N['StartX'][0,0][0,i]
       xu=ROI_N['EndX'][0,0][0,i]+1
       yl=ROI_N['StartY'][0,0][0,i]
       yu=ROI_N['EndY'][0,0][0,i]+1
       BG[i]['BG'] = FULL_BG[0,0]['IMAGE'][0,0][0,jj]['BG'][xl:xu,yl:yu]
       
    
    
    return BG