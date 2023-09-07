from scipy.interpolate import interp1d

if __name__=="__main__":
    import ReadCPIFiles
    import ReadPositionFile
    
    """
        Read the CPI files
    """
    (times,foc,len1,im1,boundaries,x,y)=ReadCPIFiles.ReadMAT( \
        pathName='/Users/mccikpc2/Downloads/CPI_cals/CPICalibration/Cal210302')
    """
        Read the positional calibration data
    """
    (posx,posy,timesp)=ReadPositionFile.ReadFile(\
        fileName='/Users/mccikpc2/Downloads/CPI_cals/Matlab/ALL_DAYS_POSITION_TIME_4.txt')
    
    
    PosXInt = interp1d(timesp,posx,kind='previous',fill_value='extrapolate')
    PosYInt = interp1d(timesp,posy,kind='previous',fill_value='extrapolate')   
    
    posxx=PosXInt(times) 
    posyy=PosYInt(times) 
        