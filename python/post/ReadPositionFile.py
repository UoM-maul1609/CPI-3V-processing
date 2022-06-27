import numpy as np
import matplotlib.pyplot as plt
import datetime as dt

def datetime2matlabdn(dt1):
   mdn = dt1 + dt.timedelta(days = 366)
   frac_seconds = (dt1-dt.datetime(dt1.year,dt1.month,dt1.day,0,0,0)).seconds / (24.0 * 60.0 * 60.0)
   frac_microseconds = dt1.microsecond / (24.0 * 60.0 * 60.0 * 1000000.0)
   return mdn.toordinal() + frac_seconds + frac_microseconds
   
def ReadFile(fileName='/Users/mccikpc2/Downloads/CPI_cals/Matlab/Analogue_All_2.txt'):
    f = open(fileName,'r')
    str1=f.readlines()
    
    posx  = np.zeros(len(str1)-1)
    posy  = np.zeros(len(str1)-1)
    times = np.zeros(len(str1)-1)
    times1=[]
    
    for i in range(1,len(str1)):
        list1=str1[i].split()
        posx[i-1]  = float(list1[0])
        posy[i-1]  = float(list1[1])
        
        dto        = dt.datetime.strptime(list1[2] + ' ' + list1[3],'%d/%m/%y %H:%M:%S')
        times1.append(dto)
        times[i-1] = (times1[i-1]-times1[0]).total_seconds()
    
    f.close()
    
    # matlab time
    for i in range(1,len(str1)):
        times[i-1]=datetime2matlabdn(times1[i-1])
    
    
    return (posx,posy,times)
    
if __name__=="__main__":
    (posx,posy,times)=ReadFile()
    
    plt.ion()
    plt.show()
    plt.plot(times,posx,'.')    
    plt.plot(times,posy,'^')