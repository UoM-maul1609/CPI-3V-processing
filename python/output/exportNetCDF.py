"""
    python file to export matlab time-series data to a NetCDF file
"""
from netCDF4 import Dataset
import hdf5storage
import scipy.io as sio
import datetime
import numpy as np
import datetime as dt

def OrdinalToDatetime(ordinal):
    plaindate = datetime.date.fromordinal(int(ordinal))
    date_time = datetime.datetime.combine(plaindate, datetime.datetime.min.time())
    return date_time + datetime.timedelta(days=ordinal-int(ordinal))

def matlab2datetime(matlab_datenum):
    day = dt.datetime.fromordinal(int(matlab_datenum))
    dayfrac = dt.timedelta(days=matlab_datenum%1) - dt.timedelta(days = 366)
    return day + dayfrac

def exportNetCDF(inputTSFile='/tmp/timeseries_class.mat',outputFile='/tmp/output.nc', \
    dropBins=[1,3,5,6],iceBins=[2,4,7,8,9],description='flight C301 on '):

    # load the mat file
    print("Reading " + inputTSFile)
    dataload=hdf5storage.loadmat(inputTSFile,variable_names=['timeser'])
    print("...done")
    
    # creating netcdf file
    print("Creating netcdf file")
    ncfile = Dataset(outputFile, mode='w', format='NETCDF4_CLASSIC')

    # date
    d=matlab2datetime(dataload['timeser']['Time'][0])
    python_datetime = datetime.fromordinal(int(matlab_datenum)) + timedelta(days=matlab_datenum%1) - timedelta(days = 366)
    # description
    ncfile.title = "CPI3V concentration and size distribution data from " + description \
        + " " + d.strftime("%Y/%m/%d")
    
    ncfile.institution = "University of Manchester."
    ncfile.source = "Data collected on the FAAM BAE-146 research aircraft during the " + \
        "DCMEX experiment in Alberquerque New Mexico. The data come from the 3V-CPI " +\
        "instrument, operated by the University of Manchester."
    ncfile.history = "Version 0.1.0"
    ncfile.references = "See python version of Prof. Paul Connolly's " + \
        "processing software at https://github.com/UoM-maul1609/CPI-3V-processing"
    ncfile.comment="Data are processed with commit " + \
        "0aca881dd68156c48b9850bd50dc9e6449904eb9 of the software. " + \
        "Note there are some particles that are unclassified (look at the difference" +\
        " between total and drops+ice)"
    
    # dimensions
    time_dim = ncfile.createDimension('time',len(dataload['timeser']['Time'][:]))
    size_dim = ncfile.createDimension('size',len(dataload['timeser']['size1'][:]))
    
    # variables
    time = ncfile.createVariable('time', np.float64, ('time',) )
    time.units = 'hours since ' + d.strftime("%Y/%m/%d %H:%M:%S.%MS")
    time.long_name = "time in hours since " + d.strftime("%Y/%m/%d %H:%M:%S.%MS")
    
    size1 = ncfile.createVariable('size1', np.float64, ('size',) )
    size1.units = 'micrometres'
    size1.long_name = 'lower bin edge in microns'
    
    size2 = ncfile.createVariable('size2', np.float64, ('size',) )
    size2.units = 'micrometres'
    size2.long_name = 'upper bin edge in microns'
    
    # total
    conc = ncfile.createVariable('conc', np.float64, ('time',) )
    conc.units = 'm-3'
    conc.long_name = "total number concentration in m-3" 
    
    conc2 = ncfile.createVariable('conc2', np.float64, ('time','size') )
    conc2.units = 'm-3'
    conc2.long_name = "total 2d number concentration in m-3" 
    
    
    # drops
    concDrops = ncfile.createVariable('concDrops', np.float64, ('time',) )
    concDrops.units = 'm-3'
    concDrops.long_name = "drop number concentration in m-3 (only > 50 micron)" 
    
    conc2Drops = ncfile.createVariable('conc2Drops', np.float64, ('time','size') )
    conc2Drops.units = 'm-3'
    conc2Drops.long_name = "drop 2d number concentration in m-3 (only > 50 micron)" 
    

    # ice
    concIce = ncfile.createVariable('concIce', np.float64, ('time',) )
    concIce.units = 'm-3'
    concIce.standard_name = "number_concentration_of_ice_crystals_in_air" 
    concIce.long_name = "ice number concentration in m-3 (only > 50 micron)"
    
    conc2Ice = ncfile.createVariable('conc2Ice', np.float64, ('time','size') )
    conc2Ice.units = 'm-3'
    conc2Ice.long_name = "ice 2d number concentration in m-3 (only > 50 micron)" 
    


    # write the data
    time[:] = 24.*(dataload['timeser']['Time'][:]-dataload['timeser']['Time'][0])
    size1[:] = dataload['timeser']['size1'][:]
    size2[:] = dataload['timeser']['size2'][:]
    conc[:] = dataload['timeser']['conc'][:]
    conc2[:,:] = dataload['timeser']['conc2'][:,:]
    
    concDrops[:] = np.sum(np.sum( \
        dataload['timeser']['conc2ar'][:,:,dropBins],axis=2),axis=1)
    conc2Drops[:,:] = np.sum(dataload['timeser']['conc2ar'][:,:,dropBins],axis=2)
    
    concIce[:] = np.sum(np.sum( \
        dataload['timeser']['conc2ar'][:,:,iceBins],axis=2),axis=1)
    conc2Ice[:,:] = np.sum(dataload['timeser']['conc2ar'][:,:,iceBins],axis=2)
    


    
    ncfile.close()
    print("...file is closed")
    
    
    
if __name__=="__main__":
    exportNetCDF()
