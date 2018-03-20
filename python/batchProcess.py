foc_crit=12 # critical value of focus for an image
min_len=100 # minimum length for particle images
dt=10  # resolution on time-step for concentrations (take with a pince of salt)
ds=10  # resolution for size bins
vel=100    # air speed - assumed fixed, used in calcTimeseriesDriver
find_particle_edges=True # output the boundary of the particles
command_line_path=True # use the commandline to define the path of files
process_sweep1_if_exist=True # if the *.roi files have been extracted once,
                              #still do if True
process_roi_driver=True
process_image_stats=True
num_cores=4

path1='/Users/mccikpc2/Dropbox (The University of Manchester)/data/'
            # path to raw data
filename1=['20180109105546.roi', 
    '20180109120515.roi']
            # list of filenames to process

outputfile='timeseries.mat'


import gc

# get the files / path from commandline input
if command_line_path:
    import sys
    path1=sys.argv[1]
    #path1='/Users/mccikpc2/Dropbox (The University of Manchester)/data/'
    #/tmp/cpi_struct/'
    from os import listdir
    #from os.path import isfile, join
    filename1 = [f for f in listdir(path1) if f.endswith(".roi")]
    del sys, listdir

#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if process_roi_driver:
    from ROIDataDriver import ROIDataDriver
   # extract ROI data from files and prcess with backgrounds
    (t_range)= \
       ROIDataDriver(path1,filename1,dt,process_sweep1_if_exist)
    # Garbage collection:
    gc.collect()
    del gc.garbage[:]
    del ROIDataDriver
#--------------------------------------------------------------------------


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
if process_image_stats:
    from imageStatsDriver import imageStatsDriver
    # find image properties, edge detection, etc
    imageStatsDriver(path1,filename1,find_particle_edges,num_cores=num_cores)

    del imageStatsDriver
#--------------------------------------------------------------------------


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# export images
from exportImagesDriver import exportImagesDriver
exportImagesDriver(path1,filename1,foc_crit,min_len)
del exportImagesDriver
#--------------------------------------------------------------------------


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# calculate number concentrations one day at a time?
from calcTimeseriesDriver import calcTimeseriesDriver
calcTimeseriesDriver(path1,filename1,foc_crit,dt,ds,vel,outputfile)
#--------------------------------------------------------------------------
