foc_crit=12 # critical value of focus for an image
min_len=100 # minimum length for particle images
dt=10  # resolution on time-step for concentrations (take with a pince of salt)
ds=10  # resolution for size bins
vel=100    # air speed - assumed fixed, used in calcTimeseriesDriver
find_particle_edges=True # output the boundary of the particles
command_line_path=True # use the commandline to define the path of files
process_sweep1_if_exist=False # if the *.roi files have been extracted once,
                              #still do if True

path1='/Users/mccikpc2/Dropbox (The University of Manchester)/data/'
            # path to raw data
filename1=['20180109105546.roi', 
    '20180109120515.roi']
            # list of filenames to process

outputfile='timeseries.mat'


from ROIDataDriver import ROIDataDriver
from imageStatsDriver import imageStatsDriver
from exportImagesDriver import exportImagesDriver
from calcTimeseriesDriver import calcTimeseriesDriver
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


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# extract ROI data from files and prcess with backgrounds
(bytes1,house,images,rois,ushort,Header,I,R,H,t_range)= \
   ROIDataDriver(path1,filename1,dt,process_sweep1_if_exist)
del bytes1, house, images, rois, ushort, Header, I, R, H
# Garbage collection:
gc.collect()
del gc.garbage[:]
#--------------------------------------------------------------------------


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# find image properties, edge detection, etc
imageStatsDriver(path1,filename1,find_particle_edges)
#--------------------------------------------------------------------------


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# export images
exportImagesDriver(path1,filename1,foc_crit,min_len)
#--------------------------------------------------------------------------


#++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
# calculate number concentrations one day at a time?
calcTimeseriesDriver(path1,filename1,foc_crit,t_range[0],dt,ds,vel,outputfile)
#--------------------------------------------------------------------------
