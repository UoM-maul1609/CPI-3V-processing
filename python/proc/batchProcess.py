foc_crit=5 # critical value of focus for an image
min_len=1 # minimum length for particle images
dt=10  # resolution on time-step for concentrations (take with a pince of salt)
ds=10  # resolution for size bins
vel=100    # air speed - assumed fixed, used in calcTimeseriesDriver
find_particle_edges=True # output the boundary of the particles
command_line_path=True # use the commandline to define the path of files
process_sweep1_if_exist=True # if the *.roi files have been extracted once,
                              #still do if True
process_roi_driver=True
process_image_stats=True
export_images=True
output_timeseries=True
num_cores=24
cpiv1 = False

path1='/tmp/CPICalibration/Cal140302/'
            # path to raw data
filename1=['20180213065852.roi','20180213092819.roi','20180213055057.roi','20180213060933.roi']
            # list of filenames to process

filename1=['20180213025037.roi']        
filename1=['11291442.roi','12011408.roi','12011606.roi','12011642.roi','12011735.roi']        
        
outputfile='timeseries.mat'

# unsupervised classification scheme
classifierFile='/models/mccikpc2/DCMEX/CPI-analysis/cnn/model_t5_epochs_100_dense64_3a_freeze_final'
classifier=True
minClassSize=50.


from os import environ
environ["OMP_NUM_THREADS"]="1"
environ["OPENBLAS_NUM_THREADS"]="1"
environ["MKL_NUM_THREADS"]="1"
environ["VECLIB_MAXIMUM_THREADS"]="1"
environ["NUMEXPR_NUM_THREADS"]="1"
import gc
from multiprocessing import set_start_method, Pool, Manager
from multiprocessing.pool import ThreadPool
"""
process_roi_driver=False
process_image_stats=False
export_images=False
"""

output_timeseries=True
process_roi_driver=False
process_image_stats=False
export_images=False

def runJobs():
    global path1
    global filename1
    # get the files / path from commandline input
    if command_line_path:
        import sys
        path1=sys.argv[1]
        #path1='/Users/mccikpc2/Dropbox (The University of Manchester)/data/'
        #path1='/tmp/test/2DSCPI/'
        from os import listdir
        #from os.path import isfile, join
        filename1 = [f for f in listdir(path1) if f.endswith(".roi")]
        filename1.sort()
        del sys, listdir

    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if process_roi_driver:
        from ROIDataDriver import ROIDataDriver
        # extract ROI data from files and prcess with backgrounds
        (t_range)= \
        ROIDataDriver(path1,filename1,dt,process_sweep1_if_exist,cpiv1)
        del ROIDataDriver
        # Garbage collection:
        gc.collect()
        del gc.garbage[:]
    #--------------------------------------------------------------------------

    
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if process_image_stats:
        from imageStatsDriver import imageStatsDriver
        # find image properties, edge detection, etc
        imageStatsDriver(path1,filename1,find_particle_edges,cpiv1,num_cores)

        #del imageStatsDriver
    #--------------------------------------------------------------------------


    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if export_images:
        # export images
        from exportImagesDriver import exportImagesDriver
        exportImagesDriver(path1,filename1,foc_crit,min_len,cpiv1)
        del exportImagesDriver
    #--------------------------------------------------------------------------


    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if output_timeseries:
        if classifier==False:
            # calculate number concentrations one day at a time?
            from calcTimeseriesDriver import calcTimeseriesDriver
            calcTimeseriesDriver(path1,filename1,foc_crit,dt,ds,vel,outputfile,cpiv1)
        elif classifier==True:
            # calculate number concentrations one day at a time?
            from calcTimeseriesClassifierDriver import calcTimeseriesClassifierDriver 
        
            calcTimeseriesClassifierDriver(path1,filename1,foc_crit,dt,ds,vel,'timeseries_class.mat',\
                cpiv1,classifierFile,minClassSize)
    #--------------------------------------------------------------------------

    

if __name__ == "__main__":


    print("Setting context")
    set_start_method('spawn',force=True)

    #p=Pool(processes=num_cores, maxtasksperchild=1)
    #m = Manager()
    #l = m.Lock()
    
    
    runJobs()


