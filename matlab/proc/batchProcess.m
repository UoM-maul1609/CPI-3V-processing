% CPI MATLAB processing example.
%
% Python is now the canonical processing implementation.  This MATLAB driver
% remains useful for parity/reproduction and deliberately mirrors the Python
% stage ordering.  Edit the configuration below for a dataset.

foc_crit=5;
min_len=1;
dt=10;
ds=10;
vel=100;
find_particle_edges=true;
cpiv1=false;

process_roi_driver=false;
process_image_stats=true;
export_images=true;
output_timeseries=true;

path1='/path/to/cpi/data';
filename1={'20180109105546.roi','20180109120515.roi'};
outputfile='timeseries.mat';

if process_roi_driver
    % The legacy MATLAB raw reader is retained, but the Python raw reader has
    % CPI-v1 support and is the reference implementation for new processing.
    ROIDataDriver(path1,filename1,dt);
end

if process_image_stats
    imageStatsDriver(path1,filename1,find_particle_edges);
end

if export_images
    exportImagesDriver(path1,filename1,foc_crit,min_len);
end

if output_timeseries
    calcTimeseriesDriver(path1,filename1,foc_crit,dt,ds,vel,outputfile,cpiv1);
end
