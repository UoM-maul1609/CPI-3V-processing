foc_crit=12; % critical value of focus for an image
min_len=20; % minimum length for particle images
dt=10;  % resolution on time-step for concentrations (take with a pince of salt)
ds=10;  % resolution for size bins
vel=100;    % air speed - assumed fixed, used in calcTimeseriesDriver
find_particle_edges=false; % output the boundary of the particles

path1='/Users/mccikpc2/Dropbox (The University of Manchester)/data/';
            % path to raw data
filename1={'20180109105546-2.roi',...
    '20180109120515.roi'};
            % list of filenames to process

outputfile='timeseries.mat';
%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% extract ROI data from files and prcess with backgrounds
[bytes,house,images,rois,ushort,Header,I,R,H,t_range]=...
    ROIDataDriver(path1,filename1,dt);
%--------------------------------------------------------------------------

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% find image properties, edge detection, etc
imageStatsDriver(path1,filename1,find_particle_edges);
%--------------------------------------------------------------------------

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% export images
exportImagesDriver(path1,filename1,foc_crit,min_len);
%--------------------------------------------------------------------------

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% calculate number concentrations one day at a time?
calcTimeseriesDriver(path1,filename1,foc_crit,t_range,dt,ds,vel,outputfile);
%--------------------------------------------------------------------------
