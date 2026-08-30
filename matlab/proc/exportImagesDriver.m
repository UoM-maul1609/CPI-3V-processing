function exportImagesDriver(path1,filename1,foc_crit,min_len)

disp('====================exporting images =============================');
load cmap;
    
% Exporting all images ++++++++++++++++++++++++++++++++++++++++++++++++++++
disp('exporting all images...');
exportImages([path1],filename1,foc_crit,min_len,MAP2);
disp('done');
%--------------------------------------------------------------------------
    

