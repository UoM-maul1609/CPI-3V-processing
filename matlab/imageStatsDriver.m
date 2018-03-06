function imageStatsDriver(path1,filename1,find_particle_edges)

disp('====================particle properties===========================');
for i=1:length(filename1)
    % load from file
    disp('Loading from file...');
    load([path1,strrep(filename1{i},'.roi','.mat')],...
        'BG','ROI_N', 'HOUSE', 'IMAGE1');
    disp('done');

    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    
    
    % Particle properties +++++++++++++++++++++++++++++++++++++++++++++++++
    disp('calculating particle properties...');
    dat=imageStats(ROI_N,BG,find_particle_edges);
    disp('done');
    %----------------------------------------------------------------------
    
    
    % save to file
    disp('Saving to file...');
    save([path1,strrep(filename1{i},'.roi','.mat')],...
        'dat','-append');
    disp('done');
end

