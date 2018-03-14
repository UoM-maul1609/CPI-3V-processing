function [bytes,house,images,rois,ushort,Header,I,R,H,t_range]=...
    ROIDataDriver(path,filename,dt)

FULL_BG.IMAGE=[];
FULL_BG.Time=[];
t_min=1e9;
t_max=0;
save_files=true;

disp('=========================1st sweep================================');
for i=1:length(filename)
    fid=fopen([path,filename{i}],'rb','l');

    disp(['Reading file...',filename{i}]);
    bytes=fread(fid,'uchar');
    disp('done');

    lb=length(bytes);
    % make bytes even
    if mod(lb,2) ~= 0
        bytes(end)=[];
    end

    order=[1:length(bytes)/2]'; 

    ushort1=double(typecast(uint8(bytes(1:end)),'uint16'));
    order1=order;
    ushort2=double(typecast(uint8([bytes(2:end); '1']),'uint16'));
    order2=order(2:end);

    ushort=[ushort1;ushort2];
    order=[order1;order2];


    bytes=double(typecast(uint16(ushort),'uint8'));



    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % find the positions of the start of block marks ++++++++++++++++++++++++++
    % 484B
    disp('Finding house keeping...');
    house=find(ushort==hex2dec('484B'));
    disp('done');

    % A3D5
    disp('Finding image data...');
    images=find(ushort==41941);
    disp('done');

    % B2E6
    disp('Finding roi data...');
    rois=find(ushort==45798);
    disp('done');

    fclose(fid);
    %--------------------------------------------------------------------------




    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % convert the bytes to Header, Images and ROI structs +++++++++++++++++++++
    disp('Post-processing data, stage 1...');
    Header=convertDataToHeaderStruct(ushort);
    [I,images]=convertDataToImageStruct(ushort,order,images);
    [R,rois]=convertDataToROIStruct(ushort,order,rois);
    H=convertDataToHouseStruct(ushort,order,house);
    disp('done');
    %--------------------------------------------------------------------------




    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % extract the images, times, etc ++++++++++++++++++++++++++++++++++++++++++
    disp('Post-processing data, stage 2...');
    [ROI_N,HOUSE,IMAGE1]=...
        postProcess(bytes,rois,R,H,I,Header);

    disp('done');
    %--------------------------------------------------------------------------




    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Backgrounds +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    disp('Getting backgrounds...');
    FULL_BG1=fullBackgrounds(ROI_N);
    if(length(FULL_BG1.Time))
    	FULL_BG.IMAGE=[FULL_BG.IMAGE,FULL_BG1.IMAGE];
    	FULL_BG.Time=[FULL_BG.Time,FULL_BG1.Time];
    end
    disp('done');
    %--------------------------------------------------------------------------

    
    
    if save_files
        % save to file
        disp('Saving to file...');
        save([path,strrep(filename{i},'.roi','.mat')],...
            'ROI_N', 'HOUSE', 'IMAGE1');
        save([path,'full_backgrounds.mat'],...
            'FULL_BG');
        disp('done');
    end
    t_min=min(min(ROI_N.Time),t_min);
    t_max=max(max(ROI_N.Time),t_max);
end

disp('=========================2nd sweep================================');
for i=1:length(filename)
    % load from file
    disp('Loading from file...');
    load([path,strrep(filename{i},'.roi','.mat')],...
        'ROI_N', 'HOUSE', 'IMAGE1');
    disp('done');

    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    % Assocate backgrounds ++++++++++++++++++++++++++++++++++++++++++++++++
    disp('Associate backgrounds...');
    BG=associateBackgrounds(ROI_N,FULL_BG);
    disp('done');
    %----------------------------------------------------------------------
    
    
    if save_files
        % save to file
        disp('Saving to file...');
        save([path,strrep(filename{i},'.roi','.mat')],...
            'BG','-append');
        disp('done');
    end
end

t_range=[floor(t_min*86400/dt)*dt/86400 ceil(t_max*86400/dt)*dt/86400];
