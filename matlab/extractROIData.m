function [bytes,house,images,rois,ushort,Header,I,R,H]=extractROIData(path,filename)

FULL_BG=[];

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
    FULL_BG=[FULL_BG,fullBackgrounds(ROI_N)]; % append here
    disp('done');
    %--------------------------------------------------------------------------

    
    
    % save to file
    disp('Saving to file...');
    save([path,strrep(filename{i},'.roi','.mat')],...
        'ROI_N', 'HOUSE', 'IMAGE1');
    save([path,'full_backgrounds.mat'],...
        'FULL_BG');
    disp('done');
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
    
    
    % save to file
    disp('Saving to file...');
    save([path,strrep(filename{i},'.roi','.mat')],...
        'BG', 'ROI_N', 'HOUSE', 'IMAGE1');
    disp('done');
end

