%   **************************************************************************
%
%	M-file to preprocess CPI data: rename, extract raw data, add image stats 
%	and add time series summary for all files on one day. All subfns follow below.
%
%   NB air speed must be set correctly to properly calculate concentrations in
%   time series.
%	
%   Written 02 Feb 2021 - A.R.D. Smedley editted from PJC's routines at 
%   https://github.com/UoM-maul1609/CPI-3V-processing/tree/master/matlab,
%                         converting to v1 of CPI data format, imagetypes, 
%                         version numbers, etc
%			10 Feb 2021 : all subfns combined into one m-file [ARDS]
%           16 Mar 2021 : added correct CPI v1 background.imagetype 33795 and 
%                         added vel as fn of flowrate [ARDS]
%           29 Mar 2021 : changed to use histcounts and treat ds as bin edges
%                         if length > 1; default now set to vary [ARDS]
%
%   **************************************************************************


	function preproc_cpi(fpath,ppdt)


%   Define velocity through sample tube
    flowrate = 15;                                          % l/min
    cpidiam = 22;                                           % mm
    vel = (1000*flowrate/60)/(100*(pi/4)*(cpidiam/10)^2);   % m/s

%	Set some other defaults
	dt = 1;  	                        % time interval (seconds)
	% ds = 10; 	                        % bin size if ds is single element (microns)
    ds = [0:2:50 50:5:200 200:20:2300]; % size bin edges (microns) if length(ds) > 1
    ds = unique(ds);

	foc_crit = 20; % set as default as per Connolly et al 2007 JTech (which defines DOF)
	find_particle_edges = 'false';


%   **************************************************************************

%	Find roi files that do not start with valid year
	fileso = dir([fpath datestr(ppdt,'mmdd') '*.roi']);
	fileso = {fileso.name};

%	Print info
	fprintf('   Renaming %d CPI files to YYYYMMDD_hhmm.roi format... ',length(fileso));

%	Create new file names and rename
	filesn = cellfun(@(c)[datestr(ppdt,'yyyy') c(1:4) '_' c(5:end)],fileso,'uni',false);
	for ii = 1:length(fileso)
		movefile([fpath fileso{ii}],[fpath filesn{ii}]);
	end
	clear files*
	fprintf('done\n');

%   **************************************************************************

%	Get list of file matching specified date token
	files = dir([fpath datestr(ppdt,'yyyymmdd') '*.roi']);
	files = {files.name};

%	Extract raw cpi data to mat format, returning daily time range
	trng = cpiextractdata(fpath,files,dt);

%	Calculate individual image stats and append data to *.mat file
	cpiimagestats(fpath,files,find_particle_edges);

%	Set output filename and calculate time series summary for day
	opfile = [datestr(ppdt,'yyyymmdd') '_ts.mat'];
	cpicalctimeseries(fpath,files,foc_crit,trng,dt,ds,vel,opfile);


	end



%   **************************************************************************
%   **************************************************************************
%   **************************************************************************



%   M-file to load in raw CPI image data from file list, convert to structures,
%   extracting image, ROI and house data and save in Matlab format.
%   
%   Written 29 Jan 2021 - A.R.D. Smedley lightly editted from PJC's version at
%   https://github.com/UoM-maul1609/CPI-3V-processing/tree/master/matlab


    function trng = cpiextractdata(fpath,fnames,dt)

%   Initialise variables
    FULL_BG.IMAGE = [];
    FULL_BG.Time = [];
    t_min = 1e9;
    t_max = 0;
    save_files = true;

%   Print info
    fprintf('   =========================1st sweep================================\n');

%   For each file
    for i = 1:length(fnames)

%       Read in data
        fid = fopen([fpath,fnames{i}],'rb','l');
        fprintf('   Reading file: %s... ',fnames{i});
        bytes=fread(fid,'uchar');
        fprintf('done\n');
        fclose(fid);

%       Make bytes even
        lb = length(bytes);
        if mod(lb,2) ~= 0
            bytes(end) = [];
        end

%       Format bytes
        order = [1:length(bytes)/2]'; 
        ushort1 = double(typecast(uint8(bytes(1:end)),'uint16'));
        order1 = order;
        ushort2 = double(typecast(uint8([bytes(2:end); '1']),'uint16'));
        order2 = order(2:end);
        ushort = [ushort1;ushort2];
        order = [order1;order2];
        bytes = double(typecast(uint16(ushort),'uint8'));

%       Find the positions of the start of house keeping block marks [484B]
        fprintf('   Finding house keeping... ');
        house = find(ushort==hex2dec('484B'));
        fprintf('done\n');

%       Find the positions of the start of image block marks [A3D5]
        fprintf('   Finding image data... ');
        images = find(ushort==41941);
        fprintf('done\n');

%       Find the positions of the start of ROI block marks [B2E6]
        fprintf('   Finding roi data... ');
        rois = find(ushort==45798);
        fprintf('done\n');

%       Convert the bytes to Header, Images and ROI structures
        fprintf('   Post-processing data, stage 1... ');
        Header = convertDataToHeaderStruct(ushort);
        [I,images] = convertDataToImageStruct(ushort,order,images);
        [R,rois] = convertDataToROIStruct(ushort,order,rois);
        H = convertDataToHouseStruct(ushort,order,house);
        fprintf('done\n');

%       Extract the images, times, etc 
        fprintf('   Post-processing data, stage 2... ');
        [ROI_N,HOUSE,IMAGE1] = postProcess(bytes,rois,R,H,I,Header);
        fprintf('done\n');

%       Get backgrounds
        fprintf('   Getting backgrounds... ');
        FULL_BG1 = fullBackgrounds(ROI_N); 
        if(length(FULL_BG1.Time))
    	   FULL_BG.IMAGE = [FULL_BG.IMAGE,FULL_BG1.IMAGE];
    	   FULL_BG.Time = [FULL_BG.Time;FULL_BG1.Time];
        end
        fprintf('done\n');

%       Save to file and add ROI times variables
        if save_files
            fprintf('   Saving to file: %s... ',strrep(fnames{i},'.roi','.mat'));
            ROI_N_Time = ROI_N.Time;
            save([fpath,strrep(fnames{i},'.roi','.mat')],'ROI_N', 'ROI_N_Time', 'HOUSE', 'IMAGE1');
            fprintf('done\n');
        end
        t_min=min(min(ROI_N.Time),t_min);
        t_max=max(max(ROI_N.Time),t_max);
    end


%   Print info
    fprintf('   =========================2nd sweep================================\n');

%   For each file
    for i=1:length(fnames)

%       Load from file
        fprintf('   Loading from file: %s... ',strrep(fnames{i},'.roi','.mat'));
        load([fpath,strrep(fnames{i},'.roi','.mat')],'ROI_N', 'HOUSE', 'IMAGE1');
        fprintf('done\n');

%       Associate backgrounds
        fprintf('   Associate backgrounds... ');
        BG = associateBackgrounds(ROI_N,FULL_BG);
        fprintf('done\n');
    
%       Save to appended file
        if save_files
            fprintf('   Saving backgrounds to file : %s... ',strrep(fnames{i},'.roi','.mat'));
            save([fpath,strrep(fnames{i},'.roi','.mat')],'BG','-append');
            fprintf('done\n');
        end
    end
    trng = [floor(t_min*86400/dt)*dt/86400 ceil(t_max*86400/dt)*dt/86400];

    end


%   **************************************************************************
%   **************************************************************************


%   Function to convert to header structure
    function Header=convertDataToHeaderStruct(ushort)

    Header=struct('ucVersion',1, ... 'ushort'), 
                  'usYear',1, ... 'ushort'), ...
                  'ucMonth',1, ... 'ushort'),...
                  'ImageX',1, ... 'ushort'), ...
                  'ImageY',1 , ...'ushort'), ...
                  'szInfo',zeros(70,1));

    Header.ucVersion=ushort(1);
    Header.usYear=ushort(2);
    Header.ucMonth=ushort(3);
    Header.ImageX=ushort(4);
    Header.ImageY=ushort(5);
    Header.szInfo=ushort([6:6+70-1]);

    end


% **************************************************************************
% **************************************************************************


%   Function to convert to image structure
    function [I,images]=convertDataToImageStruct(bytes,order,images)

    I = struct( 'ulItemSize',1, ... 'ulong' ...
                'usVersion',1, ...'ushort' ...
                'usROIsCount',1, ...'ushort' ...
                'ulTotROISize',1, ...'ulong' ...
                'day',1, ...'uchar' ...
                'hour',1, ...'uchar' ...
                'minute',1, ...'uchar' ...
                'second',1, ...'uchar'  ...
                'msecond',1, ...'ushort' ...
                'ImageType',1, ...'ushort'  ...
                'StartX',1, ...'ushort'  ...
                'StartY',1, ...'ushort' ...
                'EndX',1, ...'ushort'  ...
                'EndY',1, ...'ushort'  ...
                'BGRate',1, ...'ushort' ...
                'usBkgPDSThresh',1, ...'ushort' ...
                'ulFrmsProc',1, ...'ulong'  ...
                'IThreshold',1, ...'uchar' ...
                'ROIError',1, ...'uchar'  ...
                'ROIMinSize',1, ...'ushort'  ...
                'ROIAspectRatio',1, ...'float' ...
                'ROIFillRatio',1, ...'float' ...
                'ROIFCount',1, ...'ulong'  ...
                'ucImgMean',1, ...'uchar' ...
                'ucBkgMean',1, ...'uchar'  ...
                'Spare1',1, ...'short'  ...
                'ROIXPad',1, ...'ushort' ...
                'ROIYPad',1, ...'ushort'  ...
                'ulStrobeCount',1, ...'ulong'  ...
                'ulFrmsSaved',1, ...'ulong' ...
                'ImgMinVal',1, ...'uchar'  ...
                'ImgMaxVal',1, ...'uchar'  ...
                'ulROIsSaved',1, ...'ulong' ...
                'usPDSChkSumFlag',1, ...'ushort'  ...
                'usPDSHead',[1 1 1], ...'ushort'  ...
                'ulTime',1, ...'ulong' ...
                'ArrivalTime1',1, ...'ushort'  ...
                'ArrivalTime2',1, ...'ushort'  ...
                'TransitTime',1, ...'ushort' ...
                'usStrobes',1, ...'ushort'  ...
                'PulseInt45',1, ...'ushort'  ...
                'PulseInt90',1, ...'ushort' ...
                'PDSChkSum',1, ...'ushort' ...
                'ProbeMode',1,...); ...'ushort' ;
                'order',1);

%   Replicate image matric
    I = repmat(I,[length(images) 1]); 

%   For each image produce structure I from raw bytes
    for i=1:length(images)
        I(i).ulItemSize = double(typecast(uint16(bytes(images(i)+[1 2])),'uint32'));
        I(i).usVersion = bytes(images(i)+3);
        if(I(i).usVersion == 25) % was originally 40
            I(i).usROIsCount = bytes(images(i)+4);
            I(i).ulTotROISize = double(typecast(uint16(bytes(images(i)+[5 6])),'uint32'));

            dd = double(typecast(uint16(bytes(images(i)+[7])),'uint8'));
            I(i).day = dd(1);
            I(i).hour = dd(2);

            dd = double(typecast(uint16(bytes(images(i)+[8])),'uint8'));
            I(i).minute = dd(1);
            I(i).second = dd(2);

            I(i).msecond = bytes(images(i)+9);
            I(i).ImageType = bytes(images(i)+10);

            I(i).StartX = bytes(images(i)+11);
            I(i).StartY = bytes(images(i)+12);
            I(i).EndX = bytes(images(i)+13);
            I(i).EndY = bytes(images(i)+14);
            I(i).BGRate = bytes(images(i)+15);
            I(i).usBkgPDSThresh = bytes(images(i)+16);

            I(i).ulFrmsProc = double(typecast(uint16(bytes(images(i)+[17 18])),'uint32'));

            dd = double(typecast(uint16(bytes(images(i)+[19])),'uint8'));
            I(i).IThreshold = dd(1);
            I(i).ROIError = dd(2);

            I(i).ROIMinSize = bytes(images(i)+20);

            I(i).ROIAspectRatio = double(typecast(uint16(bytes(images(i)+[21 22])),'single'));
            I(i).ROIFillRatio = double(typecast(uint16(bytes(images(i)+[23 24])),'single'));

            I(i).ROIFCount = double(typecast(uint16(bytes(images(i)+[25 26])),'uint32'));

            dd = double(typecast(uint16(bytes(images(i)+[27])),'uint8'));
            I(i).ucImgMean = dd(1);
            I(i).ucBkgMean = dd(2);

            I(i).Spare1 = bytes(images(i)+28);
            I(i).ROIXPad = bytes(images(i)+29);
            I(i).ROIYPad = bytes(images(i)+30);

            I(i).ulStrobeCount = double(typecast(uint16(bytes(images(i)+[31 32])),'uint32'));
            I(i).ulFrmsSaved = double(typecast(uint16(bytes(images(i)+[33 34])),'uint32'));

            dd = double(typecast(uint16(bytes(images(i)+[35])),'uint8'));
            I(i).ImgMinVal = dd(1);
            I(i).ImgMaxVal = dd(2);

            I(i).ulROIsSaved = double(typecast(uint16(bytes(images(i)+[36 37])),'uint32'));

            I(i).usPDSChkSumFlag = bytes(images(i)+38);
            I(i).usPDSHead = bytes(images(i)+[39 40 41]);

            I(i).ulTime = double(typecast(uint16(bytes(images(i)+[42 43])),'uint32'));
            I(i).ArrivalTime1 = bytes(images(i)+44);
            I(i).ArrivalTime2 = bytes(images(i)+45);

%           Note, goes a bit funny here, because idl code says time is ushort and starts at 42 (84/2)
            I(i).ulTime = double(typecast(uint16(bytes(images(i)+[42])),'uint16'));
            I(i).ArrivalTime1 = bytes(images(i)+43);
            I(i).ArrivalTime2 = bytes(images(i)+44);

            I(i).TransitTime = bytes(images(i)+46);
            I(i).usStrobes = bytes(images(i)+47);
            I(i).PulseInt45 = bytes(images(i)+48);
            I(i).PulseInt90 = bytes(images(i)+49);
            I(i).PDSChkSum = bytes(images(i)+50);
            I(i).ProbeMode = bytes(images(i)+51);
            I(i).order = order(images(i));
        end    
    end

%   Get version and use to filter for valid images
    ver = cat(1,I.usVersion);
    ind = find(ver==25); % Was originally 40 for CVI-3V processing (ARDS)
    I = I(ind);
    images = images(ind);

    end


%   **************************************************************************
%   **************************************************************************


%   Function to convert all ROIs to structures
    function [R,rois] = convertDataToROIStruct(bytes,order,rois)

    R = struct( 'ulItemSize',1, ...'ulong' 
                'usVersion',1, ...'ushort' ...
                'StartX',1, ...'ushort' 
                'StartY',1, ...'ushort' 
                'EndX',1, ...'ushort' ...
                'EndY',1, ...'ushort'
                'PixBytes',1, ...'short' 
                'usROIFlags',1, ...'ushort' ...
                'fLength',1, ...'float'
                'ulStartLen',1, ...'ulong' 
                'ulEndLen',1, ...'ulong' ...
                'fWidth',1,  ...'float'));...
                'Spare',zeros(18,1),...); ... 'char'
                'order',1);

%   Replicate the roi matrix
    R = repmat(R,[length(rois) 1]);

%   For each roi produce structure R from raw bytes
    for i = 1:length(rois)
        R(i).ulItemSize = double(typecast(uint16(bytes(rois(i)+[1 2])),'uint32'));
        R(i).usVersion = bytes(rois(i)+3);
        if(R(i).usVersion==25)
            R(i).StartX = bytes(rois(i)+4);
            R(i).StartY = bytes(rois(i)+5);
            R(i).EndX = bytes(rois(i)+6);
            R(i).EndY = bytes(rois(i)+7);
            R(i).PixBytes = double(typecast(uint16(bytes(rois(i)+8)),'int16'));;
            R(i).usROIFlags = bytes(rois(i)+9);

            R(i).fLength = double(typecast(uint16(bytes(rois(i)+[10 11])),'single'));
            R(i).ulStartLen = double(typecast(uint16(bytes(rois(i)+[12 13])),'uint32'));
            R(i).ulEndLen = double(typecast(uint16(bytes(rois(i)+[14 15])),'uint32'));

            R(i).fWidth = double(typecast(uint16(bytes(rois(i)+[16 17])),'single'));
            R(i).Spare = double(typecast(uint16(bytes(rois(i)+[18:18+9-1])),'int8'));
            R(i).order = order(rois(i));
        end    
    end

%   Get version and use to filter for valid rois
    ver = cat(1,R.usVersion);
    ind = find(ver==25);
    R = R(ind);
    rois = rois(ind);

    end


% **************************************************************************
% **************************************************************************


%   Function to convert all house to structures
    function H = convertDataToHouseStruct(bytes,order,house)

    H = struct( 'BlockNum',1, ...'ushort' 
                'ulItemSize',1, ...'ushort' ...
                'Readings1',zeros(70,1), ...'ushort' 
                'TimeMSW',1, ...'ushort' 
                'TimeISW',1, ...'ushort' ...
                'TimeLSW',1, ...'ushort'
                'Readings',zeros(8,1), ...); ... 'ushort'
                'order',1);

%   Replicate the house matrix
    H = repmat(H,[length(house) 1]);

%   For each house produce structure H from raw bytes
    for i = 1:length(house)
        if bytes(house(i)+1)== 83
            H(i).BlockNum = bytes(house(i)+0);
            H(i).ulItemSize = bytes(house(i)+1);
            H(i).Readings1 = bytes(house(i)+[2:2+70-1]); % 71
            H(i).TimeMSW = bytes(house(i)+72);
            H(i).TimeISW = bytes(house(i)+73);
            H(i).TimeLSW = bytes(house(i)+74);
            H(i).Readings = bytes(house(i)+[75:75+8-1]);
            H(i).order = order(house(i));
        end
    end

%   Get ulItemSize and use to filter for valid H
    ulItemSize = cat(1,H.ulItemSize);
    ind = find(ulItemSize==1); %  was originally 83 for CPI-3V (ARDS)
    H = H(ind);

    end


%   **************************************************************************
%   **************************************************************************

%   Function to postprocess R, H, I to ROI_N, HOUSE, IMAGES
    function [ROI_N,HOUSE,IMAGE1] = postProcess(bytes,rois,R,H,I,Header)

%   Get length of bytes
    lb = length(bytes);

%   Image times in hk format
    IMAGE1.Time = cat(1,I.ArrivalTime2).*16.^8+...
                  cat(1,I.ArrivalTime1).*16.^4+cat(1,I.ulTime);

%   Time in MATLAB format, using msecond, etc
    Time = datenum(Header.usYear,Header.ucMonth,...
                   cat(1,I.day),...
                   cat(1,I.hour),...
                   cat(1,I.minute),...
                   cat(1,I.second)) ....
                  +cat(1,I.msecond)./(1000)./24./3600;
    str1 = strcat(num2str(Time.*86400-floor(Time.*86400),'%0.3f'));
    str1(:,1) = '.';
    Timestr = [datestr(Time),str1(:,2:end)];

%   Image type
    imageType = cat(1,I.ImageType);
    IMAGE1.Time1 = Time;
    IMAGE1.Timestr = Timestr;
    IMAGE1.imageType = imageType;


%   Housekeeping of packet times
    HOUSE.Time = (cat(1,H.TimeMSW).*16.^8+ ...
                  cat(1,H.TimeISW).*16.^4+...
                  cat(1,H.TimeLSW));
    Rdgs = cat(2,H.Readings1);
    HOUSE.deadtime = Rdgs(67,:)*0.000341333;

%   Map 'IMAGE' to ROI
    orderImage = cat(1,I.order);
    orderROIs = cat(1,R.order);
    timeROIs = zeros(length(R),1);
    imageTypeROIs = zeros(length(R),1);

    for i = 1:length(orderImage)-1
        ind = find(orderImage(i)<orderROIs & orderImage(i+1)>orderROIs );
        timeROIs(ind) = Time(i);
        imageTypeROIs(ind) = imageType(i);
    end

    i = length(orderImage);
    ind = find(orderImage(i)<orderROIs );
    timeROIs(ind) = Time(i);
    imageTypeROIs(ind) = imageType(i);

%   Associate variables with each ROI
    [a,b] = sort(timeROIs);
    for i = 1:length(rois)
        StartX = R(b(i)).StartX;
        StartY = R(b(i)).StartY;
        EndX = R(b(i)).EndX;
        EndY = R(b(i)).EndY;
    
        X = (EndX-StartX+1);
        Y = (EndY-StartY+1);
        numberOfChars = X*Y;

        chars1 = bytes(rois(b(i))*2+33*2-14+[1:numberOfChars]);
        ROI_N.IMAGE(i).IM = reshape(chars1,[X,Y]);
    end

    ROI_N.Time = timeROIs;
    ROI_N.imageType = imageTypeROIs;
    ROI_N.StartX = cat(1,R.StartX);
    ROI_N.StartY = cat(1,R.StartY);
    ROI_N.EndX = cat(1,R.EndX);
    ROI_N.EndY = cat(1,R.EndY);

%   Sort ROIs
    [a,b] = sort(ROI_N.Time);
    ROI_N.Time(:) = ROI_N.Time(b);
    ROI_N.imageType(:) = ROI_N.imageType(b);
    ROI_N.StartX(:) = ROI_N.StartX(b);
    ROI_N.StartY(:) = ROI_N.StartY(b);
    ROI_N.EndX(:) = ROI_N.EndX(b);
    ROI_N.EndY(:) = ROI_N.EndY(b);

%   Sort HOUSE
    [a,b] = sort(HOUSE.Time);
    HOUSE.Time(:) = HOUSE.Time(b);
    HOUSE.deadtime(:) = HOUSE.deadtime(b);

%   Sort IMAGES
    [a,b] = sort(IMAGE1.Time);
    IMAGE1.Time(:) = IMAGE1.Time(b);
    IMAGE1.Time1(:) = IMAGE1.Time1(b);
    IMAGE1.Timestr(:,:) = IMAGE1.Timestr(b,:);
    IMAGE1.imageType(:) = IMAGE1.imageType(b);

    end


%   **************************************************************************
%   **************************************************************************

%   Function to extract backgrounds
    function FULL_BG = fullBackgrounds(ROI_N)

%   Search for backgrounds and return if none found
    ind = find(ROI_N.imageType==33795);  % Was originally 19 for CVI-3V processing (ARDS)
    if(length(ind) == 0)
        fprintf('no backgrounds found... ');
        FULL_BG.IMAGE = [];
        FULL_BG.Time = [];
        return;
    end
    BG = ROI_N.IMAGE(ind);

%   Rename elements of array of struct
    f = fieldnames(BG);
    v = struct2cell(BG);
    f{strmatch('IM',f,'exact')} = 'BG';
    BG = cell2struct(v,f);
    FULL_BG.IMAGE = BG;
    FULL_BG.Time = ROI_N.Time(ind);

    end


%   **************************************************************************
%   **************************************************************************


%   Function to associate backgrounds with each ROI
    function [BG] = associateBackgrounds(ROI_N,FULL_BG)

%   Alert (but do not return if no backgrounds; instead use median later)
    if isempty(FULL_BG.IMAGE)
        fprintf('no backgrounds found... ');
    end

%   Rename elements of array of struct to produce BG structure
    BG = ROI_N.IMAGE;
    f = fieldnames(BG);
    v = struct2cell(BG);
    f{strmatch('IM',f,'exact')} = 'BG';
    BG = cell2struct(v,f);

%   For each ROI associate nearest background
    ind = 1:length(FULL_BG.Time);
    for i = 1:length(ROI_N.IMAGE)
        if length(ind)==1
            jj=ind;
        end
        if length(ind) > 1
            Timecur = ROI_N.Time(i);
            jj = interp1(FULL_BG.Time,ind,Timecur,'nearest','extrap');
        end

%       If no background available, use median value of image (ARDS)    
        if ~isempty(FULL_BG.IMAGE)
            BG(i).BG(:,:) = FULL_BG.IMAGE(jj).BG(ROI_N.StartX(i)+1:ROI_N.EndX(i)+1,...
                            ROI_N.StartY(i)+1:ROI_N.EndY(i)+1);
        else
            BG(i).BG(:,:) = median(ROI_N.IMAGE(i).IM(:));
        end
    end

    end


%   **************************************************************************
%   **************************************************************************
%   **************************************************************************


%   M-file to load in CPI image data from file list, calculate image stats and save
%   
%   Written 29 Jan 2021 - A.R.D. Smedley lightly editted from PJC's version at
%   https://github.com/UoM-maul1609/CPI-3V-processing/tree/master/matlab


    function cpiimagestats(fpath,fnames,find_particle_edges)

%   Print info
    fprintf('   =====================image statistics=============================\n');

%   For each file
    for i=1:length(fnames)
    
%       Load data from file
        fprintf('   Loading from file: %s... ',strrep(fnames{i},'.roi','.mat'));
        load([fpath,strrep(fnames{i},'.roi','.mat')],'BG','ROI_N', 'HOUSE', 'IMAGE1');
        fprintf('done\n');
    
%       Calculate particle properties
        fprintf('   Calculating particle properties...   0.0%%');
        STATS = imagestats(ROI_N,BG,find_particle_edges);
        fprintf('\b\b\b\b\b\bdone\n');    
    
%       Save results to appended file
        fprintf('   Saving STATS to file: %s... ',strrep(fnames{i},'.roi','.mat'));
        save([fpath,strrep(fnames{i},'.roi','.mat')],'STATS','-append');
        fprintf('done\n');
    end

    end


%   **************************************************************************
%   **************************************************************************


%   Core function to handle analysis of file's worth of data
    function STATS = imagestats(ROI_N,BG,b_flag)

%   Turn off warnings
    warning off;

%   Find particle images
    ind=find(ROI_N.imageType==33857); % originally was 89 for CPI-3V [ARDS]

%   Initialise arrays
    l1=length(ROI_N.IMAGE);
    STATS.len=zeros(l1,1);
    STATS.wid=zeros(l1,1);
    STATS.area=zeros(l1,1);
    STATS.round=zeros(l1,1);
    STATS.centroid=zeros(l1,2);
    if ~b_flag
        STATS.foc = repmat(struct('focus',NaN),[l1 1]);
    else
        ld=3;
        STATS.foc = repmat(struct('focus',NaN, ...
                  'boundaries',NaN.*zeros(30,2)),[l1 1]);
    end
    STATS.Time=ROI_N.Time;
    ntot = length(ind);
    ncur = 0;

%   For each valid image
    for i=ind'

%       Print info
        ncur = ncur+1;
        fprintf('\b\b\b\b\b\b%5.1f%%',100*ncur/ntot);

%       Subtract background, check for particle size, then subtract minimum value
        arr=ROI_N.IMAGE(i).IM-BG(i).BG;
        inda=find(double(arr(:))<=-15);
        indb=find(double(arr(:))>-15);
        if(length(inda)<=3), continue;, end
        arr=arr-min(arr(:));

%       Get image array size
        [r,c]=size(arr);

%       Detect edges using Sobel's method
        % bw1=edge(arr,'sobel');

%       Determine threshold level using Otsu's method, originally manually set
        BW = imbinarize(arr./256);

%       Apply median 5x5 filter to remove spikes and force edges to 1.
        BW2=medfilt2(BW,[5 5]);
        BW2(1,:)=1; BW2(end,:)=1;    
        BW2(:,1)=1; BW2(:,end)=1;

%       Find boundary
        [boundary,L,N,A] = bwboundaries(BW2);
        [B1,I1,J1]=unique(L(:));

%       If no particle found, then skip rest of loop
        if length(B1)<2
            continue;
        end

%       Find index for largest particle image
        len2=0.;
        k1=1;
        for k=2:length(B1)
            len2=max(length(find(L(:)==B1(k))),len2); %number of points in this roi
            if(length(find(L(:)==B1(k)))==len2) % saves the largest one that isn't the first
                k1=k;
            end        
        end

%       Plot images and boundary
        % imagesc(arr);
        % hold on
        % plot(boundary{k1}(:,2), boundary{k1}(:,1), 'b', 'LineWidth', 2)
    
%       Find pixels within polygon (PIP) and do stats on region
        [rows columns] = size(arr); 
        PIP = poly2mask(boundary{k1}(:,1),boundary{k1}(:,2), columns, rows);
        RPROPS = regionprops(double(PIP),'FilledArea','Centroid','MajorAxisLength','MinorAxisLength','Perimeter');

%       If stats is not empty, add to dat structure, scaling by CPI pixel size
        if ~length(RPROPS)
            continue;
        end
        STATS.len(i)=RPROPS.MajorAxisLength.*2.3;
        STATS.wid(i)=RPROPS.MinorAxisLength.*2.3;
        STATS.area(i)=RPROPS.FilledArea.*2.3.^2;
        STATS.round(i)=RPROPS.FilledArea./(pi./4.*RPROPS.MajorAxisLength.^2);
        STATS.centroid(i,:)=RPROPS.Centroid;
        boundaries=boundary{k1};

%       Calculate focus and add to STATS array
        foc=calculate_focus(boundaries,arr,b_flag); % maybe focus less than 12?
        STATS.foc(i)=foc;

    end

    end


%   **************************************************************************
%   **************************************************************************

%   Function to calculate "focus" value (seems akin to a depth of field measure).
    function foc = calculate_focus(boundaries,IM,b_flag)

%   Make these persistent
    persistent boundaries2 xs ys focus r c;

%   Initialise some variables
    ld=3;
    ind1=linspace(1,30,length(boundaries(:,1)));
    ind2=linspace(1,ind1(end),30);
    if isempty(boundaries2) 
        boundaries2=zeros(length(ind2),2); 
        [r,c]=size(boundaries2);
        xs=zeros(ld,r-1);
        ys=zeros(ld,r-1);
        focus=zeros(r-1,1);
    end

%   Calculate boundaries and midpoints
    if(length(boundaries(:,1))>9)
        boundaries2(:,1)=interp1(ind1,filtfilt(ones(1,3),3,boundaries(:,1)),ind2);
        boundaries2(:,2)=interp1(ind1,filtfilt(ones(1,3),3,boundaries(:,2)),ind2);
    else
        boundaries2(:,1)=interp1(ind1,boundaries(:,1),ind2);
        boundaries2(:,2)=interp1(ind1,boundaries(:,2),ind2);
    end
    midpointx=(boundaries2(2:end,1)+boundaries2(1:end-1,1))./2;
    midpointy=(boundaries2(2:end,2)+boundaries2(1:end-1,2))./2;

%   Calculate gradient
    grad=(boundaries2(2:end,2)-boundaries2(1:end-1,2))./(boundaries2(2:end,1)-boundaries2(1:end-1,1));
    grad=-1./(grad);

%   Find start and end point of the lines to calculate the gradients along
    dl=-2; % length 2 pixels
    dx1=dl./sqrt(1+grad.^2);
    dy1=dl./sqrt(1+1./grad.^2);

    dl=2; % length 2 pixels
    dx2=dl./sqrt(1+grad.^2);
    dy2=dl./sqrt(1+1./grad.^2);

    deltax=(boundaries2(2:end,1)-boundaries2(1:end-1,1));
    deltay=(boundaries2(2:end,2)-boundaries2(1:end-1,2));

    for i=1:r-1
        xs(:,i)=linspace(midpointx(i)+dx1(i),midpointx(i)+dx2(i),ld);
        ys(:,i)=linspace(midpointy(i)-dy1(i),midpointy(i)-dy2(i),ld);
    end

%   Correct for direction of gradient
    ind=find(deltay<0 & deltax>0);
    for i=ind'
        ys(:,i)=linspace(midpointy(i)+dy1(i),midpointy(i)+dy2(i),ld);
    end
    ind=find(deltay>0 & deltax<0);
    for i=ind'   
        ys(:,i)=linspace(midpointy(i)+dy1(i),midpointy(i)+dy2(i),ld);
    end

%   Now calculate the focus
    [rr,cc]=size(IM);
    [rr,cc]=meshgrid(1:rr,1:cc);
    try
        for i=1:r-1
            intensity=interp2(rr,cc,IM',xs(:,i),ys(:,i));
            focus(i)=nanmean(abs(parabolderiv(linspace(-2,2,ld)',intensity,1,0)));
        end
    catch
        focus=NaN;
    end
    foc.focus=nanmean(focus);

%   And if requested save the boundaries
    if b_flag
        foc.boundaries=boundaries2;
    end

    end


%   **************************************************************************
%   **************************************************************************


%   Function to calculatw parabolic derivative (for focus determination)
    function dy_dx = parabolderiv( x, y, k, graphics_flag )

%   function dy_dx = parabolderiv( x, y, k, graphics_flag )
%
%   default: example from Ref. [1], p. 322
%
%   or try: parabolderiv( ( -2 : 0.01 : 2 )', exp( -2 : 0.01 : 2 )', 2 )
%           parabolderiv( ( -2 : 0.01 : 2 )', ( exp( -2 : 0.01 : 2 ) + 0.01 * rand( 1, 401 ) )', 25 )
%           parabolderiv( ( ( 0 : 200 * pi ) / 100 )', ( sin( 0 : 0.01 : 2 * pi ) + 0.01 * rand( 1, 629 ) )', 40 )
%           for demonstration of noisy input and edge deviations
%
%   graphics_flag = 1: show result in graph (default) 
%                      (includes comparison to 2 other methods)
%                   0: show no graph
%
%   This program differentiates an empirical function (an array of evenly
%   spaced values), for instance periodically sampled data from an
%   experiment.
%
%   Method used: basically, a parabola is fit from k points to the left to k
%   points to the right of the point where the derivative is required, this
%   is then analytically differentiated. These calculations are not actually
%   performed, but the method as given by Lanczos in Ref. [1] is used: only
%   1 fit parameter of the parabola is needed, which is calculated directly
%   from the data; at the edges a parabola is fit through the first (last)
%   2k points, from which the derivative is calculated directly. Additional
%   information (on accuracy, &c.) in Ref. [2]. Noise is handled well. An 
%   example from Ref. [1] is included.
%
%   The datapoints need to be equidistant. If the graphics_flag is set, the
%   result of this method is compared to applying the standard Matlab diff()
%   function on the raw data as well as on adjacent averaged filtered data
%   (zero phase delay). The example commands given above illustrate this.
%
%   References:
%
%   [1] Title                    : Applied analysis / by Cornelius Lanczos
%       Author                   : Cornelius Lanczos
%       Edition                  : 3rd print.
%       Publisher                : London : Pitman, 1964
%       Pages                    : 539 p.
%       Bibliographic annotation : 1st print: 1957
%       see pp. 321 - 324
%
%   [2] Title                    : Digital filters / by Richard W. Hamming
%       Author                   : Richard W. Hamming
%       Edition                  : 3rd ed.
%       Publisher                : Englewoood Cliffs : Prentice-Hall, 1989
%       Pages                    : XIV, 284 p.
%       Bibliographic annotation : 1st print: 1977
%       ISBN                     : 0-13-212812-8
%       see p. 137
%
%   Last update 24-03-2004 by Robert Klein-Douwel
% 
%   mail: robertkdkd@yahoo.co.uk, R.J.H.Klein-Douwel@tue.nl
%   web:  http://www.sci.kun.nl/mlf/robertkd/

	linewidth = 1; % can be changed for plotting purposes
	% fill in some default values
	if nargin < 4
	    graphics_flag = 1;
	end
	if nargin < 3 % take example from Ref. [1], p. 322
	    y = [ 0
	          4
	         25
	         50
	         67.4
	        124.9
	        172.0
	        201.4
	        288.1
	        321.3
	        387.1 ];
	    x = ( 0 : 1 : length( y ) - 1 )';
	    k = 2;
	end
	% initialise stuff
	if length( x ) ~= length( y )
	    error( 'x and y vectors have different lengths (RKD)' );
	end
	n = length( y ); %number of points
	if 2 * k + 1 > n
	    error( 'k too large or too few datapoints (RKD)' );
	end
	dy = zeros( n, 1 );
	% check equidistancy of x
	dx = diff( x );
	d2x = diff( dx );
	rel_error_d2x = max( abs( d2x ) ) / min( dx );
	if rel_error_d2x > 401 * eps
	    % vectors like ( -2 : 0.01 : 2 ) are not equidistant, but have a small variation 
	    % in dx and d2x; if this is not more than 401 * eps, then it is neglected
	    disp( [ 'relative error in d2x = ' num2str( rel_error_d2x ) ] );
	    disp( [ 'relative error / eps = ' num2str( rel_error_d2x / eps ) ] );
	    disp( [ 'min( d2x ) = ' num2str( min( d2x ) ) ] );
	    disp( [ 'max( d2x ) = ' num2str( max( d2x ) ) ] );
	    error( 'non-equidistant data points (RKD)' );
	end
	h = dx( 1 ); % stepsize
	% start calculations
	dy_denominator = 2 * h * sum( ( 1 : k ).^2 );
	a = - k : + k;
	for i = k + 1 : n - k
	    dy( i ) = sum( a .* y( i + a )' );
	end
	dy = dy / dy_denominator;
	% first and last k points:
	% fit parabola over first and last 2k points
	kk = 2 * k;
	% first kk points
	p1 = polyfit( x( 1 : kk ), y( 1 : kk ), 2 );
	q1 = polyder( p1 ); % take derivative
	dy( 1 : k ) = polyval( q1, x( 1 : k ) );
	% last kk points
	p2 = polyfit( x( n - kk + 1 : n ), y( n - kk + 1 : n ), 2 );
	q2 = polyder( p2 ); % take derivative
	dy( n - k + 1 : n ) = polyval( q2, x( n - k + 1 : n ) );
	if nargout ~= 0 % output
	    dy_dx = dy;
	end
	% end of calculations (everything below is just for making comparisons and nice output)
	if graphics_flag == 1 % make some comparisons
	    xdiff_range = x( 1 : n - 1 ) + h / 2; % x coordinates for results of diff() function
	    % compare with diff()
	    dydx_matlab = diff( y ) / h;
	    if nargin >=3 % not enough data points in example to apply this filter
	        % another test: try first adjacent averaging and then diff()
	        % use k points to left and to right
	        kkk = 2 * k + 1;
	        b = ones( 1, kkk ) / kkk; % k point averaging filter
	        y_filt = filtfilt( b, 1, y ); % noncausal filtering, zero phase distortion
	        % calculate dy_filt/dx
	        dy_filtdx = diff( y_filt ) / h;
	    end
	    if nargin < 3 % show results of example
	        disp( '      x         y         dy' );
	        disp( [ x, y, dy ] );
	    end
	end

	% show results
	if graphics_flag == 1
	    if nargout == 0
	        close all;
	    end
	    dx_idx_range = k + 1 : n - k;
	    % parabolic fits to first (last) kk points
	    x1_fit = min( x( 1 : kk ) ) : h / 100 : max( x( 1 : kk ) );
	    y1_fit = polyval( p1, x1_fit );
	    x2_fit = min( x( n - kk + 1 : n ) ) : h / 100 : max( x( n - kk + 1 : n ) );
	    y2_fit = polyval( p2, x2_fit );
	    figure;
	    hold on;
	    plot( x, y, '-ob', 'LineWidth', linewidth );
	    plot( x( dx_idx_range ), dy( dx_idx_range ), '-or', 'LineWidth', linewidth );
	    plot( x( 1 : k ), dy( 1 : k ), '-sr', 'LineWidth', linewidth );
	    plot( x1_fit, y1_fit, '--r', 'LineWidth', linewidth );
	    plot( xdiff_range, dydx_matlab, ':sg', 'LineWidth', linewidth );
	    if nargin >= 3
	        plot( xdiff_range, dy_filtdx, ':dm', 'LineWidth', linewidth );
	    end
	    legend( 'y', [ 'dy/dx_{Lanczos: {\pm}' num2str( k ) '}' ], ...
	        [ 'first/last ' num2str( k ) ' pts' ], ...
	        [ 'y_{fit, parabola, ' num2str( kk ) ' pts}' ], 'diff( y )/ dx', ...
	        [ 'diff( y_{adj avg ({\pm}' num2str( k ) ')} )/ dx' ], 0 );
	    plot( x( n - k + 1 : n ), dy( n - k + 1 : n ), '-sr', 'LineWidth', linewidth );
	    plot( x2_fit, y2_fit, '--r', 'LineWidth', linewidth );
	    plot( x, dy, '-r', 'LineWidth', linewidth );
	    % replot to put these on top again
	    plot( x( dx_idx_range ), dy( dx_idx_range ), '-or', 'LineWidth', linewidth );
	    plot( x( 1 : k ), dy( 1 : k ), '-sr', 'LineWidth', linewidth );
	    hold off;
	    xlabel( 'x' );
	    ylabel( 'y' );
	    title( 'parabolic derivative (Lanczos)' );
	end

	end



%   **************************************************************************
%   **************************************************************************
%   **************************************************************************



%   M-file to calculate a time series for CPI images, and save to mat
%   
%   Written 01 Feb 2021 - A.R.D. Smedley lightly editted from PJC's version at
%   https://github.com/UoM-maul1609/CPI-3V-processing/tree/master/matlab


    function cpicalctimeseries(fpath,fnames,foc_crit,trng,dt,ds,vel,outputfile)


%   Print info
    fprintf('   ====================calculating timeseries=========================\n');

%   Initialise instrument specific variables including sample area and volume for CPI v1
%   Pixel size = 2.3 micron
%   CCD array size = 1024 x 1024, set at 45 degree to flow
%   For CPI sample volume for foc = 20 (Connolly et al 2007 JTech fig 19) over range 
%   of roundness and sizes mean = 1.14E-8 m3
%   Therefore DOF = 2.1 mm
    save_files=true;
    pixsz = 2.3E-6;
    nopix = 1024;
    dof = 2.1E-3;
    sa=nopix^2*pixsz^2/sqrt(2);  % sample area of image perp to flow
    sv=nopix^2*pixsz^2*dof;      % sample volume of one image

%   Consider ds to be bin edges, unless one element, then bin size
    if length(ds) > 1
        timeser.size1 = ds(1:end-1);
        timeser.size2 = ds(2:end);
        edges = ds;
    else
        timeser.size1=0:ds:2300;
        timeser.size2=timeser.size1+ds;
        edges = unique([timeser.size1 timeser.size2]);
    end
    nl=length(timeser.size1);

%   Initialise other variables
    dt2=dt/86400;
    timeser.time=[trng(1):dt2:(trng(2)+dt2)]';
    nt=length(timeser.time);
    timeser.ar1=0:0.2:1;
    timeser.ar2=timeser.ar1+0.2;
    na=length(timeser.ar1);

%   Set up holding arrays
    timeser.conc1=zeros(nt,1);
    timeser.conc2=zeros(nt,nl);
    timeser.conc2ar=zeros(nt,nl,na);
    timeser.deadtimes=zeros(nt,1);
    timeser.nimages=zeros(nt,1);
    array=zeros(nt,nl);


%   For each file
    for i=1:length(fnames)

%       Load from file
        fprintf('   Loading from file: %s... ',strrep(fnames{i},'.roi','.mat'));
        load([fpath,strrep(fnames{i},'.roi','.mat')],'ROI_N', 'HOUSE', 'IMAGE1','STATS');
        fprintf('done\n');
    
%       Calculate timeseries, finding indices of images falling in trng
        fprintf('   Calculate timeseries...   0.0%%');
        ind=find(ROI_N.imageType==33857 & cat(1,STATS.foc.focus) > foc_crit);
        if(isempty(ind))
            fprintf('\b\b\b\b\b\bno valid images... done\n');
            continue;
        end
        tmin=min(ROI_N.Time(ind));
        tmax=max(ROI_N.Time(ind));
        ilow=find(timeser.time>=tmin);
        ilow=ilow(1);
        ihigh=find(timeser.time<tmax);
        ihigh=ihigh(end);

        itot = ihigh-ilow+1;
        icur = 0;

%       For each image...
        for j=ilow:ihigh

%           Print info
            icur = icur+1;
            fprintf('\b\b\b\b\b\b%5.1f%%',100*icur/itot);

%           Sort dead-time: find image data that are in the time-window
            indim=find(IMAGE1.Time1>=timeser.time(j)-dt2/2 & ...
                       IMAGE1.Time1<timeser.time(j)+dt2/2);
            twin=interp1(IMAGE1.Time1,IMAGE1.Time,...
                        [timeser.time(j)-dt2/2 timeser.time(j)+dt2/2],...
                        'linear','extrap');
            indho=find(HOUSE.Time >=twin(1) & HOUSE.Time< twin(2));
            timeser.deadtimes(j)=sum(HOUSE.deadtime(indho));
            timeser.nimages(j)=length(indim);        
        
%           Sort particles
            ind2=find(STATS.Time(ind)>=timeser.time(j)-dt2/2 & ...
                      STATS.Time(ind)< timeser.time(j)+dt2/2);
            timeser.conc1(j)=timeser.conc1(j)+length(ind2);
        
%           Sort size bins, now using histcounts and edges [ARDS]
            [N,X]=histcounts(STATS.len(ind(ind2)),edges);
            timeser.conc2(j,:)=timeser.conc2(j,:)+N;
        
%           Sort size and roundness bins, now using histcounts and edges [ARDS]
            for k=1:na
                ind3=ind(ind2);
                ind4=find(STATS.round(ind3)>= timeser.ar1(k) & ...
                    STATS.round(ind3)< timeser.ar2(k) );
                [N,X]=histcounts(STATS.len(ind3(ind4)),edges);
                timeser.conc2ar(j,:,k)=timeser.conc2ar(j,:,k)+N;     
            end
        end

%       Print info
        fprintf('\b\b\b\b\b\bdone\n');

    end
    
%   Scale by sample volume
    dead=repmat(timeser.deadtimes,[1 nl na]);
    nimages=repmat(timeser.nimages,[1 nl na]);
    timeser.conc2ar=timeser.conc2ar./((dt-dead).*vel.*sa+nimages.*sv);
    timeser.conc2=sum(timeser.conc2ar,3);
    timeser.conc1=sum(timeser.conc2,2);

%   Save time series file
    if save_files
        fprintf('   Saving time series to file: %s...',outputfile);
        if(exist([fpath,outputfile],'file'))
            save([fpath,outputfile],...
                'timeser','-v7.3','-append');
        else
            save([fpath,outputfile],...
                'timeser','-v7.3');
        end
        fprintf('done\n');
    end

	end