%   **************************************************************************
%
%	M-file to get CPI time series and image data between required times and
%	return it. Time vectors are embedded in each structure (cpidat and cpiimg)
%	
%	Written 11 Mar 2021 : A.R.D. Smedley
%
%			29 Mar 2021 : refined structure output [ARDS]
%
%   **************************************************************************


	function [cpidat, cpiimg] = getcpidat(fpath,dts)


%	Find filename of time series data for day.
	tsfile = dir([fpath datestr(dts(1),'yyyymmdd') '_ts.mat']);
	tsfile = char(tsfile.name);

%	Print info
	if nargout > 1
	    fprintf('   Loading CPI time series data from file: %s... ',tsfile);
	end

%	Load time series data.
	load([fpath tsfile], 'timeser');

%	Extract portion of fields relating to period of interest (offset by 0.5s for consistency)
	ind = find(timeser.time >= dts(1)-(0.5/86400) & timeser.time < dts(2)-(0.5/86400));
	timeser.time      = timeser.time(ind);
	timeser.conc1     = timeser.conc1(ind);
	timeser.conc2     = timeser.conc2(ind,:);
	timeser.conc2ar   = timeser.conc2ar(ind,:,:);
	timeser.deadtimes = timeser.deadtimes(ind);
	timeser.nimages   = timeser.nimages(ind);

%	Convert concentration units to per cc (not per m3)
	timeser.conc1 = timeser.conc1/1E6;
	timeser.conc2 = timeser.conc2/1E6;
	timeser.conc2ar = timeser.conc2ar/1E6;

%	Normalise concentration for bin-width to calculate dN/dd
	dwid = timeser.size2-timeser.size1;
	nt = length(timeser.time);
	na = length(timeser.ar1);
	tmp1 = timeser.conc2./repmat(dwid,nt,1);
	tmp2 = timeser.conc2ar./repmat(dwid,nt,1,na);

%	Reallocate data inc. dNdd and dNddar variables to new structure
	cpidat.t         = timeser.time;
	cpidat.d1        = timeser.size1;
	cpidat.d2        = timeser.size2;
	cpidat.dctr      = (timeser.size1+timeser.size2)/2;
	% cpidat.dedg      = unique([timeser.size1 timeser.size2]);
	% cpidat.dwid      = timeser.size2-timeser.size1;
	cpidat.ar1       = timeser.ar1;
	cpidat.ar2       = timeser.ar2;
	% cpidat.arctr     = (timeser.ar1+timeser.ar2)/2;
	% cpidat.aredg     = unique([timeser.ar1 timeser.ar2]);
	% cpidat.arwid     = timeser.ar2-timeser.ar1;
	cpidat.N         = timeser.conc1;
	cpidat.dNdd      = tmp1;
	cpidat.dNddar    = tmp2;
	cpidat.deadtimes = timeser.deadtimes;
	cpidat.nimages   = timeser.nimages;

%	Clear old structure.
	clear timeser;

%	Print info
	if nargout > 1
	    fprintf('done\n');
	end


%   **************************************************************************

%	If second output is requested...
	if nargout > 1

%   	Search CPI path to find ROI / image files covering period.
	    files = dir([fpath datestr(dts(1),'yyyymmdd') '*.roi']);
	    files = {files.name};

%		Initialise counter for time valid images
		vv = 1;

%   	For each file
	    for ff = 1:length(files)

%       	Load ROI times data to check if any data within required times
	        fprintf('   Checking CPI file for required image data: %s... ',strrep(files{ff},'.roi','.mat'));
	        load([fpath,strrep(files{ff},'.roi','.mat')],'ROI_N_Time');
	        ind = find(ROI_N_Time >= dts(1) & ROI_N_Time < dts(2));
	        if isempty(ind)
	            fprintf('no data found... done\n')
	            continue
	        end
	        fprintf('done\n');

%       	Load ROI data, but only if falls within required times
	        fprintf('   Loading CPI image data from file: %s... ',strrep(files{ff},'.roi','.mat'));
	        load([fpath,strrep(files{ff},'.roi','.mat')],'STATS','ROI_N');
	        fprintf('done\n');

%			Print info
	        fprintf('   Filtering required CPI image data from file: %s... ',strrep(files{ff},'.roi','.mat'));

%			For each image
			for ii=1:length(ROI_N.Time)

%				If time is valid (falls within range) 
				if ROI_N_Time(ii) >= dts(1) & ROI_N_Time(ii) < dts(2)

%					Add image data to cpiimg structure
					cpiimg(vv).time = ROI_N.Time(ii);
					cpiimg(vv).image = ROI_N.IMAGE(ii).IM;
					cpiimg(vv).imagetype = ROI_N.imageType(ii);
					cpiimg(vv).startX = ROI_N.StartX(ii);
					cpiimg(vv).startY = ROI_N.StartY(ii);
					cpiimg(vv).endX = ROI_N.EndX(ii);
					cpiimg(vv).endY = ROI_N.EndY(ii);

%					Add statistics to cpiimg structure
					cpiimg(vv).length = STATS.len(ii);
					cpiimg(vv).width = STATS.wid(ii);
					cpiimg(vv).area = STATS.area(ii);
					cpiimg(vv).round = STATS.round(ii);
					cpiimg(vv).centroid = STATS.centroid(ii);
					cpiimg(vv).focus = STATS.foc(ii).focus;
					cpiimg(vv).boundary = STATS.foc(ii).boundaries;

					vv=vv+1;
				end
			end

%			Print info
			fprintf('done\n');

		end
	end