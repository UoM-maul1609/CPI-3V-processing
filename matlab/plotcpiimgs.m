%   **************************************************************************
%
%   M-file to extract CPI images and plot.
%   
%   Written 29 Jan 2021 - A.R.D. Smedley lightly editted from PJC's version 
%   exportimagesdriver.m at
%   https://github.com/UoM-maul1609/CPI-3V-processing/tree/master/matlab
%           16 Mar 2021 - redirected to get data from getcpidat.m
%
%   **************************************************************************


    function plotcpiimgs(dts,size_thres,foc_crit)


%   Set path
    fpath = 'E:\ICE-RF\CPI\';

%   Specify start and finish times, check size threshold and foc critical value
    fprintf('\n');
    if nargin < 1
        stt = input('   Enter start date and time [YYYY-MM-DD hh:mm:ss]  > ','s');
        dts(1) = datenum(stt,31);
        fin = input('   Enter finish date and time [YYYY-MM-DD hh:mm:ss] > ','s');
        dts(2) = datenum(fin,31);
    end
    if nargin < 2
        size_thres = input('   Enter size threshold [50 microns]                > ');
        if isempty(size_thres), size_thres = 50;, end
        end
    if nargin < 3
        foc_crit = input('   Enter focus critical value [20]                  > ');
        if isempty(foc_crit), foc_crit = 20;, end
        fprintf('   --\n');
    end
    
%   Print start and end times, and duration to screen.
    if nargin < 1
        fprintf('   Start     : %s\n',datestr(dts(1),'yyyy-mm-dd HH:MM:SS'));
        fprintf('   Finish    : %s\n',datestr(dts(2),'yyyy-mm-dd HH:MM:SS'));
        fprintf('   Duration  : %d s\n',round((dts(2)-dts(1))*86400));
        fprintf('   --\n');
    end

%   **************************************************************************

%   Initialise variables
    interxy=0.01;
    maxx = 0.;
    maxy = 1.;
    j = 1;
    flag = true;
    runx = 0.;
    runy = 1.;

%   Get screensize and set figure size
    scrnsz = get(0,'Screensize');
    scrnw = scrnsz(3);
    scrnh = scrnsz(4);
    figw = 512;
    figh = 640;
    figl = (scrnw-figw)/2;
    figb = (scrnh-figh-80)/2; % accounting for window menu height
    figpos = [figl figb figw figh];

%   Load colour map
    load grnlndgrad.mat
    cmap = grnlndgrad;

%   Set-up figure and colourmap
    bg = [0.95 0.95 0.95];
    hf = figure('position',figpos,'visible','on','color',bg); 
    colormap(cmap);

%   Get image data for time period requested
    [~,cpiimg] = getcpidat(fpath,dts);

%   Get number of valid images in period
    ntot = nnz([cpiimg.focus] > foc_crit & ~isnan([cpiimg.focus]) & [cpiimg.length] >= size_thres);

%   Print info
    fprintf('   --\n')
    fprintf('   Total images found    : %d\n',ntot);
    fprintf('   Plotting images ... \n')
    fprintf('   Image acquisition time:            ');

%   For images that meet focus and size criteria    
    i = 1;
    itot = length(cpiimg);
    while i<length(cpiimg)

%       Check focus and size criterion and skip if out of range
        if cpiimg(i).focus <= foc_crit || isnan(cpiimg(i).focus) || cpiimg(i).length < size_thres
            i = i+1;
            continue;
        end

%       Get image size and set figure and axes position
        [r,c] = size(cpiimg(i).image);
        figure(hf);
        h = axes('position',[runx runy-r./1280 c./1024 r./1280],'dataaspectRatiomode','manual', ...
                 'plotboxaspectratiomode','manual','units','normalized','xtick',[]','ytick',[], ...
                 'color',bg,'Xcolor', bg, 'Ycolor', bg);
        hold on;
        runx = runx+c./1024+interxy;
        maxy = min(maxy,runy-r./1280-interxy); 

%       Start new figure if full
        if(maxy<-interxy)
            delete(h);
            pause(1);
            maxx = 0.;
            maxy = 1.;
            runx = 0.;
            runy = 1.;   

            hf = figure('position',figpos+[50 0 0 0],'visible','on');
            figpos = get(gcf,'position'); 
            j = 1;
            colormap(cmap);
            continue
        end

%       If space on row add next image, despeckling and scaling colourmap to 2-98 percentiles
        if(runx<=1+interxy)
            figure(hf);
            axes(h);
            imfilt = medfilt2(cpiimg(i).image,[3 3]); 
            clims = prctile(imfilt(:),[2 98]);
            imagesc(imfilt);
            caxis(clims)
                
%           Add time of capture to image; seconds to 2 d.p.; also print to screen
            imtime = cpiimg(i).time;
            imtimestr = [datestr(imtime,13) '.' sprintf('%02d',floor(rem(86400*imtime,1).*100))];
            text(0,.95,imtimestr,'units','normalized','color','k','fontsize',4);
            fprintf('\b\b\b\b\b\b\b\b\b\b\b%s',imtimestr)
            axis tight;
            box off;
            i = i+1;
            j = j+1;

%       If no space on row, start new row
        elseif(runx>1+interxy)
            delete(h);
            runx = 0.;
            runy = maxy;
            maxy = 1.;
        end
    end

%   Print info
    fprintf('\n   Plotting complete.\n\n');
