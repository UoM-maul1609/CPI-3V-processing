function timeser=calcTimeseriesDriver(path1,filename,foc_crit,dt,ds,vel,outputfile,cpiv1)
%CALCTIMESERIESDRIVER Concentration time series matching the Python reference.
%
% The Python implementation in python/proc/calcTimeseriesDriver.py is the
% canonical implementation.  This MATLAB port fixes the principal historical
% divergences: time-bin edge handling, accumulation across multiple files,
% CPI-v1 image/dead-time handling, and histogram-based size/roundness binning.

if nargin < 8
    cpiv1=false;
end

sa=1280*1024/sqrt(2)*2.3e-6^2; % image sample area perpendicular to flow
sv=sa*sqrt(2)*3e-3;            % sample volume represented by one image

bgfile=fullfile(path1,'full_backgrounds.mat');
S=load(bgfile,'t_range');
t_range=S.t_range;

dt_days=dt/86400;
timeser.Time=t_range(1):dt_days:(t_range(2)+0.5*dt_days);
timeser.size1=0:ds:2300;
timeser.size2=timeser.size1+ds;
timeser.midsize=timeser.size1+ds/2;
timeser.ar1=0:0.2:1.0;
timeser.ar2=timeser.ar1+0.2;

nt=numel(timeser.Time);
nl=numel(timeser.size1);
na=numel(timeser.ar1);
time_edges=[timeser.Time-dt_days/2, timeser.Time(end)+dt_days/2];
size_edges=0:ds:(2300+ds);
ar_edges=[timeser.ar1, timeser.ar1(end)+0.2];

timeser.conc2=zeros(nt,nl);
timeser.conc=zeros(nt,1);
timeser.deadtimes=zeros(nt,1);
timeser.nimages=zeros(nt,1);
timeser.conc2ar=zeros(nt,nl,na);

if cpiv1
    image_type_value=33857;
else
    image_type_value=89;
end

fprintf('====================calculating timeseries=========================\n');
for ii=1:numel(filename)
    matfile=fullfile(path1,strrep(filename{ii},'.roi','.mat'));
    fprintf('[%d/%d] %s\n',ii,numel(filename),matfile);
    S=load(matfile,'HOUSE','IMAGE1','dat');
    HOUSE=S.HOUSE;
    IMAGE1=S.IMAGE1;
    dat=S.dat;

    % New Python-generated files duplicate this tiny field into dat so that
    % ROI_N (and all nested images) need not be loaded for concentration work.
    if isfield(dat,'imageType')
        image_types=dat.imageType(:);
    else
        R=load(matfile,'ROI_N');
        image_types=R.ROI_N.imageType(:);
    end

    focus=cat(1,dat.foc.focus);
    n_particle=min([numel(image_types),numel(focus),numel(dat.len)]);
    selected=(image_types(1:n_particle)==image_type_value) & ...
             (focus(1:n_particle)>foc_crit);
    ind=find(selected);

    image_times=double(IMAGE1.Time1(:));
    if ~isempty(image_times)
        timeser.nimages=timeser.nimages+histcounts(image_times,time_edges).';
    end

    % Dead time.  Accumulate rather than assign: several input files can
    % contribute samples to the same output time bin.
    if cpiv1
        if ~isempty(image_times)
            house_raw=double(HOUSE.Time(:));
            house_time=house_raw/86400-floor(house_raw/86400)+floor(image_times(1));
            dead=double(HOUSE.deadtime(:));
            for j=1:nt
                in=(house_time>=time_edges(j)) & (house_time<time_edges(j+1));
                timeser.deadtimes(j)=timeser.deadtimes(j)+sum(dead(in),'omitnan');
            end
        end
    else
        image_clock=double(IMAGE1.Time(:));
        if numel(image_times)>=2 && numel(image_clock)>=2
            house_time=double(HOUSE.Time(:));
            dead=double(HOUSE.deadtime(:));
            for j=1:nt
                twin=interp1(image_times,image_clock,time_edges(j:j+1),'linear','extrap');
                in=(house_time>=twin(1)) & (house_time<twin(2));
                timeser.deadtimes(j)=timeser.deadtimes(j)+sum(dead(in),'omitnan');
            end
        end
    end

    if isempty(ind)
        continue;
    end

    particle_time=double(dat.Time(ind));
    lengths=double(dat.len(ind));
    roundness=double(dat.round(ind));

    % MATLAB has histcounts2 but no universally available histcounts3.  A
    % short loop over the six roundness bins keeps the code vectorized in the
    % much larger time/size dimensions.
    for k=1:na
        in=(roundness>=ar_edges(k)) & (roundness<ar_edges(k+1));
        if any(in)
            N=histcounts2(particle_time(in),lengths(in),time_edges,size_edges);
            timeser.conc2ar(:,:,k)=timeser.conc2ar(:,:,k)+N;
        end
    end
end

% Convert counts to concentration using the same effective-volume expression
% as Python.  Avoid repmat of the full 3-D array.
effective_volume=(dt-timeser.deadtimes).*vel.*sa+timeser.nimages.*sv;
valid=effective_volume>0;
timeser.conc2ar(~repmat(valid,[1 nl na]))=NaN;
timeser.conc2ar(valid,:,:)=bsxfun(@rdivide,timeser.conc2ar(valid,:,:),effective_volume(valid));
timeser.conc2=sum(timeser.conc2ar,3,'omitnan');
timeser.conc=sum(timeser.conc2,2,'omitnan');

% Save via a temporary file so an interrupted write does not destroy the
% previous output.
outfile=fullfile(path1,outputfile);
tmp=[outfile,'.tmp.mat'];
save(tmp,'timeser','-v7');
movefile(tmp,outfile,'f');
fprintf('Saved %s\n',outfile);
end
