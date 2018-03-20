function calcTimeseriesDriver(path,filename,foc_crit,t_range,dt,ds,vel,outputfile)

save_files=true;
sa=1280*1024/sqrt(2)*2.3e-6^2;  % sample area of image perp to flow
sv=sa*sqrt(2)*3e-3;             % sample volume of one image

disp('====================calculating timeseries=========================');


disp('Set-up arrays');
dt2=dt/86400;
timeser.Time=t_range(1):dt2:(t_range(2)+dt2);
nt=length(timeser.Time);
timeser.size1=0:ds:2300;
timeser.size2=timeser.size1+ds;
midsize=(timeser.size1+timeser.size2)./2;
nl=length(timeser.size1);
timeser.ar1=0:0.2:1;
timeser.ar2=timeser.ar1+0.2;
na=length(timeser.ar1);

timeser.conc2=zeros(nt,nl);
timeser.conc=zeros(nt,1);
timeser.deadtimes=zeros(nt,1);
timeser.nimages=zeros(nt,1);
timeser.conc2ar=zeros(nt,nl,na);
array=zeros(nt,nl);



for i=1:length(filename)
    % load from file
    disp('Loading from file...');
    load([path,strrep(filename{i},'.roi','.mat')],...
        'ROI_N', 'HOUSE', 'IMAGE1','dat');
    disp('done');

    %++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    
    
    
    % Calculate timeseries ++++++++++++++++++++++++++++++++++++++++++++++++
    disp('Calculate timeseries...');
    ind=find(ROI_N.imageType==89 & cat(1,dat.foc.focus) > foc_crit);
    tmin=min(ROI_N.Time(ind));
    tmax=max(ROI_N.Time(ind));
    ilow=find(timeser.Time>=tmin);ilow=ilow(1);
    ihigh=find(timeser.Time<tmax);ihigh=ihigh(end);
    
    
    
    
    
    %=======time loop======================================================
    for j=ilow:ihigh
        % dead-time: find imge data that are in the time-window
                    % find corresponding funny time of tiem-windown
                    % add up dead-times
        indim=find(IMAGE1.Time1>=timeser.Time(j)-dt2/2 & ...
            IMAGE1.Time1<timeser.Time(j)+dt2/2);
        twin=interp1(IMAGE1.Time1,IMAGE1.Time,...
            [timeser.Time(j)-dt2/2 timeser.Time(j)+dt2/2],...
            'linear','extrap');
        indho=find(HOUSE.Time >=twin(1) & HOUSE.Time< twin(2));
        timeser.deadtimes(j)=sum(HOUSE.deadtime(indho));

        timeser.nimages(j)=length(indim);
        
        
        
        % particles:
        ind2=find(dat.Time(ind)>=timeser.Time(j)-dt2/2 & ...
            dat.Time(ind)<timeser.Time(j)+dt2/2);
        timeser.conc(j)=timeser.conc(j)+length(ind2);
        
        
        
        % size bins========================================================
        [N,X]=hist(dat.len(ind(ind2)),midsize);
        timeser.conc2(j,:)=timeser.conc2(j,:)+N;
        %------------------------------------------------------------------
        
        % size and roundness bins==========================================
        for k=1:na
            ind3=ind(ind2);
            ind4=find(dat.round(ind3)>= timeser.ar1(k) & ...
                dat.round(ind3)< timeser.ar2(k) );
            [N,X]=hist(dat.len(ind3(ind4)),midsize);
            timeser.conc2ar(j,:,k)=timeser.conc2ar(j,:,k)+N;     
        end
        %------------------------------------------------------------------
        
    end
    %----------------------------------------------------------------------
    
end
    
% scale by sample volume
dead=repmat(timeser.deadtimes,[1 nl na]);
nimages=repmat(timeser.nimages,[1 nl na]);
timeser.conc2ar=timeser.conc2ar./((dt-dead).*vel.*sa+nimages.*sv)

timeser.conc2=sum(timeser.conc2ar,3);
timeser.conc=sum(timeser.conc2,2);




disp('done');
%----------------------------------------------------------------------


if save_files
    % save to file
    disp('Saving to file...');
    if(exist([path,outputfile],'file'))
        save([path,outputfile],...
            'timeser','-append');
    else
        save([path,outputfile],...
            'timeser');            
    end
    disp('done');
end



