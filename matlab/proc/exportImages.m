function exportImages(pathname,filenames,foc_crit,size_thresh,MAP)


% 
interxy=0.01;
maxx=0.;
maxy=1.;
j=1;
flag=true;

hf=figure('position',[1 1 1024./2 1280./2],'visible','on');colormap(MAP);

runx=0.;
runy=1.;

hb=waitbar(0,'Please wait...');
for l=1:length(filenames)

    load([pathname,strrep(filenames{l},'.roi','.mat')],'dat','ROI_N');
   
    
    
    % calculate the focus +++++++++++++++++++++++++++++++++++++++++++++++++++++
    dat.focus=zeros(size(dat.len));
    for k=1:length(dat.len)
        if length(dat.foc(k).focus)
            dat.focus(k)=dat.foc(k).focus;
        else
            dat.focus(k)=NaN;
        end
    end
    %--------------------------------------------------------------------------


    if(length(find(dat.focus>foc_crit & dat.len>size_thresh))==0)
        waitbar(l/length(filenames),hb,...
            ['number ',num2str(l),' of ',num2str(length(filenames))]);
        continue;
    end
    
    i=1;
    while i<length(dat.focus)
        if(dat.focus(i)<=foc_crit || isnan(dat.focus(i)) || dat.len(i)<size_thresh)
            i=i+1;
            continue;
        end
        if(j==1)
            str1=datestr(ROI_N.Time(i),15);
            hour1=str2num(str1(1:2));
            time1=ROI_N.Time(i);
            dayold=floor(time1);
            filename1=[strrep(datestr(time1,20),'/','_'),'_',strrep([datestr(time1,13),'.'...
                    ,num2str(round((time1.*86400-floor(time1.*86400)).*1000),'%03d')],':','_'),'.png'];
        end
        
        str1=datestr(ROI_N.Time(i),15);
        hour1=str2double(str1(1:2));
        time1=ROI_N.Time(i);
        daynew=floor(time1);



        [r,c]=size(ROI_N.IMAGE(i).IM);
        figure(hf);
        h=axes('position',[runx runy-r./1280 c./1024 r./1280],'dataaspectRatiomode','manual'...
            ,'plotboxaspectratiomode','manual','units','normalized','xtick',[]','ytick',[]);
        hold on;
        runx=runx+c./1024+interxy;
        maxy=min(maxy,runy-r./1280-interxy); 

        if(maxy<-interxy || daynew ~= dayold)
            delete(h);
            pause(1);
            maxx=0.;
            maxy=1.;
            runx=0.;
            runy=1.;   

            if(~exist([pathname,filename1(1:8),'_gt',num2str(size_thresh)],'dir'))
                mkdir([pathname,filename1(1:8),'_gt',num2str(size_thresh)]);
            end
            print(hf,'-dpng',[pathname,filename1(1:8),'_gt',num2str(size_thresh),'/',filename1]);
            close(hf);
            hf=figure('position',[1 1 1024./2 1280./2],'visible','on');j=1;
            colormap(MAP);
            daynew=dayold;
            waitbar(l/length(filenames),hb,...
                ['number ',num2str(l),' of ',num2str(length(filenames))]);
            continue
        end
        if(runx<=1+interxy)
    %         imagesc(ROI_N.IMAGE(i).IM-BG1);%shading flat
            figure(hf);axes(h);
            imagesc(ROI_N.IMAGE(i).IM);%shading flat

            
            str1=datestr(ROI_N.Time(i),15);
            hour1=str2num(str1(1:2));
            time1=ROI_N.Time(i);
            
            text(0,.95,...
                [datestr(time1,13),'.'...
                ,num2str((time1.*86400-floor(time1.*86400)).*1000,'%.0f')],...
                'units','normalized','color','k','fontsize',4)
            axis tight;box off;
            i=i+1;j=j+1;
        elseif(runx>1+interxy)
            delete(h);
            runx=0.;
            runy=maxy;
            maxy=1.;
        end
    end
    waitbar(l/length(filenames),hb,...
        ['number ',num2str(l),' of ',num2str(length(filenames))]);
end

close(hb);

if(~exist([pathname,filename1(1:8),'_gt',num2str(size_thresh)],'dir'))
    mkdir([pathname,filename1(1:8),'_gt',num2str(size_thresh)]);
end
print(hf,'-dpng',[pathname,filename1(1:8),'_gt',num2str(size_thresh),'/',filename1]);
close(hf);


