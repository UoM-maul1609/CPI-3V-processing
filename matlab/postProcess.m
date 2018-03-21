function [ROI_N,HOUSE,IMAGE1]=postProcess(bytes,rois,R,H,I,Header)
lb=length(bytes);

%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% image times in hk format
IMAGE1.Time=...
    cat(1,I.ArrivalTime2).*16.^8+...
    cat(1,I.ArrivalTime1).*16.^4+cat(1,I.ulTime);


% msec=round(1000.*(cat(1,I.ArrivalTime1)./16.) + (1000.*cat(1,I.ArrivalTime2)./1250000.));
% msec2=(1000.*(cat(1,I.ArrivalTime1)./16.) + (1000.*cat(1,I.ArrivalTime2)./1250000.));

% time in MATLAB format, using msecond, etc
Time=datenum(Header.usYear,Header.ucMonth,...
    cat(1,I.day),...
    cat(1,I.hour),...
    cat(1,I.minute),...
    cat(1,I.second)) ....
    +cat(1,I.msecond)./(1000)./24./3600;

str1=strcat(num2str(Time.*86400-floor(Time.*86400),'%0.3f'));
str1(:,1)='.';
Timestr=[datestr(Time),str1(:,2:end)];

% image type
imageType=cat(1,I.ImageType);
IMAGE1.Time1=Time;
IMAGE1.Timestr=Timestr;
IMAGE1.imageType=imageType;
%--------------------------------------------------------------------------





%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% housekeeping packet times
HOUSE.Time=(cat(1,H.TimeMSW).*16.^8+ ...
    cat(1,H.TimeISW).*16.^4+...
    cat(1,H.TimeLSW));
Rdgs=cat(2,H.Readings1);
HOUSE.deadtime=Rdgs(67,:)*0.000341333;
%--------------------------------------------------------------------------








%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% map 'IMAGE' to ROI
orderImage=cat(1,I.order);
orderROIs=cat(1,R.order);
timeROIs=zeros(length(R),1);
imageTypeROIs=zeros(length(R),1);

for i=1:length(orderImage)-1
    ind=find(orderImage(i)<orderROIs & orderImage(i+1)>orderROIs );
    timeROIs(ind)=Time(i);
    imageTypeROIs(ind)=imageType(i);
end
i=length(orderImage);
ind=find(orderImage(i)<orderROIs );
timeROIs(ind)=Time(i);
imageTypeROIs(ind)=imageType(i);
%--------------------------------------------------------------------------






%++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% associate variables with each ROI++++++++++++++++++++++++++++++++++++++++
[a,b]=sort(timeROIs);
for i=1:length(rois)
    StartX=R(b(i)).StartX;
    StartY=R(b(i)).StartY;
    EndX=R(b(i)).EndX;
    EndY=R(b(i)).EndY;
    
    X=(EndX-StartX+1);
    Y=(EndY-StartY+1);
    numberOfChars=X*Y;

    chars1=bytes(rois(b(i))*2+33*2-14+[1:numberOfChars]);
    ROI_N.IMAGE(i).IM=reshape(chars1,[X,Y]);
end
ROI_N.Time=timeROIs;
ROI_N.imageType=imageTypeROIs;
ROI_N.StartX=cat(1,R.StartX);
ROI_N.StartY=cat(1,R.StartY);
ROI_N.EndX=cat(1,R.EndX);
ROI_N.EndY=cat(1,R.EndY);
%--------------------------------------------------------------------------


% sort
[a,b]=sort(ROI_N.Time);
ROI_N.Time(:)=ROI_N.Time(b);
ROI_N.imageType(:)=ROI_N.imageType(b);
ROI_N.StartX(:)=ROI_N.StartX(b);
ROI_N.StartY(:)=ROI_N.StartY(b);
ROI_N.EndX(:)=ROI_N.EndX(b);
ROI_N.EndY(:)=ROI_N.EndY(b);
%ROI_N.IMAGE(:)=ROI_N.IMAGE(b);

% sort
[a,b]=sort(HOUSE.Time);
HOUSE.Time(:)=HOUSE.Time(b);
HOUSE.deadtime(:)=HOUSE.deadtime(b);

% sort
[a,b]=sort(IMAGE1.Time);
IMAGE1.Time(:)=IMAGE1.Time(b);
IMAGE1.Time1(:)=IMAGE1.Time1(b);
IMAGE1.Timestr(:,:)=IMAGE1.Timestr(b,:);
IMAGE1.imageType(:)=IMAGE1.imageType(b);






