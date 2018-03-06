function dat=imageStats(ROI_N,BG,b_flag)

warning off;
ind=find(ROI_N.imageType==89);
l1=length(ROI_N.IMAGE);
dat.len=zeros(l1,1);
dat.wid=zeros(l1,1);
dat.area=zeros(l1,1);
dat.round=zeros(l1,1);
dat.centroid=zeros(l1,2);
dat.foc=repmat(struct('focus',NaN),[l1 1]);
% dat.foc=repmat(struct('focus',NaN,'xs',zeros(20,59),'ys',zeros(20,59), ...
%     'boundaries',NaN.*zeros(60,2)),[l1 1]);
dat.Time=ROI_N.Time';

h = waitbar(0,'Please wait...');
ind111=0;
for i=ind'
    arr=ROI_N.IMAGE(i).IM-BG(i).BG;
 
    inda=find(double(arr(:))<=-15);
    indb=find(double(arr(:))>-15);
    if(length(inda)<=3)
        continue;
    end

    arr=arr-min(arr(:));

    [r,c]=size(arr);
    bw1=edge(arr,'sobel');
    % bw2=edge(ROI_N.IMAGE(1).IM,'sobel');
    level=graythresh(arr./256);
%     BW=im2bw(arr./256,level); % 0.5 
    BW=im2bw(arr./256,0.15); % 0.5 
    BW2=medfilt2(BW,[5 5]);
    BW2(1,:)=1;BW2(end,:)=1;    BW2(:,1)=1;BW2(:,end)=1;
    ind=find(double(BW2)==0);
    [ii,jj]=ind2sub([r c],ind);
    ind2=find(ii==min(ii));
    
%     boundary = bwtraceboundary(BW2,[jj(ind2(lev)) ii(ind2(lev))],'NW');
    [boundary,L,N,A] = bwboundaries(BW2);
    [B1,I1,J1]=unique(L(:));
    len2=0.;
    k1=1;
    if length(B1)<2
        continue;
    end
    for k=2:length(B1)
        len2=max(length(find(L(:)==B1(k))),len2); %number of points in this roi
        if(length(find(L(:)==B1(k)))==len2) % saves the largest one that isn't the first
            k1=k;
        end
        
    end
    [xx,yy]=meshgrid(1:r,1:c);
%     IN=inpolygon(xx,yy,boundary{k1}(:,1),boundary{k1}(:,2));
    
    [rows columns] = size(arr); 
    pixelsInPolygon = poly2mask(boundary{k1}(:,1),boundary{k1}(:,2), columns, rows);
    IN=pixelsInPolygon;
    
    STATS = regionprops(double(IN),'FilledArea','Centroid','MajorAxisLength','MinorAxisLength','Perimeter');
    if ~length(STATS)
        continue;
    end
    dat.len(i)=STATS.MajorAxisLength.*2.3;
    dat.wid(i)=STATS.MinorAxisLength.*2.3;
    dat.area(i)=STATS.FilledArea.*2.3.^2;
    dat.round(i)=STATS.FilledArea./(pi./4.*STATS.MajorAxisLength.^2);
    dat.centroid(i,:)=STATS.Centroid;
    boundaries=boundary{k1};
    
    foc=calculate_focus(boundaries,arr,b_flag); % maybe focus less than 12?

    dat.foc(i)=foc;
    
    waitbar(i/length(ROI_N.IMAGE),h);
end
close(h);
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function foc=calculate_focus(boundaries,IM,b_flag)
persistent boundaries2 xs ys focus r c;



ld=3;
% [r,c]=size(boundaries);
ind1=linspace(1,30,length(boundaries(:,1)));
ind2=linspace(1,ind1(end),30);

if isempty(boundaries2) 
    boundaries2=zeros(length(ind2),2); 
    [r,c]=size(boundaries2);

    xs=zeros(ld,r-1);
    ys=zeros(ld,r-1);
    focus=zeros(r-1,1);
end

if(length(boundaries(:,1))>9)
    boundaries2(:,1)=interp1(ind1,filtfilt(ones(1,3),3,boundaries(:,1)),ind2);
    boundaries2(:,2)=interp1(ind1,filtfilt(ones(1,3),3,boundaries(:,2)),ind2);
else
%     boundaries2=boundaries;
    boundaries2(:,1)=interp1(ind1,boundaries(:,1),ind2);
    boundaries2(:,2)=interp1(ind1,boundaries(:,2),ind2);
end
midpointx=(boundaries2(2:end,1)+boundaries2(1:end-1,1))./2;
midpointy=(boundaries2(2:end,2)+boundaries2(1:end-1,2))./2;


gradient1=(boundaries2(2:end,2)-boundaries2(1:end-1,2))./(boundaries2(2:end,1)-boundaries2(1:end-1,1));

gradient2=-1./(gradient1);

% find start and end point of the lines to calculate the gradients along
dl=-2; % length 2 pixels
dx1=dl./sqrt(1+gradient2.^2);
dy1=dl./sqrt(1+1./gradient2.^2);

dl=2; % length 2 pixels
dx2=dl./sqrt(1+gradient2.^2);
dy2=dl./sqrt(1+1./gradient2.^2);

deltax=(boundaries2(2:end,1)-boundaries2(1:end-1,1));
deltay=(boundaries2(2:end,2)-boundaries2(1:end-1,2));


for i=1:r-1
    xs(:,i)=linspace(midpointx(i)+dx1(i),midpointx(i)+dx2(i),ld);
    ys(:,i)=linspace(midpointy(i)-dy1(i),midpointy(i)-dy2(i),ld);
end
% correct for direction of gradient
ind=find(deltay<0 & deltax>0);
for i=ind'
    ys(:,i)=linspace(midpointy(i)+dy1(i),midpointy(i)+dy2(i),ld);
end
ind=find(deltay>0 & deltax<0);
for i=ind'   
    ys(:,i)=linspace(midpointy(i)+dy1(i),midpointy(i)+dy2(i),ld);
end

% now calculate the focus
[rr,cc]=size(IM);
[rr,cc]=meshgrid(1:rr,1:cc);
try
    for i=1:r-1
        intensity=interp2(rr,cc,IM',xs(:,i),ys(:,i));
%        focus(i)=nanmean(abs(parabolderiv(linspace(-2,2,ld)',intensity,3,0)));
        focus(i)=nanmean(abs(parabolderiv(linspace(-2,2,ld)',intensity,1,0)));
%         foc.intensity(:,i)=intensity;
    end
catch
    focus=NaN;
%     foc.intensity=NaN.*ones(size(xs(:,1)));
end
% foc.foci=focus;
foc.focus=nanmean(focus);
if b_flag
   foc.xs=xs;
   foc.ys=ys;
   foc.boundaries=boundaries2;
end
% foc
