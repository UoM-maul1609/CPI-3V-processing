function [BG]=associateBackgrounds(ROI_N,FULL_BG)

if isempty(FULL_BG)
    disp('Error, there are no backgrounds in these files');
    return; 
end
ind=1:length(FULL_BG.Time);

% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% rename elements of array of struct
% https://blogs.mathworks.com/loren/2010/05/13/rename-a-field-in-a-structure-array/
BG=ROI_N.IMAGE;
f=fieldnames(BG);
v=struct2cell(BG);
f{strmatch('IM',f,'exact')}='BG';
BG = cell2struct(v,f);
%--------------------------------------------------------------------------


for i=1:length(ROI_N.IMAGE)
    if length(ind)==1
        jj=ind;
    else
        Timecur=ROI_N.Time(i);
        %Times=ROI_N.Times(ind);
        jj=interp1(FULL_BG.Time,ind,Timecur,'nearest');
    end
    
    BG(i).BG(:,:)=...
        FULL_BG.IMAGE(jj).BG(ROI_N.StartX(i)+1:ROI_N.EndX(i)+1,...
        ROI_N.StartY(i)+1:ROI_N.EndY(i)+1);
end