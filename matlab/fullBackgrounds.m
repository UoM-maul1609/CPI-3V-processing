function FULL_BG=fullBackgrounds(ROI_N)

% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% search for backgrounds
ind=find(ROI_N.imageType==19);
if(length(ind) == 0)
    disp('no backgrounds in this file');
    FULL_BG=[];
    return;
end
BG=ROI_N.IMAGE(ind);
%--------------------------------------------------------------------------



% +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
% rename elements of array of struct
% https://blogs.mathworks.com/loren/2010/05/13/rename-a-field-in-a-structure-array/
f=fieldnames(BG);
v=struct2cell(BG);
f{strmatch('IM',f,'exact')}='BG';
BG = cell2struct(v,f);
FULL_BG.IMAGE=BG;
FULL_BG.Time=ROI_N.Time(ind);
%--------------------------------------------------------------------------
