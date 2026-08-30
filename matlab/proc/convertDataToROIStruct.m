function [R,rois]=convertDataToROIStruct(bytes,order,rois)
% convert all rois to struct data

R=struct('ulItemSize',1, ...'ulong' 
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


R=repmat(R,[length(rois) 1]); % replicate the matrix
for i=1:length(rois)
    R(i).ulItemSize=double(typecast(uint16(bytes(rois(i)+[1 2])),'uint32'));
    R(i).usVersion=bytes(rois(i)+3);
    if(R(i).usVersion==25)
        R(i).StartX=bytes(rois(i)+4);
        R(i).StartY=bytes(rois(i)+5);
        R(i).EndX=bytes(rois(i)+6);
        R(i).EndY=bytes(rois(i)+7);
        R(i).PixBytes=double(typecast(uint16(bytes(rois(i)+8)),'int16'));;
        R(i).usROIFlags=bytes(rois(i)+9);

        R(i).fLength=double(typecast(uint16(bytes(rois(i)+[10 11])),'single'));
        R(i).ulStartLen=double(typecast(uint16(bytes(rois(i)+[12 13])),'uint32'));
        R(i).ulEndLen=double(typecast(uint16(bytes(rois(i)+[14 15])),'uint32'));

        R(i).fWidth=double(typecast(uint16(bytes(rois(i)+[16 17])),'single'));
        R(i).Spare=double(typecast(uint16(bytes(rois(i)+[18:18+9-1])),'int8'));
        R(i).order=order(rois(i));
    
    end    
end

ver=cat(1,R.usVersion);
ind=find(ver==25);
R=R(ind);
rois=rois(ind);

