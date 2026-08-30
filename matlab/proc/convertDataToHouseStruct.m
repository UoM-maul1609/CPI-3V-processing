function H=convertDataToHouseStruct(bytes,order,house)
% convert all house to struct data


H=struct('BlockNum',1, ...'ushort' 
    'ulItemSize',1, ...'ushort' ...
    'Readings1',zeros(70,1), ...'ushort' 
    'TimeMSW',1, ...'ushort' 
    'TimeISW',1, ...'ushort' ...
    'TimeLSW',1, ...'ushort'
    'Readings',zeros(8,1), ...); ... 'ushort'
    'order',1);


H=repmat(H,[length(house) 1]); % replicate the matrix
for i=1:length(house)
    if bytes(house(i)+1)== 83
        H(i).BlockNum=bytes(house(i)+0);
        H(i).ulItemSize=bytes(house(i)+1);
        H(i).Readings1=bytes(house(i)+[2:2+70-1]); % 71
        H(i).TimeMSW=bytes(house(i)+72);
        H(i).TimeISW=bytes(house(i)+73);
        H(i).TimeLSW=bytes(house(i)+74);
        H(i).Readings=bytes(house(i)+[75:75+8-1]);
        H(i).order=order(house(i));
    end
end

ulItemSize=cat(1,H.ulItemSize);
ind=find(ulItemSize==83);
H=H(ind);
