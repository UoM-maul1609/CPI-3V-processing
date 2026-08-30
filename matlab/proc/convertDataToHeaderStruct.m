function Header=convertDataToHeaderStruct(ushort)

Header=struct('ucVersion',1, ... 'ushort'), 
    'usYear',1, ... 'ushort'), ...
    'ucMonth',1, ... 'ushort'),...
    'ImageX',1, ... 'ushort'), ...
    'ImageY',1 , ...'ushort'), ...
    'szInfo',zeros(70,1));
Header.ucVersion=ushort(1);
Header.usYear=ushort(2);
Header.ucMonth=ushort(3);
Header.ImageX=ushort(4);
Header.ImageY=ushort(5);
Header.szInfo=ushort([6:6+70-1]);
