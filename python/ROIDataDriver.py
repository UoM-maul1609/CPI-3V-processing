import numpy as np
import struct
from convertDataToHeaderSA import convertDataToHeaderSA
from convertDataToImageSA import convertDataToImageSA
from convertDataToROISA import convertDataToROISA
from convertDataToHouseSA import convertDataToHouseSA
from postProcess import postProcess
from fullBackgrounds import fullBackgrounds
from associateBackgrounds import associateBackgrounds
import scipy.io as sio

def ROIDataDriver(path1,filename,dt):
   t_min=1e9
   t_max=0.;
   save_files=True
   FULL_BG={'Time':np.array([]),'IMAGE':np.array([]) }
   
   
   print("=========================1st sweep================================")
   for i in range(0,len(filename)):
      fid = open(path1 + filename[i], "rb")
      print("Reading file..." + filename[i])
      bytes1=fid.read()
      print("done")
      
      lb=len(bytes1)
      # make bytes1 even
      if np.mod(lb,2) != 0:
         bytes1=bytes1[0:-1]
         lb=lb-1
         
      order=np.mgrid[0:int(len(bytes1)/2)]
      # https://stackoverflow.com/questions/45187101/converting-bytearray-to-short-int-in-python
      ushort1=struct.unpack('H'*int(lb/2), bytes1) # the H means ushort int
      order1=order
      ushort2=struct.unpack('H'*int((lb)/2), bytes1[1:len(bytes1)]+b'1') # append a byte
      order2=order[1:len(order)+1]
      
      # append the two tuples
      ushort=ushort1+ushort2
      # append two nmpy arrays
      order=np.append(order1,order2)
      
      
      bytes1=struct.pack('H'*int(lb), *ushort) # the H means ushort int
      
      
      ushort=np.asarray(ushort)
      
      #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      #find the positions of the start of block marks ++++++++++++++++++++++++++
      #484B
      print('Finding house keeping...')
      house,=np.where(ushort==int('0x484B',0))
      print('done')
      
      # A3D5
      print('Finding image data...')
      images,=np.where(ushort==int('0xa3d5',0)) 
      print('done')
      
      #B2E6
      print('Finding roi data...')
      rois,=np.where(ushort==int('0xb2e6',0)) 
      print('done')
      
      fid.close()
      #--------------------------------------------------------------------------
      
      #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      #convert the bytes1 to Header, Images and ROI structs +++++++++++++++++++++
      print('Post-processing data, stage 1...')
      Header=convertDataToHeaderSA(ushort)
      (I,images)=convertDataToImageSA(bytes1,ushort,order,images)
      (R,rois)=convertDataToROISA(bytes1,ushort,order,rois)
      (H)=convertDataToHouseSA(bytes1,ushort,order,house)
      print('done')
      #--------------------------------------------------------------------------
      
      
      #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      # extract the images, times, etc ++++++++++++++++++++++++++++++++++++++++++
      print('Post-processing data, stage 2...')
      (ROI_N,HOUSE,IMAGE1)=postProcess(bytes1,rois,R,H,I,Header)
      print('done')
      #--------------------------------------------------------------------------



      #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      #Backgrounds +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      print('Getting backgrounds...')
      FULL_BG1=fullBackgrounds(ROI_N) # append here
      if len(FULL_BG1['Time']):
          FULL_BG['IMAGE']=np.append(FULL_BG['IMAGE'],FULL_BG1['IMAGE'])
          FULL_BG['Time']=np.append(FULL_BG['Time'],FULL_BG1['Time'])
      print('done')
      #--------------------------------------------------------------------------



      #https://stackoverflow.com/questions/10012788/python-find-min-max-of-two-lists
      t_min=np.minimum(np.min(ROI_N['Time']),t_min)
      t_max=np.maximum(np.max(ROI_N['Time']),t_max)
      
      t_range=np.array([np.floor(t_min*86400/dt)*dt/86400,
                        np.ceil(t_max*86400/dt)*dt/86400])
    

      if save_files:
          #https://docs.scipy.org/doc/scipy/reference/tutorial/io.html
          print('Saving to file...')
          sio.savemat(path1 + filename[i].replace('.roi','.mat'),
                      {'ROI_N':ROI_N, 'HOUSE':HOUSE,'IMAGE1':IMAGE1})
          sio.savemat(path1 + 'full_backgrounds.mat',
                      {'FULL_BG':FULL_BG,'t_range':t_range})
          print('done')


   print('=========================2nd sweep================================')
   for i in range(0,len(filename)):
      # load from file
      print('Loading from file...')
      #https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries      
      dataload=sio.loadmat(path1 + filename[i].replace('.roi','.mat'),
                           variable_names=['ROI_N','HOUSE','IMAGE1'])
      #dataload['ROI_N']['StartX'][0,0][0,:]
      ROI_N=dataload['ROI_N']
      HOUSE=dataload['HOUSE']
      IMAGE1=dataload['IMAGE1']
      print('done')



      #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
      # Assocate backgrounds ++++++++++++++++++++++++++++++++++++++++++++++++
      print('Associate backgrounds...')
      BG=associateBackgrounds(ROI_N,FULL_BG)
      print('done')
      #----------------------------------------------------------------------
    
    
      if save_files:
          #https://docs.scipy.org/doc/scipy/reference/tutorial/io.html
          print('Saving to file...')
          sio.savemat(path1 + filename[i].replace('.roi','.mat'),
                      {'ROI_N':ROI_N, 'HOUSE':HOUSE,'IMAGE1':IMAGE1,'BG':BG})
          #with open(path1 + filename[i].replace('.roi','.mat'),'ab') as f:
          #   sio.savemat(f, {'BG':BG})
          print('done')



      

   return (bytes1,house,images,rois,ushort,Header,I,R,H,t_range)
      

