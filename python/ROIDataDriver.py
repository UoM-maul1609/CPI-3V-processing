import numpy as np
from struct import pack, unpack
from convertDataToHeaderSA import convertDataToHeaderSA
from convertDataToImageSA import convertDataToImageSA
from convertDataToROISA import convertDataToROISA
from convertDataToHouseSA import convertDataToHouseSA
from postProcess import postProcess
from fullBackgrounds import fullBackgrounds
from associateBackgrounds import associateBackgrounds
import scipy.io as sio
import gc
import os.path
import tempfile
from multiprocessing import Pool


def ROIDataDriver(path1,filename,dt,process_sweep1_if_exist):
   t_min=1e9
   t_max=0.
   FULL_BG={'Time':np.array([]),'IMAGE':np.array([]) }
 
   save_files=True
   
   print("=========================1st sweep================================")
   for i in range(0,len(filename)):
       p=Pool(processes=1)

       """result=p.apply_async(mult_job,\
                (path1,filename[i],dt,FULL_BG,t_min,t_max,save_files,\
                 process_sweep1_if_exist))"""
       result=p.apply(mult_job,\
                (path1,filename[i],dt,FULL_BG,t_min,t_max,save_files,\
                 process_sweep1_if_exist))
       (FULL_BG,t_min,t_max)=result.get()

       p.close()
       p.join()
       del p

       
   print('=========================2nd sweep================================')
   dataload=sio.loadmat("{0}{1}".format(path1, 'full_backgrounds.mat'),
                           variable_names=['FULL_BG','t_range'])
   FULL_BG=dataload['FULL_BG']
   t_range=dataload['t_range']
   del dataload
   for i in range(0,len(filename)):
       p=Pool(processes=1)

       p.apply_async(mult_job2,\
                (path1,filename[i],FULL_BG,save_files))
       p.close()
       p.join()
       del p


   return (t_range)
 







def mult_job(path1,filename1,dt,FULL_BG,t_min,t_max,save_files,\
             process_sweep1_if_exist):   

   if (os.path.isfile("{0}{1}".format(path1 ,filename1.replace('.roi','.mat'))) 
      and not(process_sweep1_if_exist)):
       
       dataload=sio.loadmat(path1+'full_backgrounds.mat', \
                            variable_names=['FULL_BG','t_range'])
       del dataload
       FULL_BG=dataload['FULL_BG']
       t_range1=dataload['t_range']
       t_min=t_range1[0,0]
       t_max=t_range1[0,1]
      
       del dataload
      
       print("{0}{1}".format("Skipping file...", filename1))
       bytes1,house,images,rois,ushort,Header,I,R,H = \
           False, False, False, False, False, False, False, False, False
       return (FULL_BG,t_min,t_max)
       
   fid = open("{0}{1}".format(path1, filename1), "rb")
   print("{0}{1}".format("Reading file...", filename1))
   bytes1=fid.read()
   print("done")
      
   lb=len(bytes1)
   # make bytes1 even
   if np.mod(lb,2) != 0:
       bytes1=bytes1[0:-1]
       lb=lb-1
     
   order=np.mgrid[0:int(len(bytes1)/2)]
   # https://stackoverflow.com/questions/45187101/converting-bytearray-to-short-int-in-python
   ushort1=unpack('H'*int(lb/2), bytes1) # the H means ushort int
   order1=order
   ushort2=unpack('H'*int((lb)/2), bytes1[1:len(bytes1)] + b'1') # append a byte
   order2=order[1:len(order)+1]
      
   # append the two tuples
   #http://datos.io/2016/10/04/python-memory-issues-tips-and-tricks/
   #ushort="{0}{1}".format(ushort1,ushort2)
   ushort=ushort1+ushort2
      
   del ushort1, ushort2
   # append two nmpy arrays
   order=np.append(order1,order2)

   del order1, order2
      
   bytes1=pack('H'*int(lb), *ushort) # the H means ushort int
      
      
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
      
   # Garbage collection:
   # https://stackoverflow.com/questions/1316767/how-can-i-explicitly-free-memory-in-python
   gc.collect()
   del gc.garbage[:]
      
      
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
    
    
   # Garbage collection:
   gc.collect()
   del gc.garbage[:]
    
    
   #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   #Backgrounds +++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
   print('Getting backgrounds...')
   FULL_BG1=fullBackgrounds(ROI_N) # append here
   if len(FULL_BG1['Time']):
       temp_name=tempfile.mktemp()
       sio.savemat(temp_name,{'FULL_BG':FULL_BG1})
       dataload=sio.loadmat(temp_name,variable_names=['FULL_BG'])
       FULL_BG1=dataload['FULL_BG']
       del dataload
      
       r=np.shape(FULL_BG['IMAGE'])
       if len(r)==1:
           FULL_BG=FULL_BG1
       else: # append
           FULL_BG['IMAGE'][0,0]=np.append(FULL_BG['IMAGE'][0,0], \
             FULL_BG1['IMAGE'][0,0],axis=1)
           FULL_BG['Time'][0,0]=np.append(FULL_BG['Time'][0,0], \
             FULL_BG1['Time'][0,0],axis=0)
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
       sio.savemat("{0}{1}".format(path1, filename1.replace('.roi','.mat')),
                   {'ROI_N':ROI_N, 'HOUSE':HOUSE,'IMAGE1':IMAGE1})
       sio.savemat("{0}{1}".format(path1, 'full_backgrounds.mat'),
                   {'FULL_BG':FULL_BG,'t_range':t_range})
       print('done')
      
   del ROI_N, HOUSE, IMAGE1
   # Garbage collection:
   gc.collect()
   del gc.garbage[:]
       
   return (FULL_BG,t_min,t_max)
    
   
def mult_job2(path1,filename1,FULL_BG,save_files):
    # load from file
    print('Loading from file...')
    #https://stackoverflow.com/questions/7008608/scipy-io-loadmat-nested-structures-i-e-dictionaries      
    dataload=sio.loadmat("{0}{1}".format(path1, filename1.replace('.roi','.mat')),
                           variable_names=['ROI_N','HOUSE','IMAGE1'])
    #dataload['ROI_N']['StartX'][0,0][0,:]
    ROI_N=dataload['ROI_N']
    HOUSE=dataload['HOUSE']
    IMAGE1=dataload['IMAGE1']
    del dataload
      
    print('done')



    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    # Assocate backgrounds ++++++++++++++++++++++++++++++++++++++++++++++++
    print("{0}{1}".format('Associate backgrounds...', filename1))
    BG=associateBackgrounds(ROI_N,FULL_BG)
    print('done')
    #----------------------------------------------------------------------
    
    
    if save_files:
        #https://docs.scipy.org/doc/scipy/reference/tutorial/io.html
        print('Saving to file...')
        sio.savemat("{0}{1}".format(path1, filename1.replace('.roi','.mat')),
                      {'ROI_N':ROI_N, 'HOUSE':HOUSE,'IMAGE1':IMAGE1,'BG':BG})
        #with open(path1 + filename[i].replace('.roi','.mat'),'ab') as f:
        #   sio.savemat(f, {'BG':BG})
        print('done')



    del ROI_N, HOUSE, IMAGE1, BG

    # Garbage collection:
    gc.collect()
    del gc.garbage[:]    

    return