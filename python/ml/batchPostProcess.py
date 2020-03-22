foc_crit=12 # critical value of focus for an image
min_len=50 # minimum length for particle images

post_process=True

path1=['/models/mccikpc2/CPI-analysis/C072/3VCPI/' , \
       '/models/mccikpc2/CPI-analysis/C073/3VCPI/' , \
       '/models/mccikpc2/CPI-analysis/C074/3VCPI/' , \
       '/models/mccikpc2/CPI-analysis/C075/3VCPI/' , \
       '/models/mccikpc2/CPI-analysis/C076/3VCPI/' , \
       '/models/mccikpc2/CPI-analysis/C077/3VCPI/' , \
       '/models/mccikpc2/CPI-analysis/C078/3VCPI/' , \
       '/models/mccikpc2/CPI-analysis/C079/3VCPI/' , \
       '/models/mccikpc2/CPI-analysis/C080/3VCPI/' , \
       '/models/mccikpc2/CPI-analysis/C081/3VCPI/' , \
       '/models/mccikpc2/CPI-analysis/C082/3VCPI/' , \
       '/models/mccikpc2/CPI-analysis/C097/3VCPI/' , \
       '/models/mccikpc2/CPI-analysis/C098/3VCPI/' ]

outputfile='/models/mccikpc2/CPI-analysis/postProcessed_l50.h5'



def runJobs():
    global path1
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if post_process:
        from postProcessingDriver import postProcessingDriver
        # extract and post process all images and save to file
        postProcessingDriver(path1,outputfile,foc_crit,min_len)
    #--------------------------------------------------------------------------
    

if __name__ == "__main__":

    
    runJobs()


