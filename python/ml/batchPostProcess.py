foc_crit=5 # critical value of focus for an image
min_len=50 # minimum length for particle images

post_process=True

path1=['/models/mccikpc2/DCMEX/C296/' , \
       '/models/mccikpc2/DCMEX/C299/' , \
       '/models/mccikpc2/DCMEX/C300/' , \
       '/models/mccikpc2/DCMEX/C301/' , \
       '/models/mccikpc2/DCMEX/C302/' , \
       '/models/mccikpc2/DCMEX/C303/' , \
       '/models/mccikpc2/DCMEX/C304/' , \
       '/models/mccikpc2/DCMEX/C305/' , \
       '/models/mccikpc2/DCMEX/C306/' , \
       '/models/mccikpc2/DCMEX/C307/' , \
       '/models/mccikpc2/DCMEX/C308/' , \
       '/models/mccikpc2/DCMEX/C309/' , \
       '/models/mccikpc2/DCMEX/C310/' , \
       '/models/mccikpc2/DCMEX/C311/' , \
       '/models/mccikpc2/DCMEX/C312/' , \
       '/models/mccikpc2/DCMEX/C313/' , \
       '/models/mccikpc2/DCMEX/C314/' , \
       '/models/mccikpc2/DCMEX/C315/' ]

outputfile='/models/mccikpc2/DCMEX/CPI-analysis/postProcessed_t5_l50.h5'



def runJobs():
    global path1
    #++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++
    if post_process:
        from postProcessingDriver import postProcessingDriver
        # extract and post process all images and save to file
        postProcessingDriver(path1,outputfile,foc_crit,min_len,5)
    #--------------------------------------------------------------------------
    

if __name__ == "__main__":

    
    runJobs()


