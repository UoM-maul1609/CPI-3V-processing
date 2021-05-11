CPI v1 processing
--
Based on PJC's version for v3 CPI data at https://github.com/UoM-maul1609/CPI-3V-processing/tree/master/matlab.
Restructured into three m-files: one for preprocessing of roi data to mat format, one to retrieve mat format data, and one to plot images

1. preproc_cpi.m: to preprocess a day's worth of data

- all preprocessing carried in single function
- takes inputs of filepath (directory where data stored) and date ('yyyy-mm-dd') to process data
- renames files in as YYYYMMDD_hhmm.roi so data from different years can be stored in single directory (roi file names are missing year designation as produced)
- produces .mat format for each roi file, plus timeseries file for whole day in same directory as roi data
- mat files are named YYYYMMDD_hhmm.mat and time series files are named YYYYMMDD_ts.mat

- default size bins are now quasi-logarithmic allowing finer graduations at small particle sizes and increasing graduations for largest particles
- sets foc parameter default as 20 as per Connolly et al 2007 (JTech) so sample size can be considered independent of particle size
- changed to process CPI v1 data (roi data blocks, version, imagetype and background identifiers are different between v3 and v1 roi data)
- v3 to v1 changes done by interrogating data files (without reference to documentation) 
- changed to calculate and process with CPI v1 sample area and volume when calculating time series concentrations, calculated from flowrate (l/min)
- if no background found / saved, for example if auto-generating PDS triggers, uses median value of image as dummy background
- additional commenting throughout

2. getcpidat.m: to retrieve preprocessed data for later use, plotting etc.

- takes input of filepath (directory where data stored) and dts, a 2-element vector of serial date numbers defining an interval of time on day in question
- data is only returned within this time window
- two structures are returned, the second one optionally
- first structure, cpidat, broadly is 'timeser' data, but with change of units to per cc, and conversion to dN/dd (units of per cc, per micron)
- second structure, cpiimg, contains images and associated measurement parameters (from ROI_N structure) and also from image processing (from STATS structure)

3. plotcpiimgs.m: to call getcpidat.m and plot particle images meeting foc and size criteria.

- lightly editted from PJC's exportimagesdriver.m
- takes input from getcpidat.m
- calculates figure window size from screensize and offsets new windows as needed
- new colourmap scale


A.R.D. Smedley | 15 Apr 2021
