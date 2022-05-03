
# Import all required packages:

import pandas as pd
import numpy as np
import netCDF4 as nc
import feather

# LOAD AND EXAMINE THE DATA SET FROM NASA:

# Chlorophyll8 day, mapped:
# file = 'Spring_2019_LineP_8D_CHL_4km.nc'  # June 2th - 10th 2019
# file = '2019_LineP_DAY_CHL_4km.nc'  # Spring P4 verification
# file = 'May_2019_B7_CHL.nc'  # Spring B7 verification
# file = 'Spring_2019_LaPer_8D_CHL_4km.nc'  # May 25 - June 2 2019
# file = 'Summer_2019_LineP_8D_CHL_4km.nc'  # Summer 2019 8D Chl
# file = 'Summer_2019_LaPer_8D_CHL_4km.nc'  # Summer 2019 8D Chl LaPer
# file = 'May_2021_8D_CHL_4km.nc'  # May 1-9 2021
file = "Aug_2021_8D_CHL_4km.nc"  # August 26 - September 1st 2021:  GOOD
# file = 'Feb_2020_LineP_8D_CHL_4km.nc'  # Winter 2020
# file = ''  # Summer 2019 8D Chl LaPer
# file = 'LB_Line_Sept_2019_DAY_CHL_4km.nc'  # Summer LB line verification
ds = nc.Dataset(file)

# Temperature:
# file_t = '2019_LineP_DAY_SST_4km.nc'  # Spring 2019 P4 verification
# file_t = 'Spring_2019_LineP_8D_SST_4km.nc'  # June 2th - 10th 2019
# file_t = 'May_2019_B7_SST.nc'  # May 25 - June 2 2019
# file_t = 'Spring_2019_LaPer_8D_SST_4km.nc'  # May 25 - June 2 2019
# file_t = 'Feb_2020_LineP_8D_SST_4km.nc'  # # Winter 2020
# file_t = 'May_2021_8D_SST_4km.nc'  # May 1-9 2021
file_t = "Aug_2021_8D_SST.4km.nc"  # August 26 - September 1st
# file = 'LB_Line_Sept_2019_DAY_SST_4km.nc'  # Summer LB line verification
ds_t = nc.Dataset(file_t)

# POC:
# file_poc = '2019_LineP_DAY_POC_4km.nc'  # Spring 2019 P4 verification
# file_poc = 'Spring_2019_LineP_8D_POC_4km.nc'  # June 2th - 10th 2019
# file_poc = 'Spring_2019_LaPer_8D_POC_4km.nc'  # May 25 - June 2 2019
# file_poc = 'Feb_2020_LineP_8D_POC_4km.nc'  # Winter 2020
# file_poc = "May_2021_8D_POC_4km.nc"  # May 1-9 2021
file_poc = "Aug_2021_8D_POC_4km.nc"  # August 26 - September 1st
# file = 'LB_Line_Sept_2019_DAY_POC_4km.nc'  # Summer LB line verification
ds_poc = nc.Dataset(file_poc)

# ######################################################################################################################
# Define a function to clean and sort all the data before sending it to R. Returns a dataframe of all predictors
# variables indexed to lat/lon, and a R input file.


def clean(chl, temp, poc, season):

    # Takes inputs as netcdf files for chlorophyll, temperature and poc, and Season.
    # Chl, temp, and poc are all netcdf file inputs, while season needs to be a string
    # (options = Spring, Summer, Winter)

    file_chl = chl
    ds_chl = nc.Dataset(file_chl)

    file_temp = temp
    ds_temp = nc.Dataset(file_temp)

    file_poc = poc
    ds_poc = nc.Dataset(file_poc)

    # Define my variables:
    chl = pd.DataFrame(ds_chl['chlor_a'][:])  # Chlorophyll
    temp = pd.DataFrame(ds_temp['sst'][:])  # SST
    poc = pd.DataFrame(ds_poc['poc'][:])  # POC
    lat = pd.DataFrame(ds_chl['lat'][:])  # Latitude
    lon = pd.DataFrame(ds_chl['lon'][:])  # Longitude

    # Close the file when not in use:
    ds_chl.close()
    ds_temp.close()
    ds_poc.close()

    # Subset the area of interest (Northeast Pacific): 40:60N, 122:155W
    # ne_lat = pd.DataFrame(lat[719:1199])
    # ne_lon = pd.DataFrame(lon[599:1391])

    # Index out the data from the area of interest:
    # ne_chl = chl.iloc[719:1199, 599:1391]
    # ne_temp = temp.iloc[719:1199, 599:1391]
    # ne_poc = poc.iloc[719:1199, 599:1391]

    # Log all of the chlorophyll data:
    # ne_chl_log = np.log(ne_chl)
    chl_log = np.log(chl)

    # Create an array from the logged chlorophyll, POC, temperature matrices:
    ne_final = np.array(chl_log)
    sst_final = np.array(temp)
    poc_final = np.array(poc)

    # Create a dataframe with chlorophyll, temperature, POC and the respective lats/lons, and
    # stack all the columns to create one series of logged chlorophyll and SST values, with the respective lats/lons:
    chl_predict = pd.DataFrame(data=ne_final, index=lat.squeeze(), columns=lon.squeeze()).stack(dropna=False)
    sst_predict = pd.DataFrame(data=sst_final, index=lat.squeeze(), columns=lon.squeeze()).stack(dropna=False)
    poc_predict = pd.DataFrame(data=poc_final, index=lat.squeeze(), columns=lon.squeeze()).stack(dropna=False)

    # Combine stacked predictor columns:
    predictors = pd.concat([chl_predict, sst_predict, poc_predict], axis=1)

    # Add a categorical column for the appropriate season (used as a random effect in the model):
    predictors['Season'] = season

    # Add a categorical column for Arctic regions (> index 6180000):
    predictors['Arctic'] = 0
    predictors.iloc[0:6180000, "Arctic"] = 1

    # Rename the columns:
    predictors.columns = ['Log_Chl', 'Temperature', 'POC', 'Season', 'Arctic']

    # Provide a dataframe that can be imported into R:
    r_input = predictors.reset_index(drop=True)

    return predictors, r_input


#######################################################################################################################

# Cleaned data to R:

Summer_2021_cleaned = clean("Aug_2021_8D_CHL_4km.nc", "Aug_2021_8D_SST.4km.nc", "Aug_2021_8D_POC_4km.nc", "Summer")
Summer_2021_cleaned[1].to_feather('Aug_2021_LineP.feather')


#######################################################################################################################










# CLEANING, INDEXING, AND SORTING THE SATELLITE DATA:

# Transform the variables/coordinates into individual dataframes:
#Chl_df = pd.DataFrame(data=Chl)
#Lat_df = pd.DataFrame(data=Lat)
#Lon_df = pd.DataFrame(data=Lon)

#Temp_df = pd.DataFrame(data=Temp)
#POC_df = pd.DataFrame(data=POC)

# Subset the area of interest (Northeast Pacific): 40:60N, 122:155W
#NE_Lat = pd.DataFrame(Lat_df[719:1199])
#NE_Lon = pd.DataFrame(Lon_df[599:1391])

# Index out the data from the area of interest:
#NE_Chl = Chl_df.iloc[719:1199, 599:1391]
#NE_Temp = Temp_df.iloc[719:1199, 599:1391]
#NE_POC = POC_df.iloc[719:1199, 599:1391]


# Log all of the chlorophyll data:
#NE_Chl_log = np.log(NE_Chl)

# Create an array from the logged chlorophyll, POC, temperature matrices:
#NE_final = np.array(NE_Chl_log)
#SST_final = np.array(NE_Temp)
#POC_final = np.array(NE_POC)


# Create a dataframe with chlorophyll, temperature, POC and the respective lats/lons:
#Chl_Sq = pd.DataFrame(data=NE_final, index=NE_Lat.squeeze(), columns=NE_Lon.squeeze())
#SST_Sq = pd.DataFrame(data=SST_final, index=NE_Lat.squeeze(), columns=NE_Lon.squeeze())
#POC_Sq = pd.DataFrame(data=POC_final, index=NE_Lat.squeeze(), columns=NE_Lon.squeeze())


# Stack all the columns to create one series of logged chlorophyll and SST values, with the respective lats/lons:
#Chl_predict = Chl_Sq.stack(dropna=False)  # This is our main predictor value.
#SST_predict = SST_Sq.stack(dropna=False)
#POC_predict = POC_Sq.stack(dropna=False)  # This is not a predictor, but used for comparison!

# Combine the two stacked predictor columns (SST and Chlor a):
#Predictors = pd.concat([Chl_predict, SST_predict, POC_predict], axis=1)

# Add a categorical column for the appropriate season (used as a random effect in the model):
#Predictors['Season'] = 'Summer'

# Rename the columns:
#Predictors.columns = ['Log_Chl', 'Temperature', 'POC', 'Season']


# Import the dataframe into R:
#R_input = Predictors.reset_index(drop=True)
# R_input.to_feather('May_2019_B7_day_LineP.feather')
# R_input.to_feather('May_2019_LaPer.feather')
# R_input.to_feather('June_2019_P4_day_LineP.feather')
# R_input.to_feather('June_2019_LineP.feather')
# R_input.to_feather('Aug_2019_LineP.feather')
# R_input.to_feather('May_2021_LineP.feather')
#R_input.to_feather('Aug_2021_LineP.feather')
# R_input.to_feather('Sept_2019_LaPer.feather')
# R_input.to_feather('Feb_2020_LineP.feather')
#R_input.to_feather('2019_LBline_DAY.feather')
