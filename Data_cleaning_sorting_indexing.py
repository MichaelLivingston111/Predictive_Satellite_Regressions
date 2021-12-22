
# Import all required packages:

import pandas as pd
import numpy as np
import netCDF4 as nc
import feather

# LOAD AND EXAMINE THE DATA SET FROM NASA:

# Chlorophyll:
file = 'A20191612019168.L3m_8D_CHL_chlor_a_4km.nc'  # June 6th - 14th 2019 Chlorophyll a data, 8 day, mapped
# file = 'A20211292021136.L3m_8D_CHL_chlor_a_4km.nc'  # May 4-12 2021, Chlorophyll a data, 8 day, mapped
# file = "Aug_2021_8D_CHL_4km.nc"  # August 26 - September 1st 2021:  GOOD
ds = nc.Dataset(file)

# Temperature:
file_t = 'AQUA_MODIS.20190610_20190617.L3m.8D.SST.sst.4km.nc'  # June 6th - 14th 2019 temperature data, 8 day, mapped
# file_t = 'AQUA_MODIS.20210509_20210516.L3m.8D.SST.sst.4km.nc'  # May 4-12 2021, SST data, 8 day, mapped
# file_t = "Aug_2021_8D_SST.4km.nc"  # August 26 - September 1st
ds_t = nc.Dataset(file_t)

# POC:
file_poc = "June_2019_8D_POC_4km.nc"  # June 6th - 14th 2019 POC data, 8 day, mapped
# file_poc = "May_2021_8D_POC4km.nc"  # May 4-12 2021, POC, 8 day, mapped
# file_poc = "Aug_2021_8D_POC_4km.nc"  # August 26 - September 1st
ds_poc = nc.Dataset(file_poc)


# Define my variables:
Chl = ds['chlor_a'][:]  # Chlorophyll
Lat = ds['lat'][:]  # Latitude
Lon = ds['lon'][:]  # Longitude

Temp = ds_t['sst'][:]  # SST
POC = ds_poc['poc'][:]  # POC

# Close the file when not in use:
ds.close()
ds_t.close()
ds_poc.close()


# CLEANING, INDEXING, AND SORTING THE SATELLITE DATA:

# Transform the variables/coordinates into individual dataframes:
Chl_df = pd.DataFrame(data=Chl)
Lat_df = pd.DataFrame(data=Lat)
Lon_df = pd.DataFrame(data=Lon)

Temp_df = pd.DataFrame(data=Temp)
POC_df = pd.DataFrame(data=POC)

# Subset the area of interest (Northeast Pacific): 40:60N, 122:155W
NE_Lat = pd.DataFrame(Lat_df[719:1199])
NE_Lon = pd.DataFrame(Lon_df[599:1391])

# Index out the data from the area of interest:
NE_Chl = Chl_df.iloc[719:1199, 599:1391]
NE_Temp = Temp_df.iloc[719:1199, 599:1391]
NE_POC = POC_df.iloc[719:1199, 599:1391]

# Log all of the chlorophyll data:
NE_Chl_log = np.log(NE_Chl)

# Create a dataframe with chlorophyll, temperature, POC and the respective lats/lons:
Chl_Sq = pd.DataFrame(data=NE_Chl_log, index=NE_Lat.squeeze(), columns=NE_Lon.squeeze())
SST_Sq = pd.DataFrame(data=NE_Temp, index=NE_Lat.squeeze(), columns=NE_Lon.squeeze())
POC_Sq = pd.DataFrame(data=NE_POC, index=NE_Lat.squeeze(), columns=NE_Lon.squeeze())

# Stack all the columns to create one series of logged chlorophyll and SST values, with the respective lats/lons:
Chl_predict = Chl_Sq.stack(dropna=False)  # This is our main predictor value.
SST_predict = SST_Sq.stack(dropna=False)
POC_comp = POC_Sq.stack(dropna=False)  # This is not a predictor, but used for comparison!

# Combine the two stacked predictor columns (SST and Chlor a):
Predictors = pd.concat([Chl_predict, SST_predict], axis=1)

# Add a categorical column for the appropriate season (used as a random effect in the model):
Predictors['Season'] = 'Spring'

# Rename the columns:
Predictors.columns = ['Log_Chl', 'Temperature', 'Season']


# Import the dataframe into R:
R_input = Predictors.reset_index(drop=True)
R_input.to_feather('June_2019_LineP.feather')
# R_input.to_feather('May_2021_LineP.feather')
# R_input.to_feather('Aug_2021_LineP.feather')