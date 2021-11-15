# Import all required packages:

import pandas as pd
import numpy as np
import netCDF4 as nc
import cartopy.crs as ccrs  # NEED TO USE CONDA
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import matplotlib as mpl
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
import statsmodels.formula.api as smf
import scipy.stats as stats
from statsmodels.stats.diagnostic import het_white

#######################################################################################################################

# LOAD AND EXAMINE THE DATA SET FROM NASA: (Note: some of this code is unnecessary for the actual model,
# but is helpful in visualizing and examining the data set)

# Load the NetCDF dataset from NASA:
# file = 'A20191612019168.L3m_8D_CHL_chlor_a_4km.nc'  # June 6th - 14th 2019 Chlorophyll a data, 8 day, mapped
file = 'A20211292021136.L3m_8D_CHL_chlor_a_4km.nc'  # May 4-12 2021, Chlorophyll a data, 8 day, mapped
ds = nc.Dataset(file)

# Load the Temperature NetCDF file: June 6th - 14th 2019 Temperature data, 8 day, mapped
# file_t = 'AQUA_MODIS.20190610_20190617.L3m.8D.SST.sst.4km.nc'
file_t = 'AQUA_MODIS.20210509_20210516.L3m.8D.SST.sst.4km.nc'  # May 4-12 2021, SST data, 8 day, mapped
ds_t = nc.Dataset(file_t)

# Load the POC NetCDF file: Sept 6th - 14th 2021 POC data, 8 day, mapped
file_poc = "May_2021_8D_POC4km.nc"  # May 4-12 2021, POC, 8 day, mapped
ds_poc = nc.Dataset(file_poc)

# NetCDF files have three parts: metadata, dimensions and variables. Variables contain both data and metadata.
# netcdf4 allow sus to access all of this.

# Print the dataset: get info about the variables in the file and their dimensions:
print(ds)
print(ds_t)
print(ds_poc)

# We can access information on a variety of things, bust most notably file format, data source, data version,
# citation, dimensions and variables.

# Metadata can also be accessed as a Python dictionary (similar to the code in my core algorithm code):
print(ds.__dict__)  # whole dictionary
print(ds.__dict__['temporal_range'])  # data instrument
print(ds.__dict__['spatialResolution'])  # Resolution

# Access all the dimensions by looping through all available dimensions:
for dim in ds.dimensions.values():
    print(dim)

# Define my variables:
Chl = ds['chlor_a'][:]  # Chlorophyll
Lat = ds['lat'][:]  # Latitude
Lon = ds['lon'][:]  # Longitude

Temp = ds_t['sst'][:]  # SST
POC = ds_poc['poc'][:]  # POC

# Close the file when not in use:
ds.close()
ds_t.close()

#######################################################################################################################

# CLEANING, INDEXING, AND SORTING THE SATELLITE DATA:

# Transform the variables/coordinates into individual dataframes:
Chl_df = pd.DataFrame(data=Chl)
Lat_df = pd.DataFrame(data=Lat)
Lon_df = pd.DataFrame(data=Lon)

Temp_df = pd.DataFrame(data=Temp)
POC_df = pd.DataFrame(data=POC)

# Drop NaN values from data sets, and fill it with data points that can be logged and/or predicted
# in a model:
Temp_x = Temp_df.fillna(0.0001)
Chl_x = Chl_df.fillna(0.0001)
POC_x = POC_df

# Subset the area of interest (Northeast Pacific): 40:60N, 122:155W
NE_Lat = pd.DataFrame(Lat_df[719:1199])
NE_Lon = pd.DataFrame(Lon_df[599:1391])

# Index out the data from the area of interest:
NE_Chl = Chl_x.iloc[719:1199, 599:1391]
NE_Temp = Temp_x.iloc[719:1199, 599:1391]
NE_POC = POC_x.iloc[719:1199, 599:1391]

# Log all of the chlorophyll data:
NE_Chl_log = np.log(NE_Chl)

# Create an array from the logged chlorophyll, POC, temperature matrices:
NE_final = np.array(NE_Chl_log)
SST_final = np.array(NE_Temp)
POC_final = np.array(NE_POC)

# Create a dataframe with chlorophyll, temperature, POC and the respective lats/lons:
Chl_Sq = pd.DataFrame(data=NE_final, index=NE_Lat.squeeze(), columns=NE_Lon.squeeze())
SST_Sq = pd.DataFrame(data=SST_final, index=NE_Lat.squeeze(), columns=NE_Lon.squeeze())
POC_Sq = pd.DataFrame(data=POC_final, index=NE_Lat.squeeze(), columns=NE_Lon.squeeze())

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

# Now, I can apply the algorithm to predict values for each independent data point from the Satellite! But first,
# I need to create the regression algorithm (below).

#######################################################################################################################

# CREATING THE REGRESSION ALGORITHM: Linear mixed effects model

# Import the raw dataset:
raw_data = pd.read_csv("Total_Sat_Data.csv")

# CREATE A LINEAR MIXED EFFECTS MODEL USING THE ENTIRE DATA SET:
model = smf.mixedlm("TEP ~ Log_Chl + Temperature", raw_data, groups=raw_data["Season"])
mdf = model.fit()
print(mdf.summary())

#######################################################################################################################

# APPLYING THE REGRESSION:

# MAKE THE PREDICTIONS:
y_prediction = mdf.predict(Predictors)
print(y_prediction.shape)  # View shape of the intended predictions

# Reshape the predictions into 2 dimensions in order for the model to operate on it:
Predictions_map = pd.DataFrame(y_prediction)  # Create a dataframe of all the predictions
print(Predictions_map.shape)  # View the dataframe shape

#######################################################################################################################

# RESHAPE COORDINATE VARIABLES:

# View the shape of each variable for reference: Need to change dimensions.
print(NE_Chl_log.shape)  # 2D array: This is required as it is mapped to both arrays below
print(NE_Lat.shape)  # 2D array: needs to be 1D
print(NE_Lon.shape)  # 2D array: needs to be 1D
print(NE_Temp.shape)  # 2D array: This is required as it is mapped to both arrays above

# Reshape the coordinate variables in order to map them: Need to make 2D coordinate dfs in 1D array
Lat_map = np.array(NE_Lat)  # Change lat df to array
Lat_map = Lat_map.flatten()  # Change 2D array to 1D array

Lon_map = np.array(NE_Lon)  # Change lon df to array
Lon_map = Lon_map.flatten()  # Change 2D array to 1D array

# Reprint the shapes to check: Valid.
print(NE_Chl_log.shape)  # Needs to remain a 2D array
print(Lat_map.shape)  # 1D array
print(Lon_map.shape)  # 1D array

#######################################################################################################################

# MAP ALL THE PREDICTIONS:

# View the shape of the coordinates, and predicted values:
Predictions_map = np.array(Predictions_map)

print(Lat_map.shape)  # 1D array
print(Lon_map.shape)  # 1D array
print(Predictions_map.shape)  # 2D array, but not the right dimensions

# Reshape the predictions to match the lats/lons:
Predictions_map = Predictions_map.reshape(480, 792)  # Corrected dimensions

# Replace all original NaN values with NaNs:
Predictions_map = np.where(Predictions_map < -387, np.nan, Predictions_map)  # -388.32414 = the given Nan value,
# reinstate the Nans. These were originally removed in order for the regression algorithm to make predictions.


# Replace all negative predictions with zeros:
Predictions_map = np.where(Predictions_map < 0, 0, Predictions_map)

# Colormap details: Optional
cmap = mpl.cm.cool  # Optional - can use "cool" instead of "jet"
norm = mpl.colors.Normalize(vmin=-6, vmax=6)  # Normalize the colors

# Plot all the predictions:
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines()
ct = ax.pcolormesh(Lon_map, Lat_map, Predictions_map, transform=ccrs.PlateCarree(), cmap="jet")  # Continuous color bar
plt.colorbar(ct, orientation="vertical")
ax.set_xticks(np.arange(-155, -122, 5), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(40, 60, 5), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, zorder=80, edgecolor='k', facecolor='silver')
ax.add_feature(cfeature.BORDERS)
ax.set_ylim([40, 60])
ax.set_xlim([-155, -122])
# ax.set_ylim([47.9, 50.6])
# ax.set_xlim([-130, -123])

#######################################################################################################################

# PLOT THE PREDICTOR VARIABLES FOR REFERENCE:

# Plot the variables (chlorophyll a) (Northeast Pacific):
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines()
ct = ax.pcolormesh(Lon_map, Lat_map, NE_Chl_log, transform=ccrs.PlateCarree(), cmap="jet")  # Continuous color bar
plt.colorbar(ct, orientation="vertical")
# ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
# ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, zorder=80, edgecolor='k', facecolor='silver')
ax.add_feature(cfeature.BORDERS)

# Plot the variables (SST) (Northeast Pacific):
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines()
ct = ax.pcolormesh(Lon_map, Lat_map, NE_Temp, transform=ccrs.PlateCarree(), cmap="jet")  # Continuous color bar
plt.colorbar(ct, orientation="vertical")
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, zorder=80, edgecolor='k', facecolor='silver')
ax.add_feature(cfeature.BORDERS)

# Plot the variables (POC) (Northeast Pacific):
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines()
ct = ax.pcolormesh(Lon_map, Lat_map, NE_POC, transform=ccrs.PlateCarree(), cmap="jet")  # Continuous color bar
plt.colorbar(ct, orientation="vertical")
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, zorder=80, edgecolor='k', facecolor='silver')
ax.add_feature(cfeature.BORDERS)

#######################################################################################################################

# Now, I need to calculate the fraction of the total POC pool that my predicted values make up. Need ot divide the
# total prediction matrix by the total POC matrix, adn plot the results!

print(Predictions_map.shape)  # Predictions
print(NE_POC.shape)  # POC


# Confirmed both matrices (arrays) are the sane dimensions. Divide predictions by POC:
TEPC_map = np.divide(Predictions_map, NE_POC)
TEPC_map = TEPC_map * 100


# Replace all values >100 with 100:
TEPC_map = np.where(TEPC_map > 100, 100, TEPC_map)


# Plot the results:
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines()
ct = ax.pcolormesh(Lon_map, Lat_map, TEPC_map, transform=ccrs.PlateCarree(), cmap="jet")  # Continuous color bar
plt.colorbar(ct, orientation="vertical")
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, zorder=80, edgecolor='k', facecolor='silver')
ax.add_feature(cfeature.BORDERS)

#######################################################################################################################

# Combine the prediction plot and the ratio plot:

fig = plt.figure(figsize=(12, 6))

# Prediction plot:
ax = fig.add_subplot(1, 2, 1, projection=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines()
ct = ax.pcolormesh(Lon_map, Lat_map, Predictions_map, transform=ccrs.PlateCarree(), cmap="jet")  # Continuous color bar
plt.colorbar(ct, orientation="vertical")
ax.set_xticks(np.arange(-155, -122, 5), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(40, 60, 5), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, zorder=80, edgecolor='k', facecolor='silver')
ax.add_feature(cfeature.BORDERS)
ax.set_ylim([40, 60])
ax.set_xlim([-155, -122])

# Ratio plot:
ax = fig.add_subplot(1, 2, 2, projection=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines()
ct = ax.pcolormesh(Lon_map, Lat_map, TEPC_map, transform=ccrs.PlateCarree(), cmap="jet")  # Continuous color bar
plt.colorbar(ct, orientation="vertical")
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, zorder=80, edgecolor='k', facecolor='silver')
ax.add_feature(cfeature.BORDERS)




#######################################################################################################################
# To do for this project:

# Input temperature (SST) data from the exact same period
# Index the SST data in the same way, and merge it with Chl_predict_x
# Merge a category 'Spring' to each data point
# Create and apply a linear mixed effects model to the satellite data

# Get the satellite data for the spring seasons 2012 - 2017.
# Get the satellite data for the 2019, 2021 cruises and use it as a cross validation.
# Obtain POC values for the same region, and compare with model estimates
# Get the satellite values for the appropriate cruises and perform cross validations!

# Get the independent regressions for each season between TEP:Chl and TEP:Temperature - this could be used as
# justification in a model
