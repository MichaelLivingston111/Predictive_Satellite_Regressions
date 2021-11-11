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

#######################################################################################################################

# LOAD AND EXAMINE THE DATA SET FROM NASA: (Note: some of this code is unnecessary for the actual model,
# but is helpful in visualizing and examining the data set)

# Load the NetCDF dataset from NASA: Sept 6th - 14th 2021 Chlorophyll a data
file = 'A20212492021256.L3m_8D_CHL_chlor_a_4km.nc'  # name the file
ds = nc.Dataset(file)

# NetCDF files have three parts: metadata, dimensions and variables. Variables contain both data and metadata.
# netcdf4 allow sus to access all of this.

# Print the dataset: get info about the variables in the file and their dimensions:
print(ds)

# We can access information on a variety of things, bust most notably file format, data source, data version,
# citation, dimensions and variables.

# Metadata can also be accessed as a Python dictionary (similar to the code in my core algorithm code):
print(ds.__dict__)  # whole dictionary
print(ds.__dict__['temporal_range'])  # data instrument
print(ds.__dict__['spatialResolution'])  # Resolution

# Access all the dimensions by looping through all available dimensions:
for dim in ds.dimensions.values():
    print(dim)

# Access variable metadata in a similar fashion:
for var in ds.variables.values():
    print(var)

# Examine metadata for just chlorophyll:
print(ds['chlor_a'])

# Define my variables:
Chl = ds['chlor_a'][:]  # Chlorophyll
Lat = ds['lat'][:]  # Latitude
Lon = ds['lon'][:]  # Longitude

# Close the file when not in use:
ds.close()

#######################################################################################################################

# CLEANING, INDEXING, AND SORTING THE SATELLITE DATA:

# Transform the variables/coordinates into individual dataframes:
Chl_df = pd.DataFrame(data=Chl)
Lat_df = pd.DataFrame(data=Lat)
Lon_df = pd.DataFrame(data=Lon)

# Drop NaN values from the chl data set, and fill it with data points that can be logged and predicted in a model:
Chl_x = Chl_df.fillna(0.0001)

# Subset the area of interest (Northeast Pacific): 40:60N, 122:155W
NE_Lat = pd.DataFrame(Lat_df[719:1199])
NE_Lon = pd.DataFrame(Lon_df[599:1391])

# Index out the chlorophyll data from the area of interest:
NE_Chl = Chl_x.iloc[719:1199, 599:1391]

# Log all of the chlorophyll data:
NE_Chl_log = np.log(NE_Chl)

# Create an array from the logged chlorophyll matrix:
NE_final = np.array(NE_Chl_log)

# Create a dataframe with chlorophyll and the respective lats/lons:
Chl_Sq = pd.DataFrame(data=NE_final, index=NE_Lat.squeeze(), columns=NE_Lon.squeeze())

# Stack all the columns to create one series of logged chlorophyll values, with the respective lats/lons:
Chl_predict = Chl_Sq.stack(dropna=False)  # This is our main predictor value.


# Now, I can apply the algorithm to predict values for each independent data point from the Satellite! But first,
# I need to create the regression algorithm (below).

#######################################################################################################################

# CREATING THE REGRESSION ALGORITHM

# Import the raw dataset:
raw_data = pd.read_csv("Total_data.csv")

# Create the training data sets (using the entire data set, validation has been performed in a separate module):
chlor_a = raw_data[['Log_Chl']]  # Predictor
TEP = raw_data[['TEP']]  # Target

# Create object of LinearRegression class:
LR = LinearRegression()

# Fit the training data:
LR.fit(chlor_a, TEP)  # The model has now been trained on the training data!

#######################################################################################################################

# APPLYING THE REGRESSION:

print(Chl_predict.shape)  # View shape of the intended predictions
print(chlor_a.shape)  # View shape of the predictor that the model was trained on

# Reshape the predictions into 2 dimensions in order for the model to operate on it:
Chl_predict_x = pd.DataFrame(Chl_predict)  # Create a dataframe
print(Chl_predict_x.shape)

# MAKE THE PREDICTIONS:
Predictions = LR.predict(Chl_predict_x)

#######################################################################################################################

# PLOT THE SATELLITE CHLOROPHYLL DATA FOR REFERENCE: THE NORTHEAST PACIFIC

# View the shape of each variable for reference: Need to change dimensions.
print(NE_Chl_log.shape)  # 2D array: This is required as it is mapped to both arrays below
print(NE_Lat.shape)  # 2D array: needs to be 1D
print(NE_Lon.shape)  # 2D array: needs to be 1D


# Reshape the variables in order to map them: Need to make 2D coordinate dfs in 1D array
Lat_map = np.array(NE_Lat)  # Change lat df to array
Lat_map = Lat_map.flatten()  # Change 2D array to 1D array

Lon_map = np.array(NE_Lon)  # Change lon df to array
Lon_map = Lon_map.flatten()  # Change 2D array to 1D array


# Reprint the shapes to check: Valid.
print(Chl_df.shape)  # Needs to remain a 2D array
print(Lat_map.shape)  # 1D array
print(Lon_map.shape)  # 1D array


# Colormap details: Optional
cmap = mpl.cm.cool  # Optional - can use "cool" instead of "jet"
norm = mpl.colors.Normalize(vmin=-6, vmax=6)  # Normalize the colors

# Plot the variables (chlorophyll a) on for the new variables (Northeast Pacific):
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


#######################################################################################################################

# MAP ALL THE PREDICTIONS:

# View the shape of the coordinates, and predicted values

print(Lat_map.shape)  # 1D array
print(Lon_map.shape)  # 1D array
print(Predictions.shape)  # 2D array, but not the right dimensions


# Reshape the predictions to match the lats/lons:
Predictions_map = Predictions.reshape(480, 792)  # Corrected dimensions


# Replace all original NaN values with NaNs:
Predictions_map = np.where(Predictions_map < -387, np.nan, Predictions_map)  # -388.32414 = the given Nan value,
# reinstate the Nans. These were originally removed in order for the regression algorithm to make predictions.

# Replace all negative predictions with zeros:
Predictions_map = np.where(Predictions_map < 0, 0, Predictions_map)


# Plot the variables (chlorophyll a) on for the new variables (Northeast Pacific):
fig = plt.figure(figsize=(10, 4))
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
