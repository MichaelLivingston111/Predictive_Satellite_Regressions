# Import all libraries:

import pandas as pd
import numpy as np
import netCDF4 as nc
import cartopy.crs as ccrs  # NEED TO USE CONDA
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.mpl.geoaxes
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

# PLOT THE SATELLITE DATA FOR REFERENCE:

# Colormap details:
cmap = mpl.cm.cool  # Optional - can use "cool" instead of "jet"
norm = mpl.colors.Normalize(vmin=-6, vmax=6)  # Normalize the colors

# Plot the variables (chlorophyll a) a coordinates on a global scale:
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines()
ct = ax.pcolormesh(Lon, Lat, np.log(Chl), transform=ccrs.PlateCarree(), cmap="jet")  # Continuous color bar
plt.colorbar(ct, orientation="vertical")
ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, zorder=80, edgecolor='k', facecolor='silver')
ax.add_feature(cfeature.BORDERS)
ax.set_ylim([45, 60])
ax.set_xlim([-160, -120])

#######################################################################################################################

# CLEANING, INDEXING, AND SORTING THE SATELLITE DATA:

# Transform the variables/coordinates into individual dataframes:
Chl = pd.DataFrame(data=Chl)
Lat = pd.DataFrame(data=Lat)
Lon = pd.DataFrame(data=Lon)

# Drop NaN values from the chl data set:
Chl_x = Chl.fillna(0.0001)

# Subset the area of interest (Northeast Pacific): 40:60N, 122:155W
NE_Lat = pd.DataFrame(Lat[719:1199])
NE_Lon = pd.DataFrame(Lon[599:1391])

# Index out the chlorophyll data from the above selected coordinates:
NE_Chl = Chl_x.iloc[719:1199, 599:1391]

# Log all of the chlorophyll data:
NE_Chl_log = np.log(NE_Chl)

# Create an array from the matrix:
NE_final = np.array(NE_Chl_log)

# Create a dataframe with chlorophyll and the respective lats/lons:
Chl_Sq = pd.DataFrame(data=NE_final, index=NE_Lat.squeeze(), columns=NE_Lon.squeeze())

# Stack all the columns to create one series of logged chlorophyll values, with the respective lats/lons:
Chl_predict = Chl_Sq.stack(dropna=False)

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

# Vectorize the LR:
LR_V = np.vectorize(LR)

#######################################################################################################################

# APPLYING THE REGRESSION: Not finished...


# To do for this project:

# Create the linear regression algorithm in python
# Apply the model to the Northeast Pacific region
# Obtain POC values for the same region, and compare with model estimates
# Get the satellite values for the appropriate cruises and perform cross validations!

# Get the independent regressions for each season between TEP:Chl and TEP:Temperature - this could be used as
# justification in a model
