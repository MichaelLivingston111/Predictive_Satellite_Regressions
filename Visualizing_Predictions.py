
# Import all libraries:

import pandas as pd
import numpy as np
import netCDF4 as nc
import cartopy.crs as ccrs  # NEED TO USE CONDA
import cartopy.feature as cfeature
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import matplotlib.pyplot as plt
import matplotlib as mpl
import feather

# Import a file to obtain the necessary coordinates:
file = 'A20211292021136.L3m_8D_CHL_chlor_a_4km.nc'  # May 4-12 2021, Chlorophyll a data, 8 day, mapped
ds = nc.Dataset(file)

Lat = ds['lat'][:]  # Latitude
Lon = ds['lon'][:]  # Longitude

ds.close()

# Transform the variables/coordinates into individual dataframes:
Lat_df = pd.DataFrame(data=Lat)
Lon_df = pd.DataFrame(data=Lon)

# Subset the area of interest (Northeast Pacific): 40:60N, 122:155W
NE_Lat = pd.DataFrame(Lat_df[719:1199])
NE_Lon = pd.DataFrame(Lon_df[599:1391])


# Receive the prediction dataframe from R:
y_prediction = pd.read_feather('June_2019_Predictions.feather')

Predictions_map = pd.DataFrame(y_prediction)  # Create a dataframe of all the predictions
print(Predictions_map.shape)  # View the dataframe shape

# Reshape the coordinate variables in order to map them: Need to make 2D coordinate dfs in 1D array
Lat_map = np.array(NE_Lat)  # Change lat df to array
Lat_map = Lat_map.flatten()  # Change 2D array to 1D array

Lon_map = np.array(NE_Lon)  # Change lon df to array
Lon_map = Lon_map.flatten()  # Change 2D array to 1D array


# View the shape of the coordinates, and predicted values:
Predictions_map = np.array(Predictions_map)

print(Lat_map.shape)  # 1D array
print(Lon_map.shape)  # 1D array
print(Predictions_map.shape)  # 2D array, but not the right dimensions

# Reshape the predictions to match the lats/lons:
Predictions_map = Predictions_map.reshape(480, 792)  # Corrected dimensions

# Replace all negative predictions with zeros:
Predictions_map = np.where(Predictions_map < 0, 0, Predictions_map)


# Plot all the predictions:
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
ax.gridlines()
ct = ax.pcolormesh(Lon_map, Lat_map, Predictions_map, transform=ccrs.PlateCarree(), cmap="jet")  # Continuous color bar
plt.colorbar(ct, orientation="horizontal")
ax.set_xticks(np.arange(-155, -122, 5), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(40, 60, 5), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)
ax.add_feature(cfeature.COASTLINE)
ax.add_feature(cfeature.LAND, zorder=80, edgecolor='k', facecolor='silver')
ax.add_feature(cfeature.BORDERS)
ax.set_ylim([47, 56])
ax.set_xlim([-150, -122])


# Combine the predictions with the coordinates to validate the predictions with measured values:
TEP_Sq = pd.DataFrame(data=Predictions_map, index=NE_Lat.squeeze(), columns=NE_Lon.squeeze())

