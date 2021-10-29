# Import all libraries:

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import netCDF4 as nc
import cartopy.crs as ccrs  # NEED TO USE CONDA
import cartopy.feature as cfeature
from cartopy.mpl.geoaxes import GeoAxes
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter
import cartopy.mpl.geoaxes
import matplotlib.pyplot as plt


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

# Examine metadata for chlorophyll:
print(ds['chlor_a'])

# Define my variables:
Chl = ds['chlor_a'][:]  # Chlorophyll
Lat = ds['lat'][:]  # Latitude
Lon = ds['lon'][:]  # Longitude

# Close the file when not in use:
ds.close()

# Plot the variables (chlorophyll a) a coordinates on a global scale:
fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.coastlines()
ax.set_global()
ct = ax.contourf(Lon, Lat, np.log(Chl), transform=ccrs.PlateCarree(),
                 cmap="jet", vmax=3)
ax.gridlines()
cb = plt.colorbar(ct, orientation="vertical", extendrect='True')
ax.set_xticks(np.arange(-180, 181, 60), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(-90, 91, 30), crs=ccrs.PlateCarree())
lon_formatter = LongitudeFormatter(zero_direction_label=True)
lat_formatter = LatitudeFormatter()
ax.xaxis.set_major_formatter(lon_formatter)
ax.yaxis.set_major_formatter(lat_formatter)

