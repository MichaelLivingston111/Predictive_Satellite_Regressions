# Import all libraries:
import pandas as pd
import numpy as np
import netCDF4 as nc
import cartopy.crs as ccrs  # NEED TO USE CONDA
import cartopy.feature as cfeature
import matplotlib.pyplot as plt
import cmocean as cm
import feather
from adjustText import adjust_text

#######################################################################################################################

# Formatting all coordinates:
# Import a file to obtain the necessary coordinates: Use these as default coordinates
file = 'A20211292021136.L3m_8D_CHL_chlor_a_4km.nc'  # May 4-12 2021 coordinates
ds = nc.Dataset(file)

Lat = ds['lat'][:]  # Latitude
Lon = ds['lon'][:]  # Longitude

ds.close()

# Transform the variables/coordinates into individual dataframes:
Lat_df = pd.DataFrame(data=Lat)
Lon_df = pd.DataFrame(data=Lon)


#######################################################################################################################

# Define a function to clean and map prediction data: Global (includes Arctic regions)

# prediction_feather is the derived predictions from R in the associated feather format.
# lat_df and lon_df are the coordinates from the satellite data frame (global size).
# lat_min, lat_max etc. is the range of the prediction map you want to plot.
# proj is the projection style you wish to use.


def map_predict_global(prediction_feather, lat_df, lon_df, lat_min, lat_max, lon_min, lon_max, proj):

    # Read prediction file from R:
    y_predictions = pd.read_feather(prediction_feather)

    # Remove the POC variable, but return for use later:
    poc = y_predictions.pop("POC")

    # Create Array:
    predictions_map = np.array(pd.DataFrame(y_predictions))

    # Reshape the coordinate variables in order to map them: Need to make 2D coordinate dfs in 1D array
    lat_map = np.array(lat_df).flatten()
    lon_map = np.array(lon_df).flatten()

    # Reshape the predictions to match the lat/lon, remove any negative values::
    predictions_map = predictions_map.reshape(4320, 8640)  # Corrected global dimensions based on prediction df
    predictions_map = np.where(predictions_map < 0, 5, predictions_map)  # Remove negative, replace with detection limit

    # PLOT ALL PREDICTIONS
    points = pd.read_csv("Lats_lons_Arctic.csv")
    points = proj.transform_points(ccrs.PlateCarree(), np.array(points.Station_lon), np.array(points.Station_lat))

    fig1 = plt.figure(figsize=(10, 9))
    ax1 = fig1.add_subplot(1, 1, 1, projection=proj)  # Create the geoaxes for an orthographic projection
    ct1 = ax1.pcolormesh(lon_map, lat_map, predictions_map, transform=ccrs.PlateCarree(),  # Color predictions
                         cmap=cm.cm.haline, vmax=160)  # Continuous color bar
    plt.colorbar(ct1, orientation="horizontal", fraction=0.02, pad=0.03)
    ax1.plot(points[:, 0], points[:, 1], 'ro', markeredgecolor='r', alpha=0.6, ms=9)  # Plot points on an orthographic projection
    ax1.set_ylabel("Latitude ˚N")  # Labels
    ax1.set_xlabel("Longitude ˚W")
    ax1.set_title("Estimated TEP concentrations from a multivariate regression model \n \n TEP ~ Chlorophyll a + POC + "
                  "Temperature + Region")
    ax1.coastlines()
    # ax.gridlines(color='gray', linestyle='--')  # optional
    ax1.add_feature(cfeature.LAND, zorder=1, edgecolor='k', facecolor='silver')  # Features
    ax1.add_feature(cfeature.COASTLINE, zorder=1, linewidth=0.3)

    return fig1, poc, predictions_map, lon_map, lat_map


#######################################################################################################################

# Define a function to clean and map prediction data: Regional

# prediction_feather is the derived predictions from R in the associated feather format.
# lat_df and lon_df are the coordinates from the satellite data frame (global size).
# lat_min, lat_max etc. is the range of the prediction map you want to plot.
# proj is the projection style you wish to use.


def map_predict_regional(prediction_feather, lat_df, lon_df, lat_min, lat_max, lon_min, lon_max, proj):
    y_predictions = pd.read_feather(prediction_feather)  # Read prediction csv file from R

    poc = y_predictions.pop("POC")  # Remove poc for now

    predictions_map = np.array(pd.DataFrame(y_predictions))  # Create Array

    # Reshape the coordinate variables in order to map them: Need to make 2D coordinate dfs in 1D array
    lat_map = np.array(lat_df).flatten()
    lon_map = np.array(lon_df).flatten()

    # Reshape the predictions to match the lat/lon, remove any negative values::
    predictions_map = predictions_map.reshape(4320, 8640)  # Corrected global dimensions based on prediction df
    predictions_map = np.where(predictions_map < 0, 5, predictions_map)  # Remove negative, replace with detection limit

    # PLOT ALL PREDICTIONS
    points = pd.read_csv("Lats_lons_Arctic.csv")

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(1, 1, 1, projection=proj)  # Create the geoaxes for a specified projection
    ct = ax.pcolormesh(lon_map, lat_map, predictions_map, transform=ccrs.PlateCarree(),  # Color predictions
                       cmap=cm.cm.thermal, vmax=160)  # Continuous color bar
    plt.colorbar(ct, orientation="vertical", fraction=0.02, pad=0.03)
    plt.scatter(x=points.Station_lon, y=points.Station_lat, facecolors='none', edgecolors='red', s=12)
    ax.set_ylabel("Latitude ˚N")  # Labels
    ax.set_xlabel("Longitude ˚W")
    ax.set_title("TEP (µg XG eq/L)")
    ax.coastlines()
    # ax.gridlines(color='gray', linestyle='--')  # optional
    ax.add_feature(cfeature.LAND, zorder=1, edgecolor='k', facecolor='silver')  # Features
    ax.add_feature(cfeature.COASTLINE, zorder=1, linewidth=0.3)
    ax.set_ylim([lat_min, lat_max])  # Limits
    ax.set_xlim([lon_min, lon_max])

    return fig, poc, predictions_map, lon_map, lat_map


#######################################################################################################################

# APPLY THE PREDICTION MAPPING FUNCTIONS:

# Prediction maps: 8 day averages
Summer_2021 = map_predict_regional('Aug_Sept_2021_Predictions.feather', Lat_df, Lon_df, 46, 56, -146, -123,
                                   proj=ccrs.PlateCarree())

Spring_2021 = map_predict_regional('May_2021_Predictions.feather', Lat_df, Lon_df, 46, 56, -146, -123,
                                   proj=ccrs.PlateCarree())

# Prediction maps: 8 day averages, Arctic coverage
Summer_2021_Total = map_predict_global('Aug_Sept_2021_Predictions_Global.feather', Lat_df, Lon_df, 45, 90, -172, -123,
                                       ccrs.NearsidePerspective(central_longitude=-145.0, central_latitude=58.0,
                                                                satellite_height=500000))


# Prediction maps: Monthlies
# Summer_2019 = map_predict('Sept_2019_Predictions.feather', NE_Lat, NE_Lon)
# Spring_2019 = map_predict('May_2019_Predictions.feather', NE_Lat, NE_Lon)
# Winter_2019 = map_predict_regional('Feb_2019_Predictions.feather', NE_Lat, NE_Lon)

# Prediction maps: Monthly averages, Arctic coverage
# Summer_2021 = map_predict_regional('Sept_2019_Predictions.feather', Lat_df, Lon_df, 45, 90, -172, -123,
# ccrs.NearsidePerspective(central_longitude=-145.0, central_latitude=60.0,
# satellite_height=500000))

#######################################################################################################################

# Define a function to clean and map RATIO prediction data:


def map_ratio_predict(prediction_feather, lat, lon):
    # predictions are the derived prediction from R and the associated feather files, in csv format.
    # lat and lon are the dataframe with size/indices [719:1199] and [599:1391], respectively

    y_predictions = pd.read_feather(prediction_feather)  # Read prediction csv file from R

    poc = y_predictions.pop("POC")  # Remove poc for now

    predictions_map = np.array(pd.DataFrame(y_predictions))  # Create Array

    # Reshape the coordinate variables in order to map them: Need to make 2D coordinate dfs in 1D array
    lat_map = np.array(lat).flatten()
    lon_map = np.array(lon).flatten()

    # Reshape the predictions to match the lats/lons:
    predictions_map = predictions_map.reshape(480, 792)  # Corrected dimensions

    # Replace all negative predictions with detection limit (5):
    predictions_map = np.where(predictions_map < 0, 5, predictions_map)

    # PLOT ALL PREDICTIONS

    fig = plt.figure(figsize=(12, 6))
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    ax.gridlines()
    ct = ax.pcolormesh(lon_map, lat_map, predictions_map, transform=ccrs.PlateCarree(),  # Color predictions
                       cmap="jet")  # Continuous color bar
    plt.colorbar(ct, orientation="vertical", fraction=0.02, pad=0.03)
    ax.set_xticks(np.arange(-155, -122, 5), crs=ccrs.PlateCarree())  # Tick marks
    ax.set_yticks(np.arange(40, 60, 5), crs=ccrs.PlateCarree())
    ax.set_ylabel("Latitude ˚N")  # Labels
    ax.set_xlabel("Longitude ˚W")
    ax.set_title("TEP (µg XG eq/L)")
    ax.add_feature(cfeature.LAND, zorder=1, edgecolor='k', facecolor='silver')  # Features
    ax.add_feature(cfeature.COASTLINE, zorder=1, linewidth=0.3)
    ax.set_ylim([46, 56])  # Limits
    ax.set_xlim([-146, -123])

    return fig, poc, predictions_map, lon_map, lat_map


#######################################################################################################################

# Validation/annotation plot: Only summer 2021
# Combine the predictions with the coordinates to validate the predictions with measured values:
TEP_Sq = pd.DataFrame(data=Summer_2021[2], index=Lat.squeeze(), columns=Lon.squeeze())

# Now, lets plot a cross validation map:
# Load the validation data:
Val = pd.read_csv("Sat_Val_Ann.csv")

# Plot all the predictions:
fig = plt.figure(figsize=(12, 6))
ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
ax.gridlines()

# Continuous color:
ct = ax.pcolormesh(Summer_2021[3], Summer_2021[4], Summer_2021[2], transform=ccrs.PlateCarree(), cmap="jet")
plt.colorbar(ct, orientation="vertical", fraction=0.02, pad=0.03)

# Empty plots to label:
texts = []
plt.scatter(x=Val.Lon, y=Val.Lat, facecolors='none', edgecolors='red', s=12)
for i, txt in enumerate(Val.Ann):
    texts.append(ax.annotate(txt, (Val.Lon[i], Val.Lat[i]), xytext=(Val.Lon[i], Val.Lat[i]),
                             arrowprops=dict(lw=0.01),
                             fontsize=5, color="black",
                             bbox={'facecolor': '0.9', 'edgecolor': 'black', 'boxstyle': 'square'}))
    adjust_text(texts)
    ax = fig.axes[0]

# Tick marks:
ax.set_xticks(np.arange(-155, -122, 5), crs=ccrs.PlateCarree())
ax.set_yticks(np.arange(40, 60, 5), crs=ccrs.PlateCarree())

# Labels
ax.set_ylabel("Latitude ˚N")
ax.set_xlabel("Longitude ˚W")
ax.set_title("TEP (µg XG eq/L)")

# Features:
ax.add_feature(cfeature.LAND, zorder=1, edgecolor='k', facecolor='silver')
ax.add_feature(cfeature.COASTLINE, zorder=1, linewidth=0.3)

# Limits:
ax.set_ylim([46, 56])
ax.set_xlim([-146, -123])
