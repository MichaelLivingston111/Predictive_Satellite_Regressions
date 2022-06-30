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
from cartopy.mpl.ticker import LongitudeFormatter, LatitudeFormatter

#######################################################################################################################

# Formatting all coordinates:
# Import a file to obtain the necessary coordinates: Use these as default coordinates
file = 'May_2021_8D_CHL_4km.nc'  # May 4-12 2021 coordinates
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
    ax1.plot(points[:, 0], points[:, 1], 'ro', markeredgecolor='r', alpha=0.6,
             ms=9)  # Plot points on an orthographic projection
    ax1.set_ylabel("Latitude ˚N")  # Labels
    ax1.set_xlabel("Longitude ˚W")
    ax1.set_title("Estimated TEP concentrations from a multivariate regression model \n \n TEP ~ Chlorophyll a + POC + "
                  "Temperature + Region + Bloom")
    ax1.coastlines()
    # ax.gridlines(color='gray', linestyle='--')  # optional
    ax1.add_feature(cfeature.LAND, zorder=1, edgecolor='k', facecolor='silver')  # Features
    ax1.add_feature(cfeature.COASTLINE, zorder=1, linewidth=0.3)

    return fig1  # poc, predictions_map, lon_map, lat_map


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
    plt.scatter(x=points.Station_lon, y=points.Station_lat, facecolors='none', edgecolors='red', s=12)  # Stations
    ax.set_ylabel("Latitude ˚N")  # Labels
    ax.set_xlabel("Longitude ˚W")
    ax.set_title("TEP (µg XG eq/L)")  # Title
    ax.coastlines()  # Coastlines
    # ax.gridlines(color='gray', linestyle='--')  # optional
    ax.add_feature(cfeature.LAND, zorder=1, edgecolor='k', facecolor='silver')  # Features
    ax.add_feature(cfeature.COASTLINE, zorder=1, linewidth=0.3)  # More features
    ax.set_ylim([lat_min, lat_max])  # Limits
    ax.set_xlim([lon_min, lon_max])

    return fig, poc, predictions_map, lon_map, lat_map


#######################################################################################################################

# Define a function to clean and map prediction data: 6 panel with predictor variables

# prediction_feather is the derived predictions from R in the associated feather format.
# lat_df and lon_df are the coordinates from the satellite data frame (global size).
# lat_min, lat_max etc. is the range of the prediction map you want to plot.
# proj is the projection style you wish to use.


def map_predict_parameters(prediction_feather, lat_df, lon_df, lat_min, lat_max, lon_min, lon_max, proj):
    y_predictions = pd.read_feather(prediction_feather)  # Read prediction csv file from R

    poc = y_predictions.pop("POC")  # Remove variables for now
    temp = y_predictions.pop("Temperature")
    chl = y_predictions.pop("Log_Chl")

    predictions_map = np.array(pd.DataFrame(y_predictions))  # Create Arrays
    poc_map = np.array(pd.DataFrame(poc))  # Create Arrays
    temp_map = np.array(pd.DataFrame(temp))  # Create Arrays
    chl_map = np.array(pd.DataFrame(chl))  # Create Arrays

    ratio_map = (np.array(pd.DataFrame(y_predictions * 0.7))) / (np.array(pd.DataFrame(poc))) * 100  # Create Ratio Array

    # Reshape the coordinate variables in order to map them: Need to make 2D coordinate dfs in 1D array
    lat_map = np.array(lat_df).flatten()
    lon_map = np.array(lon_df).flatten()

    # Reshape the predictions to match the lat/lon, remove any negative values::
    predictions_map = predictions_map.reshape(4320, 8640)  # Corrected global dimensions based on prediction df
    predictions_map = np.where(predictions_map < 0, 5, predictions_map)  # Remove negative, replace with detection limit

    poc_map = poc_map.reshape(4320, 8640)  # Corrected global dimensions based on prediction df
    temp_map = temp_map.reshape(4320, 8640)  # Corrected global dimensions based on prediction df
    chl_map = chl_map.reshape(4320, 8640)  # Corrected global dimensions based on prediction df
    ratio_map = ratio_map.reshape(4320, 8640)  # Corrected global dimensions based on prediction df

    # PLOT ALL PREDICTIONS
    points = pd.read_csv("Lats_lons_Arctic.csv")

    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()

    fig = plt.figure(figsize=(14, 10))
    plt.subplots_adjust(wspace=0.2, hspace=0.01)

    # TEP predictions:
    ax1 = fig.add_subplot(3, 2, 1, projection=proj)  # Create the geoaxes for a specified projection
    ct1 = ax1.pcolormesh(lon_map, lat_map, predictions_map, transform=ccrs.PlateCarree(),  # Color predictions
                         cmap=cm.cm.haline, vmax=160)  # Continuous color bar
    cbar1 = plt.colorbar(ct1, orientation="vertical", fraction=0.02, pad=0.03)
    cbar1.set_label('TEP (\u03bcg XGeq/L)', rotation=270, labelpad=15)
    ax1.coastlines()
    ax1.add_feature(cfeature.LAND, zorder=1, edgecolor='k', facecolor='silver')  # Features
    ax1.add_feature(cfeature.COASTLINE, zorder=1, linewidth=0.3)
    ax1.set_ylim([lat_min, lat_max])  # Limits
    ax1.set_xlim([lon_min, lon_max])
    ax1.gridlines(color='gray', linestyle='--')
    ax1.set_yticks(np.arange(lat_min, lat_max, 3), crs=ccrs.PlateCarree())
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)

    # Measured values scatter plot:
    ax2 = fig.add_subplot(3, 2, 2, projection=proj)  # Create the geoaxes for a specified projection
    ct2 = ax2.scatter(x=points.Station_lon, y=points.Station_lat, c=points.Summer_2021_TEP, s=14,
                      cmap=cm.cm.haline, vmin=5, vmax=160, alpha=0.9, transform=ccrs.PlateCarree())
    cbar2 = plt.colorbar(ct2, orientation="vertical", fraction=0.02, pad=0.03)
    cbar2.set_label('TEP (\u03bcg XGeq/L)', rotation=270, labelpad=15)
    ax2.coastlines()
    ax2.add_feature(cfeature.LAND, zorder=1, edgecolor='k', facecolor='silver')  # Features
    ax2.add_feature(cfeature.COASTLINE, zorder=1, linewidth=0.3)
    ax2.set_ylim([lat_min, lat_max])  # Limits
    ax2.set_xlim([lon_min, lon_max])
    ax2.gridlines(color='gray', linestyle='--')
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.yaxis.set_major_formatter(lat_formatter)

    # POC map
    ax3 = fig.add_subplot(3, 2, 3, projection=proj)  # Create the geoaxes for a specified projection
    ct3 = ax3.pcolormesh(lon_map, lat_map, poc_map, transform=ccrs.PlateCarree(),  # Color predictions
                         cmap=cm.cm.haline)  # Continuous color bar
    cbar3 = plt.colorbar(ct3, orientation="vertical", fraction=0.02, pad=0.03)
    cbar3.set_label('POC (\u03bcg/L)', rotation=270, labelpad=15)
    ax3.coastlines()
    ax3.add_feature(cfeature.LAND, zorder=1, edgecolor='k', facecolor='silver')  # Features
    ax3.add_feature(cfeature.COASTLINE, zorder=1, linewidth=0.3)
    ax3.set_ylim([lat_min, lat_max])  # Limits
    ax3.set_xlim([lon_min, lon_max])
    ax3.gridlines(color='gray', linestyle='--')
    ax3.set_yticks(np.arange(lat_min, lat_max, 3), crs=ccrs.PlateCarree())
    ax3.xaxis.set_major_formatter(lon_formatter)
    ax3.yaxis.set_major_formatter(lat_formatter)

    # Chlorophyll map
    ax4 = fig.add_subplot(3, 2, 4, projection=proj)  # Create the geoaxes for a specified projection
    ct4 = ax4.pcolormesh(lon_map, lat_map, chl_map, transform=ccrs.PlateCarree(),  # Color predictions
                         cmap=cm.cm.haline, vmin=-2)  # Continuous color bar
    cbar4 = plt.colorbar(ct4, orientation="vertical", fraction=0.02, pad=0.03)
    cbar4.set_label('Chlorophyll a, Log scale (\u03bcg/L)', rotation=270, labelpad=15)
    ax4.coastlines()
    ax4.add_feature(cfeature.LAND, zorder=1, edgecolor='k', facecolor='silver')  # Features
    ax4.add_feature(cfeature.COASTLINE, zorder=1, linewidth=0.3)
    ax4.set_ylim([lat_min, lat_max])  # Limits
    ax4.set_xlim([lon_min, lon_max])
    ax4.gridlines(color='gray', linestyle='--')
    ax4.xaxis.set_major_formatter(lon_formatter)
    ax4.yaxis.set_major_formatter(lat_formatter)

    # Temp map
    ax5 = fig.add_subplot(3, 2, 5, projection=proj)  # Create the geoaxes for a specified projection
    ct5 = ax5.pcolormesh(lon_map, lat_map, ratio_map, transform=ccrs.PlateCarree(),  # Color predictions
                         cmap=cm.cm.thermal, vmin=3, vmax=45)  # Continuous color bar
    cbar5 = plt.colorbar(ct5, orientation="vertical", fraction=0.02, pad=0.03)
    cbar5.set_label('%POC as TEP', rotation=270, labelpad=15)
    ax5.coastlines()
    ax5.add_feature(cfeature.LAND, zorder=1, edgecolor='k', facecolor='silver')  # Features
    ax5.add_feature(cfeature.COASTLINE, zorder=1, linewidth=0.3)
    ax5.set_ylim([lat_min, lat_max])  # Limits
    ax5.set_xlim([lon_min, lon_max])
    ax5.gridlines(color='gray', linestyle='--')
    ax5.set_xticks(np.arange(lon_min, lon_max, 3), crs=ccrs.PlateCarree())
    ax5.set_yticks(np.arange(lat_min, lat_max, 3), crs=ccrs.PlateCarree())
    ax5.xaxis.set_major_formatter(lon_formatter)
    ax5.yaxis.set_major_formatter(lat_formatter)

    # Prediction ratio map:
    ax6 = fig.add_subplot(3, 2, 6, projection=proj)  # Create the geoaxes for a specified projection
    ct6 = ax6.pcolormesh(lon_map, lat_map, temp_map, transform=ccrs.PlateCarree(),  # Color predictions
                         cmap=cm.cm.thermal, vmin=12, vmax=20)  # Continuous color bar
    cbar6 = plt.colorbar(ct6, orientation="vertical", fraction=0.02, pad=0.03)
    cbar6.set_label('Temperature (\u00b0C)', rotation=270, labelpad=15)
    ax6.coastlines()
    ax6.add_feature(cfeature.LAND, zorder=1, edgecolor='k', facecolor='silver')  # Features
    ax6.add_feature(cfeature.COASTLINE, zorder=1, linewidth=0.3)
    ax6.set_ylim([lat_min, lat_max])  # Limits
    ax6.set_xlim([lon_min, lon_max])
    ax6.gridlines(color='gray', linestyle='--')
    ax6.set_xticks(np.arange(lon_min, lon_max, 3), crs=ccrs.PlateCarree())
    ax6.xaxis.set_major_formatter(lon_formatter)
    ax6.yaxis.set_major_formatter(lat_formatter)

    return fig


#######################################################################################################################

# Define a function to clean and map prediction data: Cross validation plot

# prediction_feather is the derived predictions from R in the associated feather format.
# lat_df and lon_df are the coordinates from the satellite data frame (global size).
# lat_min, lat_max etc. is the range of the prediction map you want to plot.
# proj is the projection style you wish to use.


def map_predict_cv(prediction_feather, lat_df, lon_df, lat_min, lat_max, lon_min, lon_max, proj):
    y_predictions = pd.read_feather(prediction_feather)  # Read prediction csv file from R

    poc = y_predictions.pop("POC")  # Remove variables for now
    temp = y_predictions.pop("Temperature")
    chl = y_predictions.pop("Log_Chl")

    predictions_map = np.array(pd.DataFrame(y_predictions))  # Create Arrays

    ratio_map = (np.array(pd.DataFrame(y_predictions * 0.7))) / (np.array(pd.DataFrame(poc))) * 100  # Create Ratio Array

    # Reshape the coordinate variables in order to map them: Need to make 2D coordinate dfs in 1D array
    lat_map = np.array(lat_df).flatten()
    lon_map = np.array(lon_df).flatten()

    # Reshape the predictions to match the lat/lon, remove any negative values::
    predictions_map = predictions_map.reshape(4320, 8640)  # Corrected global dimensions based on prediction df
    predictions_map = np.where(predictions_map < 0, 5, predictions_map)  # Remove negative, replace with detection limit

    ratio_map = ratio_map.reshape(4320, 8640)  # Corrected global dimensions based on prediction df

    # PLOT ALL PREDICTIONS
    points = pd.read_csv("Lats_lons_Arctic.csv")

    lon_formatter = LongitudeFormatter(zero_direction_label=True)
    lat_formatter = LatitudeFormatter()

    fig = plt.figure(figsize=(14, 10))
    plt.subplots_adjust(wspace=0.2, hspace=0.01)

    # TEP predictions:
    ax1 = fig.add_subplot(2, 2, 1, projection=proj)  # Create the geoaxes for a specified projection
    ct1 = ax1.pcolormesh(lon_map, lat_map, predictions_map, transform=ccrs.PlateCarree(),  # Color predictions
                         cmap=cm.cm.haline, vmax=160)  # Continuous color bar
    cbar1 = plt.colorbar(ct1, orientation="vertical", fraction=0.02, pad=0.03)
    cbar1.set_label('TEP (\u03bcg XGeq/L)', rotation=270, labelpad=15)
    ax1.coastlines()
    ax1.add_feature(cfeature.LAND, zorder=1, edgecolor='k', facecolor='silver')  # Features
    ax1.add_feature(cfeature.COASTLINE, zorder=1, linewidth=0.3)
    ax1.set_ylim([lat_min, lat_max])  # Limits
    ax1.set_xlim([lon_min, lon_max])
    ax1.gridlines(color='gray', linestyle='--')
    ax1.set_yticks(np.arange(lat_min, lat_max, 3), crs=ccrs.PlateCarree())
    ax1.xaxis.set_major_formatter(lon_formatter)
    ax1.yaxis.set_major_formatter(lat_formatter)

    # Ratio map
    ax2 = fig.add_subplot(2, 2, 2, projection=proj)  # Create the geoaxes for a specified projection
    ct2 = ax2.pcolormesh(lon_map, lat_map, ratio_map, transform=ccrs.PlateCarree(),  # Color predictions
                         cmap=cm.cm.thermal, vmin=3, vmax=50)  # Continuous color bar
    cbar2 = plt.colorbar(ct2, orientation="vertical", fraction=0.02, pad=0.03)
    cbar2.set_label('%POC as TEP', rotation=270, labelpad=15)
    ax2.coastlines()
    ax2.add_feature(cfeature.LAND, zorder=1, edgecolor='k', facecolor='silver')  # Features
    ax2.add_feature(cfeature.COASTLINE, zorder=1, linewidth=0.3)
    ax2.set_ylim([lat_min, lat_max])  # Limits
    ax2.set_xlim([lon_min, lon_max])
    ax2.gridlines(color='gray', linestyle='--')
    ax2.set_xticks(np.arange(lon_min, lon_max, 3), crs=ccrs.PlateCarree())
    ax2.set_yticks(np.arange(lat_min, lat_max, 3), crs=ccrs.PlateCarree())
    ax2.xaxis.set_major_formatter(lon_formatter)
    ax2.yaxis.set_major_formatter(lat_formatter)

    # Measured values scatter plot: TEP
    ax3 = fig.add_subplot(2, 2, 3, projection=proj)  # Create the geoaxes for a specified projection
    ct3 = ax3.scatter(x=points.Station_lon, y=points.Station_lat, c=points.Summer_2021_TEP, s=14,
                      cmap=cm.cm.haline, vmin=5, vmax=160, alpha=0.9, transform=ccrs.PlateCarree())
    cbar3 = plt.colorbar(ct3, orientation="vertical", fraction=0.02, pad=0.03)
    cbar3.set_label('TEP (\u03bcg XGeq/L)', rotation=270, labelpad=15)
    ax3.coastlines()
    ax3.add_feature(cfeature.LAND, zorder=1, edgecolor='k', facecolor='silver')  # Features
    ax3.add_feature(cfeature.COASTLINE, zorder=1, linewidth=0.3)
    ax3.set_ylim([lat_min, lat_max])  # Limits
    ax3.set_xlim([lon_min, lon_max])
    ax3.gridlines(color='gray', linestyle='--')
    ax3.set_xticks(np.arange(lon_min, lon_max, 3), crs=ccrs.PlateCarree())
    ax3.set_yticks(np.arange(lat_min, lat_max, 3), crs=ccrs.PlateCarree())
    ax3.xaxis.set_major_formatter(lon_formatter)
    ax3.yaxis.set_major_formatter(lat_formatter)

    # Measured values scatter plot: TEP ratio
    ax4 = fig.add_subplot(2, 2, 4, projection=proj)  # Create the geoaxes for a specified projection
    ct4 = ax4.scatter(x=points.Station_lon, y=points.Station_lat, c=points.Summer_2021_Ratio, s=14,
                      cmap=cm.cm.thermal, vmin=3, vmax=50, alpha=0.9, transform=ccrs.PlateCarree())
    cbar4 = plt.colorbar(ct4, orientation="vertical", fraction=0.02, pad=0.03)
    cbar4.set_label('%POC as TEP)', rotation=270, labelpad=15)
    ax4.coastlines()
    ax4.add_feature(cfeature.LAND, zorder=1, edgecolor='k', facecolor='silver')  # Features
    ax4.add_feature(cfeature.COASTLINE, zorder=1, linewidth=0.3)
    ax4.set_ylim([lat_min, lat_max])  # Limits
    ax4.set_xlim([lon_min, lon_max])
    ax4.gridlines(color='gray', linestyle='--')
    ax4.xaxis.set_major_formatter(lon_formatter)
    ax4.yaxis.set_major_formatter(lat_formatter)
    ax4.set_xticks(np.arange(lon_min, lon_max, 3), crs=ccrs.PlateCarree())

    return fig


#######################################################################################################################

# Define a function to clean and map RATIO prediction data:


def map_ratio_predict(prediction_feather, lat_df, lon_df, lat_min, lat_max, lon_min, lon_max, proj):
    # predictions are the derived prediction from R and the associated feather files, in csv format.
    # lat and lon are the dataframe with size/indices [719:1199] and [599:1391], respectively

    y_predictions = pd.read_feather(prediction_feather)  # Read prediction csv file from R

    poc = y_predictions.pop("POC")  # Remove poc for now

    predictions_map = (np.array(pd.DataFrame(y_predictions))) / (np.array(pd.DataFrame(poc))) * 100  # Create Array

    # Reshape the coordinate variables in order to map them: Need to make 2D coordinate dfs in 1D array
    lat_map = np.array(lat_df).flatten()
    lon_map = np.array(lon_df).flatten()

    # Reshape the predictions to match the lat/lon, remove any negative values::
    predictions_map = predictions_map.reshape(4320, 8640)  # Corrected global dimensions based on prediction df
    predictions_map = np.where(predictions_map < 0, 0, predictions_map)  # Remove negative, replace with detection limit

    # PLOT ALL PREDICTIONS
    points = pd.read_csv("Lats_lons_Arctic.csv")

    fig = plt.figure(figsize=(10, 9))
    ax = fig.add_subplot(1, 1, 1, projection=proj)  # Create the geoaxes for a specified projection
    ct = ax.pcolormesh(lon_map, lat_map, predictions_map, transform=proj,  # Color predictions
                       cmap=cm.cm.thermal, vmax=60)  # Continuous color bar
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

    return fig


# ---------------------------------------------------------------------------------------------------------------------

# APPLY THE PREDICTION MAPPING FUNCTIONS:

# Prediction maps: 8 day averages
# Summer_2021 = map_predict_regional('Aug_Sept_2021_Predictions.feather', Lat_df, Lon_df, 46, 56, -146, -123,
                                   # proj=ccrs.PlateCarree())

# Spring_2021 = map_predict_regional('May_2021_Predictions.feather', Lat_df, Lon_df, 46, 56, -146, -123,
                                   # proj=ccrs.PlateCarree())

# Prediction maps: 8 day averages, Global coverage, summer 2021
Summer_2021_Total = map_predict_global('Aug_Sept_2021_Predictions_Global.feather', Lat_df, Lon_df, 45, 90, -172, -123,
                                       ccrs.NearsidePerspective(central_longitude=-145.0, central_latitude=58.0,
                                                                satellite_height=500000))
Summer_2021_Total.savefig("Global_predictions_Aug2021.png")  # bbox_inches='tight' - optional to specify blank space

Summer_2021_Total_ratio = map_ratio_predict('Aug_Sept_2021_Predictions_Global.feather', Lat_df, Lon_df, 45, 90, -172, -123,
                                      ccrs.NearsidePerspective(central_longitude=-145.0, central_latitude=58.0,
                                                               satellite_height=500000))
Summer_2021_Total_ratio.savefig("Global_predictions_ratio_Aug2021.png")


# Prediction maps: 8 day averages, Global coverage, spring 2021
Spring_2021_Total = map_predict_global('May_2021_Predictions_Global.feather', Lat_df, Lon_df, 45, 90, -172, -123,
                                       ccrs.NearsidePerspective(central_longitude=-145.0, central_latitude=58.0,
                                                                satellite_height=500000))
Spring_2021_Total.savefig("Global_predictions_May2021.png")

Spring_2021_Total_ratio = map_ratio_predict('May_2021_Predictions_Global.feather', Lat_df, Lon_df, 45, 90, -172, -123,
                                      ccrs.NearsidePerspective(central_longitude=-145.0, central_latitude=58.0,
                                                               satellite_height=500000))
Spring_2021_Total_ratio.savefig("Global_predictions_ratio_May2021.png")

# ---------------------------------------------------------------------------------------------------------------------

Summer_2021 = map_predict_parameters('Aug_Sept_2021_Predictions.feather', Lat_df, Lon_df, 46, 56, -146, -123,
                                     proj=ccrs.PlateCarree())
Summer_2021.savefig("Parameter_plot.png", bbox_inches='tight')


Summer_2021_CV = map_predict_cv('Aug_Sept_2021_Predictions.feather', Lat_df, Lon_df, 46, 56, -146, -123,
                                     proj=ccrs.PlateCarree())
Summer_2021_CV.savefig("Parameter_plot_CV.png", bbox_inches='tight')

# Prediction maps: Monthlies
# Summer_2019 = map_predict('Sept_2019_Predictions.feather', NE_Lat, NE_Lon)
# Spring_2019 = map_predict('May_2019_Predictions.feather', NE_Lat, NE_Lon)
# Winter_2019 = map_predict_regional('Feb_2019_Predictions.feather', NE_Lat, NE_Lon)

# Prediction maps: Monthly averages, Arctic coverage
# Summer_2021 = map_predict_regional('Sept_2019_Predictions.feather', Lat_df, Lon_df, 45, 90, -172, -123,
# ccrs.NearsidePerspective(central_longitude=-145.0, central_latitude=60.0,
# satellite_height=500000))


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
