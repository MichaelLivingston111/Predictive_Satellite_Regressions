# A set of code used to extract satellite derived photosynthetically active radiation (PAR) for a specific set of
# coordinates and 24h period.

# Import:
import pandas as pd
import numpy as np
import netCDF4 as nc
import xarray as xr

# Load data files, by individual dates:

# August 2021:
NC_Aug25 = xr.open_dataset("Sat_PAR/AQUA_MODIS.20210825.L3m.DAY.PAR.par.4km.nc")
NC_Aug26 = xr.open_dataset("Sat_PAR/AQUA_MODIS.20210826.L3m.DAY.PAR.par.4km.nc")
NC_Aug27 = xr.open_dataset("Sat_PAR/AQUA_MODIS.20210827.L3m.DAY.PAR.par.4km.nc")
NC_Aug28 = xr.open_dataset("Sat_PAR/AQUA_MODIS.20210828.L3m.DAY.PAR.par.4km.nc")
NC_Aug29 = xr.open_dataset("Sat_PAR/AQUA_MODIS.20210829.L3m.DAY.PAR.par.4km.nc")
NC_Aug30 = xr.open_dataset("Sat_PAR/AQUA_MODIS.20210830.L3m.DAY.PAR.par.4km.nc")
NC_Aug31 = xr.open_dataset("Sat_PAR/AQUA_MODIS.20210831.L3m.DAY.PAR.par.4km.nc")
NC_Sep01 = xr.open_dataset("Sat_PAR/AQUA_MODIS.20210901.L3m.DAY.PAR.par.4km.nc")
NC_Sep02 = xr.open_dataset("Sat_PAR/AQUA_MODIS.20210902.L3m.DAY.PAR.par.4km.nc")

# June 2019:
NC_2019_June03 = xr.open_dataset("Sat_PAR/AQUA_MODIS.20190603.L3m.DAY.PAR.par.4km.nc")
NC_2019_June04 = xr.open_dataset("Sat_PAR/AQUA_MODIS.20190604.L3m.DAY.PAR.par.4km.nc")
NC_2019_June05 = xr.open_dataset("Sat_PAR/AQUA_MODIS.20190605.L3m.DAY.PAR.par.4km.nc")
NC_2019_June06 = xr.open_dataset("Sat_PAR/AQUA_MODIS.20190606.L3m.DAY.PAR.par.4km.nc")
NC_2019_June07 = xr.open_dataset("Sat_PAR/AQUA_MODIS.20190607.L3m.DAY.PAR.par.4km.nc")
NC_2019_June08 = xr.open_dataset("Sat_PAR/AQUA_MODIS.20190608.L3m.DAY.PAR.par.4km.nc")
NC_2019_June09 = xr.open_dataset("Sat_PAR/AQUA_MODIS.20190609.L3m.DAY.PAR.par.4km.nc")
NC_2019_June10 = xr.open_dataset("Sat_PAR/AQUA_MODIS.20190610.L3m.DAY.PAR.par.4km.nc")
NC_2019_June11 = xr.open_dataset("Sat_PAR/AQUA_MODIS.20190611.L3m.DAY.PAR.par.4km.nc")


# Define Station Coordinates:
P4 = [48.3898, -126.3994]
P12 = [48.583, -130.4001]
P16 = [49.1693, -134.3986]
P20 = [49.3398, -138.399]
P26 = [50, -145]


# Define a function to take coordinates and extract 24h average PAR:

def par_fun(ds, station):

    subset = ds.sel(lat=station[0], lon=station[1], method='nearest')
    x = subset.par.values
    x = x*1000000/86400  # convert from einsteins > microeinsteins, and days > seconds
    return x
    print(x)


par_fun(NC_2019_June09, P26)


print(NC_2019_June04.par)

