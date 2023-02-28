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
B7 = [48.32, -125.355]
LBP3 = [50.032, -127.553]
LC11 = [48.189, -126.267]
LD09 = [48.357, -126.3]
LG02 = [49.187, -126.381]
CS02 = [50.413, -129.28]
LC06 = [48.365, -125.54]
LB01 = [48.404, -124.5965]
LB15 = [48.044, -126.085]
LG09 = [48.512, -127.194]
Haro59 = [48.3685, -123.1482]
DIX3 = [54.2879, -132.1811]
HECS8 = [54.2322, -131.0558]
CH03 = [54.4146, -130.5566]
CH05 = [54.3369, -130.3766]
CH12 = [54.237, -130.3401]
CH19 = [54.152, -130.3564]
CH28 = [54.11133, -130.30966]
P1 = [48.3441, -125.2963]
P2 = [48.3595, -125.5967]
P3 = [48.3748, -126.1975]
P4 = [48.3898, -126.3994]
P6 = [48.4459, -127.3995]
P7 = [48.4658, -128.0977]
P8 = [48.4898, -128.3975]
P9 = [48.514, -129.0987]
P10 = [48.537, -129.3999]
P12 = [48.583, -130.4001]
P13 = [49.0241, -131.4022]
P16 = [49.1693, -134.3986]
P17 = [49.2096, -135.3985]
P19 = [49.3002, -137.4002]
P20 = [49.3398, -138.399]
P21 = [49.3805, -139.4006]
P22 = [49.4205, -140.3995]
P23 = [49.4604, -141.3998]
P25 = [50.0002, -143.3624]
P35 = [50.0002, -144.1802]
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

