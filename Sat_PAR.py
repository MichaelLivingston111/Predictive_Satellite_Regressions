# A set of code used to extract satellite derived photosynthetically active radiation (PAR) for a specific set of
# coordinates and 24h period.

# Import:
import pandas as pd
import numpy as np
import netCDF4 as nc

# Load data file:
ds = nc.Dataset("Sat_PAR/AQUA_MODIS.20210824.L3m.DAY.PAR.par.4km.nc")


# Examine:
