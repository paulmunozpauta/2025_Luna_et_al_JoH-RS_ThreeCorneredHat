# -*- coding: utf-8 -*-
"""
IMERG Precipitation Map Generation.

Reads raw IMERG HDF5 files, extracts 'precipitationUncal', 
clips to basin, converts rate to accumulation, and generates a map.

Author: Patricio Luna Abril
"""

import os
import time
import pickle
import warnings
import h5py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import geopandas as gpd
import xarray as xr
import rioxarray
from shapely.geometry import mapping
from pyproj import CRS

warnings.filterwarnings('ignore')

# FLAGS
RUN_PROCESS = True  
SAVE_OUTPUT = True

# PATHS
BASE_DIR = os.getcwd()
SHAPEFILE_PATH = os.path.join(BASE_DIR, 'Data', 'Shapes_MSF', 'Jubones', 'jubonesMSF_catch.shp')
RAW_DATA_DIR = r'C:\Users\patri\OneDrive\Documentos\MAESTRIA_HIDROLOGIA_UCUENCA\DataFusion_TESIS\Datos_Caudal\Data\Precipitation\IMERG' 
PICKLE_OUTPUT = os.path.join(BASE_DIR, 'Data', 'pickle_IMERG', 'IMERG_Accumul_pcp_2019_GIT.p')

# Date Range
START_DATE = pd.Timestamp("2019-01-01 00:00")
END_DATE = pd.Timestamp("2019-01-01 23:30")

def filter_imerg_files(file_list, start_date, end_date):
    filtered = []
    start_int = int(start_date.strftime('%Y%m%d%H%M'))
    end_int = int(end_date.strftime('%Y%m%d%H%M'))

    for filename in file_list:
        try:
            # Format: ...20220402-S000000...
            f_year = int(filename[23:27])
            f_month = int(filename[27:29])
            f_day = int(filename[29:31])
            f_hour = int(filename[33:37])
            f_date = int(f"{f_year:04d}{f_month:02d}{f_day:02d}{f_hour:04d}")

            if start_int <= f_date <= end_int:
                filtered.append(filename)
        except Exception:
            continue
    return sorted(filtered)

if __name__ == "__main__":
    import pandas as pd 

    basin_shp = gpd.read_file(SHAPEFILE_PATH).to_crs(epsg=4326)
    accum_data = None

    if RUN_PROCESS:
        if os.path.exists(RAW_DATA_DIR):
            files = os.listdir(RAW_DATA_DIR)
            target_files = filter_imerg_files(files, START_DATE, END_DATE)
            
            list_arrays = []
            print(f"Processing {len(target_files)} files...")
            
            for i, filename in enumerate(target_files):
                if (i+1) % 50 == 0: print(f"Processing {i+1}...")
                
                path = os.path.join(RAW_DATA_DIR, filename)
                try:
                    with h5py.File(path, 'r') as f:
                        precip = f['Grid/precipitationUncal'][0][:][:]
                        precip = np.transpose(precip) # Transpose to (Lat, Lon)
                        lats = f['Grid/lat'][:]
                        lons = f['Grid/lon'][:]
                        
                    da = xr.DataArray(precip, dims=('lat', 'lon'), coords={'lat': lats, 'lon': lons})
                    da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
                    da.rio.write_crs("epsg:4326", inplace=True)
                    
                    da_clipped = da.rio.clip(basin_shp.geometry.apply(mapping), basin_shp.crs, all_touched=True)
                    list_arrays.append(da_clipped)
                except Exception as e:
                    print(f"Error {filename}: {e}")
            
            if list_arrays:
                print("Accumulating...")
                combined = xr.concat(list_arrays, dim='time')
                # Filter negatives
                combined = combined.where(combined >= 0)
                # Sum and convert to mm (Rate * 0.5h)
                accum_data = combined.sum(dim='time', skipna=True) * 0.5
                
                if SAVE_OUTPUT:
                    os.makedirs(os.path.dirname(PICKLE_OUTPUT), exist_ok=True)
                    with open(PICKLE_OUTPUT, "wb") as f:
                        pickle.dump(accum_data, f)
        else:
            print(f"Directory not found: {RAW_DATA_DIR}")
    else:
        if os.path.exists(PICKLE_OUTPUT):
            with open(PICKLE_OUTPUT, "rb") as f:
                accum_data = pickle.load(f)

    if accum_data is not None:
        fig, ax = plt.subplots(figsize=(10, 10))
        accum_data.plot(ax=ax, cmap='GnBu', cbar_kwargs={'label': 'mm'})
        basin_shp.boundary.plot(ax=ax, color='black')
        plt.title('IMERG Cumulative Precipitation', fontsize=16)
        plt.show()
