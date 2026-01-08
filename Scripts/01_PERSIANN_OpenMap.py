# -*- coding: utf-8 -*-
"""
PERSIANN-CCS Precipitation Map Generation.

Author: Patricio Luna Abril
"""
import os
import gzip
import time
import pickle
import warnings
import numpy as np
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
RAW_DATA_DIR = r'C:\Users\patri\OneDrive\Documentos\MAESTRIA_HIDROLOGIA_UCUENCA\DataFusion_TESIS\Datos_Caudal\Data\Precipitation\PERSIANN' 
PICKLE_OUTPUT = os.path.join(BASE_DIR, 'Data', 'pickle_PERSIANN', 'PERSIANN_cumul_pcp_2019_GIT.p')

# Julian Dates integers
START_DATE_INT = 1900100 
END_DATE_INT = 1900123

# PERSIANN Grid
LON_GRID = np.arange(-180, 180, 0.04)
LAT_GRID = np.arange(-60, 60, 0.04)

def filter_persiann_files(files, start_int, end_int):
    filtered = []
    for fname in files:
        try:
            # Assumes format: rgccs1hYYJJJHH.bin.gz
            f_date = int(fname[-14:-7]) 
            if start_int <= f_date <= end_int:
                filtered.append(fname)
        except: continue
    return filtered

def read_persiann_binary(filepath):
    with gzip.GzipFile(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.dtype('>h'))
    data = data.reshape((3000, 9000))
    data = data[::-1, :] # Flip
    data = np.hstack((data[:, 4500:], data[:, :4500])) # Shift
    data = data.astype(float) / 100.0
    data[data < 0] = np.nan
    return data

if __name__ == "__main__":
    basin_shp = gpd.read_file(SHAPEFILE_PATH).to_crs(epsg=4326)
    final_da = None
    
    if RUN_PROCESS and os.path.exists(RAW_DATA_DIR):
        files = filter_persiann_files(os.listdir(RAW_DATA_DIR), START_DATE_INT, END_DATE_INT)
        print(f"Processing {len(files)} files...")
        
        # Accumulate in memory (Efficient)
        running_sum = np.zeros((3000, 9000), dtype=np.float32)
        
        for i, fname in enumerate(files):
            if (i+1)%100 == 0: print(f"Processing {i+1}...")
            try:
                data = read_persiann_binary(os.path.join(RAW_DATA_DIR, fname))
                running_sum += np.nan_to_num(data, nan=0.0)
            except Exception as e:
                print(f"Error {fname}: {e}")
        
        # Convert to GeoSpatial
        da = xr.DataArray(running_sum, dims=["lat", "lon"], coords=[LAT_GRID, LON_GRID])
        da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
        da.rio.write_crs("epsg:4326", inplace=True)
        
        final_da = da.rio.clip(basin_shp.geometry.apply(mapping), basin_shp.crs, all_touched=True)
        
        if SAVE_OUTPUT:
            os.makedirs(os.path.dirname(PICKLE_OUTPUT), exist_ok=True)
            with open(PICKLE_OUTPUT, "wb") as f:
                pickle.dump(final_da, f)
    else:
        if os.path.exists(PICKLE_OUTPUT):
            with open(PICKLE_OUTPUT, "rb") as f:
                final_da = pickle.load(f)

    if final_da is not None:
        fig, ax = plt.subplots(figsize=(10, 10))
        final_da.plot(ax=ax, cmap='GnBu', vmin=0, vmax=1750)
        basin_shp.boundary.plot(ax=ax, color='black')
        plt.show()
