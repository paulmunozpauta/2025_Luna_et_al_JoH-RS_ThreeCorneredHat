# -*- coding: utf-8 -*-
"""
GSMaP Precipitation Map Generation.

Reads GSMaP CSV files (RainRate), converts to spatial grid, clips, and accumulates.

Author: Patricio Luna Abril

"""

import os
import time
import pickle
import warnings
import pandas as pd
import xarray as xr
import geopandas as gpd
import matplotlib.pyplot as plt
import numpy as np
from shapely.geometry import mapping
from pyproj import CRS

warnings.filterwarnings('ignore')

# FLAGS
RUN_PROCESS = True
SAVE_OUTPUT = True

# PATHS & DATES
BASE_DIR = os.getcwd()
SHAPEFILE_PATH = os.path.join(BASE_DIR, 'Data', 'Shapes_MSF', 'Jubones', 'jubonesMSF_catch.shp')
RAW_DATA_DIR = r'C:\Users\patri\OneDrive\Documentos\MAESTRIA_HIDROLOGIA_UCUENCA\DataFusion_TESIS\Datos_Caudal\Data\Precipitation\GSMaP' 
PICKLE_OUTPUT = os.path.join(BASE_DIR, 'Data', 'Test_GSMAP', 'pickle_data_GSMAP', 'GSMAP_Accumul_pcp_2019_GIT.p')

START_DATE = pd.Timestamp("2019-01-01 00:00")
END_DATE = pd.Timestamp("2019-01-01 23:00")

# SPATIAL GRID (GSMaP Standard)
LON_GRID = np.arange(-81.95, -33.85, 0.1)
LAT_GRID = np.arange(-9.95, 13.05, 0.1)

def get_gsmap_files(root_dir, start_date, end_date):
    """Finds and filters GSMaP CSV/ZIP files."""
    files = []
    start_int = int(start_date.strftime('%Y%m%d%H'))
    end_int = int(end_date.strftime('%Y%m%d%H'))
    
    for dirpath, _, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.csv') or filename.endswith('.zip'):
                try:
                    f_ymd = int(filename[-31:-23])
                    f_hour = int(filename[-22:-20])
                    f_date = int(f"{f_ymd:04d}{f_hour:02d}")
                    if start_int <= f_date <= end_int:
                        files.append(os.path.join(dirpath, filename))
                except Exception:
                    continue
    return sorted(files)

if __name__ == "__main__":
    
    basin_shp = gpd.read_file(SHAPEFILE_PATH)
    basin_proj = basin_shp.to_crs(epsg=4326)
    
    accumulated_data = None
    
    if RUN_PROCESS:
        files = get_gsmap_files(RAW_DATA_DIR, START_DATE, END_DATE)
        print(f"Found {len(files)} files to process.")
        
        list_arrays = []
        
        for i, csv_path in enumerate(files):
            try:
                df = pd.read_csv(csv_path)
                # Pivot to grid
                data_2d = df.pivot(index=' Lat', columns='  Lon', values='  RainRate')
                
                # Convert to Xarray
                ds = xr.DataArray(data_2d.values, dims=['lat', 'lon'], coords=[LAT_GRID, LON_GRID])
                ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
                ds.rio.write_crs("epsg:4326", inplace=True)
                
                # Clip
                ds_clipped = ds.rio.clip(basin_proj.geometry.apply(mapping), basin_proj.crs, all_touched=True)
                list_arrays.append(ds_clipped)
                
            except Exception as e:
                print(f"\n[ERROR] failed in {i+1} | Archivo: {csv_path}")
                print(f"       Reason: {e}")
                continue
        
        if list_arrays:
            print("Accumulating...")
            combined = xr.concat(list_arrays, dim='time')
            accumulated_data = combined.sum(dim='time', skipna=False)
            
            if SAVE_OUTPUT:
                os.makedirs(os.path.dirname(PICKLE_OUTPUT), exist_ok=True)
                with open(PICKLE_OUTPUT, "wb") as f:
                    pickle.dump(accumulated_data, f)
    else:
        if os.path.exists(PICKLE_OUTPUT):
            with open(PICKLE_OUTPUT, "rb") as f:
                accumulated_data = pickle.load(f)

    if accumulated_data is not None:
        print(f"Max precip: {accumulated_data.max().values}")
        accumulated_data.plot(cmap='GnBu', vmin=0, vmax=1750)
        plt.show()
