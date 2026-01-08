# -*- coding: utf-8 -*-
"""
MSWEP Precipitation Map Generation.

This script reads MSWEP NetCDF files, clips them to the basin,
accumulates precipitation, and generates a map.

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
from shapely.geometry import mapping
from pyproj import CRS

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION FLAGS
# ==========================================
# Set to TRUE to re-read NetCDF files and recalculate accumulation.
# Set to FALSE to load existing pickle data.
RUN_PROCESS = True 

# Save the calculated result to pickle?
SAVE_OUTPUT = True

# ==========================================
# PATHS & DATES
# ==========================================
BASE_DIR = os.getcwd()
SHAPEFILE_PATH = os.path.join(BASE_DIR, 'Data', 'Shapes_MSF', 'Jubones', 'jubonesMSF_catch.shp')
RAW_DATA_DIR = r'C:\Users\patri\OneDrive\Documentos\MAESTRIA_HIDROLOGIA_UCUENCA\DataFusion_TESIS\Datos_Caudal\Data\Precipitation\MSWEP' # Adjust if needed
PICKLE_OUTPUT_DIR = os.path.join(BASE_DIR, 'Data', 'pickle_MSWEP_2019_GIT')

# Numeric Date Range: YYYYJJJHH (Year, Julian Day, Hour)
START_DATE_INT = 201900100 
END_DATE_INT = 201900200 

# ==========================================
# FUNCTIONS
# ==========================================
def filter_mswep_files(file_list, start_int, end_int):
    """Filters files based on YYYYJJJHH integer format."""
    filtered = []
    for fname in file_list:
        try:
            # Filename format assumption: YYYYJJJ.HH.nc
            f_year = int(fname[:4])
            f_day = int(fname[4:7])
            f_hour = int(fname[8:10])
            f_date = int(f"{f_year:04d}{f_day:03d}{f_hour:02d}")
            
            if start_int <= f_date <= end_int:
                filtered.append(fname)
        except ValueError:
            continue
    return filtered

def process_mswep_data(files, raw_dir, basin_gdf):
    """Reads, clips, and accumulates NetCDF data."""
    print(f"Starting processing of {len(files)} files...")
    start_time = time.time()
    
    list_arrays = []
    
    for i, filename in enumerate(files):
        if (i+1) % 50 == 0: print(f"Processing {i+1}/{len(files)}")
        
        path = os.path.join(raw_dir, filename)
        try:
            with xr.open_dataset(path) as ds:
                # Extract precipitation
                da = ds['precipitation']
                da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
                da.rio.write_crs("epsg:4326", inplace=True)
                
                # Clip
                da_clipped = da.rio.clip(basin_gdf.geometry.apply(mapping), basin_gdf.crs, all_touched=True)
                list_arrays.append(da_clipped)
        except Exception as e:
            print(f"Error reading {filename}: {e}")
            continue

    print("Concatenating and accumulating...")
    combined = xr.concat(list_arrays, dim='time')
    # Accumulate (skipna=True treats NaNs as 0 during sum)
    total_accum = combined.sum(dim='time', skipna=True)
    
    # Post-process: Mask 0 as NaN if desired (as per original script)
    final_accum = total_accum.where(total_accum != 0, other=float('nan'))
    
    print(f"Elapsed time: {time.time() - start_time:.2f} s")
    return final_accum

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # 1. Load Shapefile
    print("Loading shapefile...")
    basin_shp = gpd.read_file(SHAPEFILE_PATH)
    target_crs = CRS.from_epsg(4326)
    basin_proj = basin_shp.to_crs(target_crs)
    
    accumulated_data = None

    # 2. Processing Block
    if RUN_PROCESS:
        if os.path.exists(RAW_DATA_DIR):
            all_files = os.listdir(RAW_DATA_DIR)
            target_files = filter_mswep_files(all_files, START_DATE_INT, END_DATE_INT)
            
            if target_files:
                accumulated_data = process_mswep_data(target_files, RAW_DATA_DIR, basin_proj)
                
                if SAVE_OUTPUT:
                    os.makedirs(PICKLE_OUTPUT_DIR, exist_ok=True)
                    # Note: Original filename had 2019-2023, adjust as needed
                    out_path = os.path.join(PICKLE_OUTPUT_DIR, "MSWEP_Cumul_pcp_calculated.p")
                    with open(out_path, "wb") as f:
                        pickle.dump(accumulated_data, f)
                    print(f"Saved to {out_path}")
            else:
                print("No files found in range.")
        else:
            print(f"Directory not found: {RAW_DATA_DIR}")
            
    else:
        # Load existing
        pickle_path = os.path.join(PICKLE_OUTPUT_DIR, "MSWEP_Cumul_pcp_calculated.p")
        if os.path.exists(pickle_path):
            print("Loading data from pickle...")
            with open(pickle_path, "rb") as f:
                accumulated_data = pickle.load(f)
        else:
            print("Pickle file not found. Set RUN_PROCESS = True.")

    # 3. Plotting
    if accumulated_data is not None:
        print(f"Max: {accumulated_data.max().values:.2f} mm")
        
        fig, ax = plt.subplots(figsize=(10, 10))
        accumulated_data.plot(ax=ax, cmap='GnBu', vmin=400, vmax=1600, cbar_kwargs={'label': 'mm'})
        basin_proj.boundary.plot(ax=ax, color='black')
        plt.title('MSWEP Cumulative Precipitation', fontsize=16)
        plt.show()
