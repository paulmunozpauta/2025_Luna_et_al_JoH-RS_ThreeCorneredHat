# -*- coding: utf-8 -*-
"""
Kernel Smoothing

Applies a spatial filter (Kernel) to the TCH-fused precipitation data
to reduce dimensionality/noise before the Random Forest model.

Author: Patricio Luna Abril
"""

import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
from scipy import signal
from shapely.geometry import mapping
from pyproj import CRS

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.getcwd()
SHAPEFILE_PATH = os.path.join(BASE_DIR, 'Data', 'Shapes_MSF', 'Jubones', 'jubonesMSF_catch.shp')

# Input: TCH Fused Data
INPUT_TCH_FILE = os.path.join(BASE_DIR, 'Data', 'Fused_pcp', 'TCH_DFrames', 'TCH_Fusion_Results_GIT.p')
# Reference Grid (IMERG)
IMERG_REF_FILE = os.path.join(BASE_DIR, 'Data', 'pickle_IMERG', 'gridref_IMERG.p')
# Output
OUTPUT_FILE = os.path.join(BASE_DIR, 'Data', 'Fused_pcp', 'Kernel_Balanced_TCH_DF', 'TCH_Kernel_2019_GIT.p')

# Date Range
START_DATE = pd.Timestamp("2019-01-01 00:00")
END_DATE = pd.Timestamp("2019-01-01 23:00")

# Kernel Definition (2x2)
KERNEL = np.array([[1, 1],
                   [1, 1]])

# Sampling Step (Stride for downsampling)
# Original logic used [0:rows:3], so step is 3
STRIDE = 3

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    start_time = time.time()
    
    # 1. Load Data & References
    print("Loading data...")
    basin_shp = gpd.read_file(SHAPEFILE_PATH).to_crs(epsg=4326)
    
    if not os.path.exists(IMERG_REF_FILE):
        raise FileNotFoundError(f"Reference grid not found: {IMERG_REF_FILE}")
    ds_imerg = pickle.load(open(IMERG_REF_FILE, "rb"))
    lons, lats = ds_imerg.lon, ds_imerg.lat
    grid_shape = ds_imerg.shape
    index_ds_imerg = np.arange(grid_shape[0] * grid_shape[1])

    if not os.path.exists(INPUT_TCH_FILE):
        raise FileNotFoundError(f"Input TCH file not found: {INPUT_TCH_FILE}")
    df_tch = pickle.load(open(INPUT_TCH_FILE, "rb"))
    
    # Filter Dates
    df_subset = df_tch.loc[START_DATE:END_DATE]
    print(f"Processing {len(df_subset)} time steps...")

    # Containers for results
    list_processed_arrays = []
    list_timestamps = []

    # 2. Kernel Loop
    for i in range(len(df_subset)):
        if (i + 1) % 500 == 0:
            print(f"Step {i + 1} / {len(df_subset)}")
            
        try:
            # A. Reconstruct 2D Grid from DataFrame Row
            current_date = df_subset.index[i]
            # Transpose and reindex to ensure full grid alignment
            flat_data = df_subset.iloc[i].T.reindex(index_ds_imerg)
            grid_2d = flat_data.values.reshape(grid_shape)
            
            # B. Create DataArray for Clipping
            da = xr.DataArray(grid_2d, coords={'lat': lats, 'lon': lons}, dims=['lat', 'lon'])
            da.rio.write_crs("epsg:4326", inplace=True)
            da_clipped = da.rio.clip(basin_shp.geometry.apply(mapping), basin_shp.crs, all_touched=True)
            
            # C. Convolution (Spatial Filter)
            # mode='valid': Output consists only of those elements that do not rely on zero-padding
            convolved = signal.convolve2d(da_clipped.values, KERNEL, boundary='wrap', mode='valid')
            
            # Normalize and Downsample
            convolved_norm = convolved / KERNEL.sum()
            downsampled = convolved_norm[0::STRIDE, 0::STRIDE]
            
            # D. Flatten and Store
            # Note: Using numpy flatten is operationally identical to nested loops but faster
            list_processed_arrays.append(downsampled.flatten())
            list_timestamps.append(current_date)
            
        except Exception as e:
            print(f"[ERROR] Step {i} ({current_date}): {e}")
            continue

    # 3. Create DataFrame
    print("Constructing final DataFrame...")
    if list_processed_arrays:
        matrix_result = np.vstack(list_processed_arrays)
        df_result = pd.DataFrame(matrix_result, index=list_timestamps)
        df_result.index.name = 'timestamp'
        
        # 4. Save
        os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
        with open(OUTPUT_FILE, "wb") as f:
            pickle.dump(df_result, f)
        
        print(f"Saved to: {OUTPUT_FILE}")
        print(f"Total time: {time.time() - start_time:.2f} s")
    else:
        print("No data processed.")
