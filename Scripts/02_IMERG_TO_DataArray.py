# -*- coding: utf-8 -*-
"""
IMERG Data Extraction (HDF5 to DataFrame).

Extracts precipitation, clips to basin, and resamples from 30min to Hourly.

Author: Patricio Luna Abril
"""

import os
import glob
import time
import pickle
import warnings
import h5py
import numpy as np
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
from datetime import datetime
from shapely.geometry import mapping

warnings.filterwarnings('ignore')

# FLAGS
RUN_EXTRACTION = True
SAVE_PICKLE = True

# CONFIG
BASE_DIR = os.getcwd()
SHAPEFILE_PATH = os.path.join(BASE_DIR, 'Data', 'Shapes_MSF', 'Jubones', 'jubonesMSF_catch.shp')
RAW_DATA_DIR = r'C:\Users\patri\OneDrive\Documentos\MAESTRIA_HIDROLOGIA_UCUENCA\DataFusion_TESIS\Datos_Caudal\Data\Precipitation\IMERG'
OUTPUT_PICKLE = os.path.join(BASE_DIR, 'Data', 'Pcp_Dataframes', 'SPP_DFrames', 'DATAFRAME_IMERG_2019_GIT.p')

START_DATE = pd.Timestamp("2019-01-01 00:00")
END_DATE = pd.Timestamp("2019-01-01 23:30")

def filter_imerg_files(files, start, end):
    filtered = []
    start_int = int(start.strftime('%Y%m%d%H%M'))
    end_int = int(end.strftime('%Y%m%d%H%M'))
    for f in files:
        fname = os.path.basename(f)
        try:
            # Format: ...20230101-S000000...
            ymd = int(fname[-40:-32])
            hm = int(fname[-30:-26])
            dt_int = int(f"{ymd:04d}{hm:04d}")
            if start_int <= dt_int <= end_int:
                filtered.append(f)
        except:
            continue
    return sorted(filtered)

if __name__ == "__main__":
    
    basin = gpd.read_file(SHAPEFILE_PATH).to_crs(epsg=4326)
    
    if RUN_EXTRACTION:
        print("Searching files...")
        all_files = glob.glob(os.path.join(RAW_DATA_DIR, "*.RT-H5"))
        target_files = filter_imerg_files(all_files, START_DATE, END_DATE)
        
        list_dfs = []
        print(f"Extracting {len(target_files)} files...")
        
        for i, filepath in enumerate(target_files):
            if (i+1)%100==0: print(f"Processing {i+1}")
            try:
                with h5py.File(filepath, 'r') as f:
                    precip = f['/Grid/precipitationUncal'][:]
                    precip[precip < 0] = np.nan
                    # Transpose (Lat, Lon)
                    precip = precip[0, :, :].transpose()
                    
                    lats = f['Grid/lat'][:]
                    lons = f['Grid/lon'][:]
                    
                    # To Xarray
                    da = xr.DataArray(precip, coords={'lat': lats, 'lon': lons}, dims=['lat', 'lon'])
                    da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
                    da.rio.write_crs("epsg:4326", inplace=True)
                    
                    # Clip
                    da_clipped = da.rio.clip(basin.geometry.apply(mapping), basin.crs, all_touched=True)
                    
                    # Timestamp
                    fname = os.path.basename(filepath)
                    dt = datetime.strptime(f"{fname[-40:-32]}{fname[-30:-26]}", '%Y%m%d%H%M')
                    
                    # To DataFrame row
                    df = pd.DataFrame(da_clipped.values.flatten(), dtype='float32').T
                    df['timestamp'] = dt
                    df.set_index('timestamp', inplace=True)
                    list_dfs.append(df)
            except Exception as e:
                print(f"Error {filepath}: {e}")
                continue
        
        if list_dfs:
            print("Concatenating...")
            df_full = pd.concat(list_dfs)
            
            print("Resampling to Hourly...")
            df_hourly = df_full.resample('H').mean()
            
            if SAVE_PICKLE:
                os.makedirs(os.path.dirname(OUTPUT_PICKLE), exist_ok=True)
                with open(OUTPUT_PICKLE, "wb") as f:
                    pickle.dump(df_hourly, f)
                print("Saved.")
