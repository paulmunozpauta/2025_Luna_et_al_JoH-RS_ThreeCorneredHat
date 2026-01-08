# -*- coding: utf-8 -*-
"""
GSMaP Data Extraction (CSV to DataFrame).

Author: Patricio Luna Abril
"""
import os
import time
import pickle
import warnings
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
import numpy as np
from datetime import datetime
from shapely.geometry import mapping

warnings.filterwarnings('ignore')

# FLAGS
RUN_EXTRACTION = True
SAVE_PICKLE = True

# CONFIG
BASE_DIR = os.getcwd()
SHAPEFILE_PATH = os.path.join(BASE_DIR, 'Data', 'Shapes_MSF', 'Jubones', 'jubonesMSF_catch.shp')
RAW_DATA_DIR = r'C:\Users\patri\OneDrive\Documentos\MAESTRIA_HIDROLOGIA_UCUENCA\DataFusion_TESIS\Datos_Caudal\Data\Precipitation\GSMaP'
OUTPUT_FILE = os.path.join(BASE_DIR, 'Data', 'Pcp_Dataframes', 'SPP_DFrames', 'DATAFRAME_GSMAP_2019_GIT.p')

START_DATE = pd.Timestamp("2019-01-01 00:00")
END_DATE = pd.Timestamp("2019-01-01 23:00")

# Spatial Grid
LON_GRID = np.arange(-81.95, -33.85, 0.1)
LAT_GRID = np.arange(-9.95, 13.05, 0.1)

def get_gsmap_files(root_dir, start_date, end_date):
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
    
    if RUN_EXTRACTION:
        basin = gpd.read_file(SHAPEFILE_PATH).to_crs(epsg=4326)
        files = get_gsmap_files(RAW_DATA_DIR, START_DATE, END_DATE)
        
        list_dfs = []
        print(f"Processing {len(files)} files...")
        
        for i, filepath in enumerate(files):
            if (i+1)%100==0: print(f"Processing {i+1}...")
            try:
                df = pd.read_csv(filepath)
                # Pivot
                data_2d = df.pivot(index=' Lat', columns='  Lon', values='  RainRate')
                
                # To Xarray
                ds = xr.DataArray(data_2d.values, dims=['lat', 'lon'], coords=[LAT_GRID, LON_GRID])
                ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
                ds.rio.write_crs("epsg:4326", inplace=True)
                
                # Clip
                ds_clipped = ds.rio.clip(basin.geometry.apply(mapping), basin.crs, all_touched=True)
                
                # Timestamp
                fname = os.path.basename(filepath)
                dt = datetime.strptime(f"{fname[-31:-23]}{fname[-22:-20]}", '%Y%m%d%H')
                
                # DataFrame Row
                df_row = pd.DataFrame(ds_clipped.values.flatten(), dtype='float32').T
                df_row['timestamp'] = dt
                df_row.set_index('timestamp', inplace=True)
                list_dfs.append(df_row)
            except Exception as e:
                print(f"Error {filepath}: {e}")
                
        if list_dfs:
            df_final = pd.concat(list_dfs)
            if SAVE_PICKLE:
                os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
                with open(OUTPUT_FILE, "wb") as f:
                    pickle.dump(df_final, f)
            print("Done.")
