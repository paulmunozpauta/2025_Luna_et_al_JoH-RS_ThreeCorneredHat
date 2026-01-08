# -*- coding: utf-8 -*-
"""
PERSIANN Data Extraction (Binary GZIP to DataFrame).

Reads binary files, performs spatial interpolation (regridding) to match IMERG grid,
clips to basin, and saves as DataFrame.

Author: Patricio Luna Abril
"""

import os
import gzip
import time
import pickle
import warnings
import pandas as pd
import numpy as np
import xarray as xr
import geopandas as gpd
import rioxarray
from datetime import datetime
from shapely.geometry import mapping

warnings.filterwarnings('ignore')

# FLAGS
RUN_EXTRACTION = True

# CONFIG
BASE_DIR = os.getcwd()
SHAPEFILE_PATH = os.path.join(BASE_DIR, 'Data', 'Shapes_MSF', 'Jubones', 'jubonesMSF_catch.shp')
RAW_DATA_DIR = r'C:\Users\patri\OneDrive\Documentos\MAESTRIA_HIDROLOGIA_UCUENCA\DataFusion_TESIS\Datos_Caudal\Data\Precipitation\PERSIANN'
IMERG_REF_FILE = os.path.join(BASE_DIR, 'Data', 'pickle_IMERG', 'gridref_IMERG.p')
OUTPUT_PICKLE = os.path.join(BASE_DIR, 'Data', 'Pcp_Dataframes', 'SPP_DFrames', 'DATAFRAME_PERSIANN_2019_GIT.p')

def read_persiann_gz(filepath):
    """Reads PERSIANN-CCS binary GZIP format."""
    with gzip.GzipFile(filepath, 'rb') as f:
        data = np.frombuffer(f.read(), dtype=np.dtype('>h'))
    
    data = data.reshape((3000, 9000))
    # Geometric transformations specific to CCS
    data = data[::-1, :]
    data = np.hstack((data[:, 4500:], data[:, :4500]))
    data = data.astype(float) / 100.0
    data[data < 0] = np.nan
    return data

if __name__ == "__main__":
    
    if RUN_EXTRACTION:
        basin = gpd.read_file(SHAPEFILE_PATH).to_crs(epsg=4326)
        
        # Load Reference Grid for Interpolation
        ds_imerg = pickle.load(open(IMERG_REF_FILE, "rb"))
        
        # Grid definition
        lat_grid = np.arange(-60, 60, 0.04)
        lon_grid = np.arange(-180, 180, 0.04)
        
        # Files (Add your filter logic here if needed)
        files = sorted([f for f in os.listdir(RAW_DATA_DIR) if f.endswith('.gz')])
        # Example filter for 2023 (file name usually contains date)
        # files = [f for f in files if '23' in f] 
        
        list_dfs = []
        
        print(f"Processing {len(files)} files...")
        for i, filename in enumerate(files):
            if (i+1)%50==0: print(f"Processing {i+1}")
            
            path = os.path.join(RAW_DATA_DIR, filename)
            try:
                raw_data = read_persiann_gz(path)
                
                # To Xarray
                da = xr.DataArray(raw_data, coords=[lat_grid, lon_grid], dims=["lat", "lon"])
                da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
                da.rio.write_crs("epsg:4326", inplace=True)
                
                # Interpolation (Regridding) to match IMERG
                da_regridded = da.interp_like(ds_imerg, method='linear')
                da_regridded.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
                da_regridded.rio.write_crs("epsg:4326", inplace=True)
                
                # Clip
                da_clipped = da_regridded.rio.clip(basin.geometry.apply(mapping), basin.crs, all_touched=True)
                
                # Timestamp (Parse from filename)
                # Example: rgccs1h2300100.bin.gz -> YYJJJHH
                ts_str = filename[-14:-7]
                dt = datetime.strptime(ts_str, '%y%j%H')
                
                df = pd.DataFrame(da_clipped.values.flatten(), dtype='float32').T
                df['timestamp'] = dt
                df.set_index('timestamp', inplace=True)
                list_dfs.append(df)
                
            except Exception as e:
                print(f"Error {filename}: {e}")
                continue
                
        if list_dfs:
            df_final = pd.concat(list_dfs)
            os.makedirs(os.path.dirname(OUTPUT_PICKLE), exist_ok=True)
            with open(OUTPUT_PICKLE, "wb") as f:
                pickle.dump(df_final, f)
            print("Done.")
