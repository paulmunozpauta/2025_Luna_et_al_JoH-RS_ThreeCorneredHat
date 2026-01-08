# -*- coding: utf-8 -*-
"""
MSWEP Data Extraction (NetCDF to DataFrame).

Author: Patricio Luna Abril
"""
import os
import pickle
import warnings
import pandas as pd
import geopandas as gpd
import xarray as xr
import rioxarray
from datetime import datetime
from shapely.geometry import mapping

warnings.filterwarnings('ignore')

# FLAGS
RUN_EXTRACTION = True

# CONFIG
BASE_DIR = os.getcwd()
SHAPEFILE_PATH = os.path.join(BASE_DIR, 'Data', 'Shapes_MSF', 'Jubones', 'jubonesMSF_catch.shp')
RAW_DATA_DIR = r'C:\Users\patri\OneDrive\Documentos\MAESTRIA_HIDROLOGIA_UCUENCA\DataFusion_TESIS\Datos_Caudal\Data\Precipitation\MSWEP'
OUTPUT_FILE = os.path.join(BASE_DIR, 'Data', 'Pcp_Dataframes', 'SPP_DFrames', 'DATAFRAME_MSWEP_2019_GIT.p')

START_DATE = '2019-01-01 00:00'
END_DATE = '2019-01-01 23:00'

def filter_mswep_files(files, start, end):
    # Logic to filter YYYYJJJHH
    # ... (Same as previous script) ...
    # Simplified for brevity, assumes all_files in dir
    return sorted(files) # Add actual date logic here if directory has mixed years

if __name__ == "__main__":
    
    if RUN_EXTRACTION and os.path.exists(RAW_DATA_DIR):
        basin = gpd.read_file(SHAPEFILE_PATH).to_crs(epsg=4326)
        
        # Add filtering logic here similar to Map script if needed
        files = os.listdir(RAW_DATA_DIR) 
        
        list_dfs = []
        print(f"Processing {len(files)} files...")
        
        for i, filename in enumerate(files):
            if (i+1)%100==0: print(f"Processing {i+1}...")
            path = os.path.join(RAW_DATA_DIR, filename)
            try:
                with xr.open_dataset(path) as ds:
                    da = ds['precipitation']
                    da.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
                    da.rio.write_crs("epsg:4326", inplace=True)
                    da_clipped = da.rio.clip(basin.geometry.apply(mapping), basin.crs, all_touched=True)
                    
                    # Timestamp from filename (YYYYJJJ.HH.nc)
                    year = int(filename[:4])
                    day = int(filename[4:7])
                    hour = int(filename[8:10])
                    dt = datetime.strptime(f"{year}{day:03d}{hour:02d}", '%Y%j%H')
                    
                    df = pd.DataFrame(da_clipped.values.flatten(), dtype='float32').T
                    df['timestamp'] = dt
                    df.set_index('timestamp', inplace=True)
                    list_dfs.append(df)
            except Exception as e:
                print(f"Error {filename}: {e}")
        
        if list_dfs:
            df_final = pd.concat(list_dfs)
            os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
            with open(OUTPUT_FILE, "wb") as f:
                pickle.dump(df_final, f)
            print("Done.")
