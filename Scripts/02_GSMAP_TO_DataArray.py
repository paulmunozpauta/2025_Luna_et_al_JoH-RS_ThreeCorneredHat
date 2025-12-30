# -*- coding: utf-8 -*-
"""
Created on Mon Aug 21 17:04:58 2023

@author: PatricioJavier
"""

import os
import zipfile
import pandas as pd
import geopandas as gpd
import pyproj
from pyproj import CRS
import datetime
from datetime import datetime
import rasterio
import rasterio.mask
from shapely.geometry import mapping, box
import rioxarray
import numpy as np
import xarray as xr
import matplotlib.pyplot as plt
import time
import pickle

# Function to get all the zip files in a directory
def find_zip_files_in_directory(root_dir):
    zip_files = []  # Initialize a list to store the paths of ZIP files

    # Walk through the directory tree using os.walk
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if filename.endswith('.zip'):
                zip_files.append(os.path.join(dirpath, filename))

    return zip_files

# Function to get all the files within the given date range
def filter_files_GSMAP(file_list, start_date, end_date):
    """
    Filter a list of files based on a date range.

    Parameters:
    - file_list: List of file names.
    - start_date: Starting date (numeric format, e.g., 2019122003).
    - end_date: Ending date (numeric format, e.g., 2019122021).

    Returns:
    - List of filtered file names.
    """

    filtered_files = []
    startdate_int = int(start_date.strftime('%Y%m%d%H'))
    enddate_int = int(end_date.strftime('%Y%m%d%H'))

    for filename in file_list:
        # Extract date components from the filename
        # Assuming the format is YYYYMMDD.HH.nc
        file_ymd = int(filename[-31:-23])
        file_hour = int(filename[-22:-20])

        # Convert the extracted components to a single numeric date
        file_date = int(f"{file_ymd:04d}{file_hour:02d}")

        if startdate_int <= file_date <= enddate_int:
            filtered_files.append(filename)

    return filtered_files

###### Specify the date range
start_date = pd.Timestamp("2023-01-01 00:00")
end_date = pd.Timestamp("2023-12-31 23:00")

# Specify the base folder containing the data
# Path for data stored in PC: 'C:\Users\patri\OneDrive\Documentos\MAESTRIA_HIDROLOGIA_UCUENCA\DataFusion_TESIS\Datos_Caudal\Data\Test_GSMAP'
folder = os.getcwd()
# base_folder =r'C:\Users\patri\OneDrive\Documentos\MAESTRIA_HIDROLOGIA_UCUENCA\DataFusion_TESIS\Datos_Caudal\Data\Test_GSMAP'
base_folder =r'D:\GSMaP'

# List with all the files in the directory
zip_files_list = find_zip_files_in_directory(base_folder)

# Get a list of CSV paths within the specified date range
csv_paths = filter_files_GSMAP(zip_files_list, start_date, end_date)

# Add the shapefile of Jubones basin
basin_shp = gpd.read_file(r'Data\Shapes_MSF\Jubones\jubonesMSF_catch.shp')
##Get the boundary geometry
basin_geometry = basin_shp.geometry.iloc[0]  # Assuming only one geometry in the shapefile
basin_shp.plot() #To plot the shp
source_crs = CRS.from_string('+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs') #Source CRS: XY coordinates
target_crs = CRS.from_epsg(4326) #To what CRS is going to be projected
basin_proj = basin_shp.to_crs(target_crs)
basin_proj.plot() #Plot the projected shapefile
ext = basin_proj.total_bounds #It creates an array with the bounds in the form (minx, miny, maxx, maxy)
rect_ext = box(ext[0], ext[1], ext[2], ext[3])
rect_gdf = gpd.GeoDataFrame(geometry=[rect_ext], crs="EPSG:4326")
lon = np.arange(-81.95,-33.85,0.1)
lat = np.arange(-9.95,13.05,0.1)

# Initialize an empty DataFrame to store concatenated data
# list_arrays = []
list_df = []
# sum_data2d = pd.DataFrame(0, index=np.arange(230), columns=np.arange(481))
# Iterate through each CSV path and concatenate the data
# csv_path = csv_paths[1] #Solo para prueba rapida
# index = 2
# item = csv_paths[2]
start_time = time.time()
for index, item in enumerate(csv_paths):
    try:
        print(index+1, 'out of', len(csv_paths))
        df = pd.read_csv(item)

    except (IOError, EOFError) as e:    
    # Catch specific errors related to file reading or handling
        continue
    
    n_lat, n_lon = df[' Lat'].nunique(), df['  Lon'].nunique()
    data_2d = df.pivot(index=' Lat', columns ='  Lon', values = '  RainRate')
    # data_3d.index = df[' Lat'].unique()
    # data_3d.columns = df['  Lon'].unique()
    # sum_data2d.index = df[' Lat'].unique()
    # sum_data2d.columns = df['  Lon'].unique()
    # data_1 = data_2d[:,240:] #for ccs
    # data_2 = data_2d[:,:240] #for ccs
    # data_2d_4 = np.hstack((data_1,data_2))
    # sum_data2d = sum_data2d+data_2d
    ds = xr.DataArray(data=data_2d, dims=['Latitude', 'Longitude'], coords=[lat, lon]) #Convert to Data Array taking precipitation values
    #ds = xr.DataArray(data=data_2d, dims=['Latitude', 'Longitude'], coords={'Latitude': df[' Lat'].values[:n_lat], 'Longitude': df['  Lon'].values[:n_lon]}) #Convert to Data Array taking precipitation values
    ds.rio.set_spatial_dims(x_dim="Longitude", y_dim="Latitude", inplace=True) #Define spatial dimensions based on longitude and latitude
    ds.rio.write_crs("epsg:4326", inplace=True) #Define the CRS for the data array
    # ds_1 = ds.rio.clip(basin_proj.geometry.apply(mapping), basin_proj.crs, all_touched=True) #Clipping the data array to the study area
    ds_1 = ds.rio.clip(basin_proj.geometry.apply(mapping), basin_proj.crs, all_touched=True) #Clipping the data array to the study area
    # Adding the timestamp
    file_ymd = int(item[-31:-23])
    file_hour = int(item[-22:-20])
    file_date = str(f"{file_ymd:04d}{file_hour:02d}")
    timestamp = datetime.strptime(file_date, '%Y%m%d%H')
    df = pd.DataFrame(ds_1.values.flatten(), dtype='float32').T
    df['timestamp'] = timestamp
    df.set_index('timestamp', inplace=True)
    # list_arrays.append(ds_1) #Each ds is saved in the list 
    list_df.append(df)

# combined_data_array = xr.concat(list_arrays, dim='time') #concatenate the data simulating a time dimension
df_pcp = pd.concat(list_df, axis=0, ignore_index=False)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

####### Save as pickle variable 
pickle.dump(df_pcp, open(r'Data\Pcp_Dataframes\SPP_DFrames\DATAFRAME_GSMAP_2023.p', "wb" ))
