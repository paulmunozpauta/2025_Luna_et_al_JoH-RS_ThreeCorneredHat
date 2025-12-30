# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 08:52:04 2023

@author: PatricioJavier

Code for converting satellite information from IMERG-ER to a data array containing all the SPPs
for each time step
"""
#%% IMPORT LIBRARIES & FUNCTIONS
import pickle
import numpy as np
import matplotlib.dates as dates
import os
import datetime
from datetime import datetime
import pandas as pd
import geopandas as gpd
import pyproj
from pyproj import CRS
import glob
import h5py
import rasterio
import rioxarray
import xarray as xr
from shapely.geometry import mapping, box
import time

def filter_files_MSWEP(files, start_date, end_date):
    """
    Filter a list of files based on a date range.

    Parameters:
    - files: List of file names.
    - start_date: Starting date (numeric format, e.g., 2019122003).
    - end_date: Ending date (numeric format, e.g., 2019122021).

    Returns:
    - List of filtered file names.
    """

    filtered_files = []

    for filename in files:
        # Extract date components from the filename
        # Assuming the format is YYYYMMDD.HH.nc
        file_year = int(filename[:4])
        file_day = int(filename[4:7])
        file_hour = int(filename[8:10])

        # Convert the extracted components to a single numeric date
        file_date = int(f"{file_year:04d}{file_day:03d}{file_hour:02d}")

        if start_date <= file_date <= end_date:
            filtered_files.append(filename)

    return filtered_files

def date_to_julian(date_str):
    # Convert date string to datetime object
    date = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
    
    # Convert datetime object to Julian calendar format
    julian_date = date.strftime('%Y%j%H')
    
    return julian_date

#%% LOADING THE STUDY AREA
# Add the shapefile of Jubones basin
jubones_shp = gpd.read_file(r'C:\Users\patri\OneDrive\Documentos\MAESTRIA_HIDROLOGIA_UCUENCA\DataFusion_TESIS\Datos_Caudal\Data\Shapes_MSF\Jubones\jubonesMSF_catch.shp')
##Get the boundary geometry
jubones_geometry = jubones_shp.geometry.iloc[0]  # Assuming only one geometry in the shapefile
jubones_shp.plot() #To plot the shp
##Projecting the shapefile to EPSG: 4326
source_crs = CRS.from_string('+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs') #Source CRS: XY coordinates
target_crs = CRS.from_epsg(4326) #To what CRS is going to be projected
jubones_proj = jubones_shp.to_crs(target_crs)

#List of NetCDF files in the directory
    #MSWEP v2.2: D:\MSWEP_v220
    #MSWEP v2.8: D:\MSWEP_v280
file_list = os.listdir(r'D:\MSWEP_v280') 
direccion_principal = r'D:\MSWEP_v280'
folder = os. getcwd()
# Example usage
start_date = '2019-01-01 00:00'
end_date = '2019-12-31 23:00'

start_julian = int(date_to_julian(start_date))
end_julian = int(date_to_julian(end_date))

filtered_files = filter_files_MSWEP(file_list, start_julian, end_julian)


#%% READING PRECIPITATION FILES
list_df = []
### FOR loop to clip the netcdf file and extract the pixels that intersect the shp
start_time = time.time()
for index, item in enumerate(filtered_files):
    print(index+1, 'out of', len(filtered_files))
    path = os.path.join(direccion_principal, item) #To get the list of files in the directory
    ds = xr.open_dataset(path) #Open each file as dataset using xarray
    ds = xr.DataArray(data=ds['precipitation']) #Convert to Data Array taking precipitation values
    ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True) #Define spatial dimensions based on longitude and latitude
    ds.rio.write_crs("epsg:4326", inplace=True) #Define the CRS for the data array
    ds_1 = ds.rio.clip(jubones_shp.geometry.apply(mapping), jubones_shp.crs, all_touched=True) #Clipping the data array to the study area

   # Adding the timestamp
    file_ymd = int(item[:7])
    file_hour = int(item[8:10])
    file_date = str(f"{file_ymd:07d}{file_hour:02d}")
    timestamp = datetime.strptime(file_date, '%Y%j%H')
    df = pd.DataFrame(ds_1.values.flatten(), dtype='float32').T
    df['timestamp'] = timestamp
    df.set_index('timestamp', inplace=True)
    list_df.append(df)
    ds.close()

df = pd.concat(list_df, axis=0, ignore_index=False)


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

####### Save as pickle variable 
pickle.dump(df, open(r'Data\Pcp_Dataframes\SPP_DFrames\DATAFRAME_MSWEP_2019.p', "wb" ))

#%% Merge all the dataframes 2019-2023 (OPTIONAL)
df19 = pickle.load(open( r'Data\Pcp_Dataframes\SPP_DFrames\DATAFRAME_MSWEP_2019.p', "rb"))
df20 = pickle.load(open( r'Data\Pcp_Dataframes\SPP_DFrames\DATAFRAME_MSWEP_2020.p', "rb"))
df21 = pickle.load(open( r'Data\Pcp_Dataframes\SPP_DFrames\DATAFRAME_MSWEP_2021.p', "rb"))
df22 = pickle.load(open( r'Data\Pcp_Dataframes\SPP_DFrames\DATAFRAME_MSWEP_2022.p', "rb"))
df23 = pickle.load(open( r'Data\Pcp_Dataframes\SPP_DFrames\DATAFRAME_MSWEP_2023.p', "rb"))

merged_df = pd.concat([df19, df20, df21, df22, df23], ignore_index=False)
pickle.dump(merged_df, open(r'Data\Pcp_Dataframes\SPP_DFrames\DATAFRAME_MSWEP_complete.p', "wb" ))
