# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 08:52:04 2023

@author: PatricioJavier

Code for converting satellite information from IMERG-ER to a data array containing all the SPPs
for each time step
"""

import os
import pickle
import numpy as np
import math
import matplotlib.dates as dates
import datetime
from datetime import datetime
import pandas as pd
import geopandas as gpd
import pyproj
from pyproj import CRS
import glob
import gzip
import rasterio
from rasterio.enums import Resampling
import rioxarray
import xarray as xr
from shapely.geometry import mapping, box
import matplotlib.pyplot as plt
import geopandas as gpd
import time

def filter_files_PERSIANN(files, start_date, end_date):
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
        file_date = int(filename[-14:-7])
        # file_year = int(filename[7:9])
        # file_day = int(filename[9:12])
        # file_hour = int(filename[12:14])

        # Convert the extracted components to a single numeric date
        # file_date = int(f"{file_year:02d}{file_day:03d}{file_hour:02d}")

        if start_date <= file_date <= end_date:
            filtered_files.append(filename)

    return filtered_files

def date_to_julian(date_str):
    # Convert date string to datetime object
    date = datetime.strptime(date_str, '%Y-%m-%d %H:%M')
    
    # Convert datetime object to Julian calendar format
    julian_date = date.strftime('%y%j%H')
    
    return julian_date

#### Getting the shapefile data
basin_shp = gpd.read_file(r'C:\Users\patri\OneDrive\Documentos\MAESTRIA_HIDROLOGIA_UCUENCA\DataFusion_TESIS\Datos_Caudal\Data\Shapes_MSF\Jubones\jubonesMSF_catch.shp')
basin_geometry = basin_shp.geometry.iloc[0]  #Get the boundary geometry
basin_shp.plot() #To plot the shp
##Projecting the shapefile to EPSG: 4326
source_crs = CRS.from_string('+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs') #Source CRS: XY coordinates
target_crs = CRS.from_epsg(4326) #To what CRS is going to be projected
basin_proj = basin_shp.to_crs(target_crs)
basin_proj.plot() #Plot the projected shapefile
ext = basin_proj.total_bounds #It creates an array with the bounds in the form (minx, miny, maxx, maxy)
rect_ext = box(math.floor(ext[0]*10)/10, math.floor(ext[1]*10)/10, math.ceil(ext[2]*10)/10, math.ceil(ext[3]*10)/10)
rect_gdf = gpd.GeoDataFrame(geometry=[rect_ext], crs="EPSG:4326")

#### Reading the satellite precipitation files
folder_files = r'D:\PERSIANN-CCS\hrly'
pattern = "**/*.gz"
list_of_Files = glob.glob(os.path.join(folder_files, pattern), recursive=True)
list_of_Files.sort()
folder = os.getcwd()

# Example usage
start_date = '2023-01-01 00:00'
end_date = '2023-12-31 23:00'

start_julian = int(date_to_julian(start_date))
end_julian = int(date_to_julian(end_date))

filtered_files = filter_files_PERSIANN(list_of_Files, start_julian, end_julian)
list_arrays = []
list_df = []
#Dimensions
lon = np.arange(-180,180,0.04)
lat = np.arange(-60,60,0.04)

# Read the reference IMERG map
dsIMERG = pickle.load(open(folder + r'\Data\pickle_IMERG\gridref_IMERG.p', "rb" ))
#### Reading of IMERG data
# index = 0
# item = filtered_files[0]
start_time = time.time()
dataset = xr.Dataset()
clip_bounds = rect_gdf.geometry.apply(mapping)
for index, item in enumerate(filtered_files):
    path = os.path.join(folder_files, item)
    print(index+1, 'out of', len(filtered_files))
    f=gzip.GzipFile(path)
    # with open(path, 'rb') as f:
    #     file_content = f.read()
    try:
        file_content = f.read()
    
    except (IOError, EOFError) as e:
        
        # Catch specific errors related to file reading or handling
        continue

    data = np.frombuffer(file_content, dtype=np.dtype('>h'))
    data = data.reshape((3000,9000))
    data = data[::-1,:]
    data_1 = data[:,4500:] #for ccs
    data_2 = data[:,:4500] #for ccs
    data = np.hstack((data_1,data_2))
    data= data/100 #for ccs
    data[data < 0] = np.nan # set values < 0 to nodata
    # data = np.flipud(data)
    data = xr.DataArray(data=data, dims=["lat", "lon"], coords=[lat,lon])
    data.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    data.rio.write_crs("epsg:4326", inplace=True)
    data = data.rio.clip(clip_bounds, rect_gdf.crs, all_touched=True)
    
    ### Upscaling using Linear interpolation
    da_rshp_PERSIANN = data.interp_like(dsIMERG, method='linear')
    # reindex_flat = data.reindex(lon=dsIMERG.lon, lat=dsIMERG.lat, method='nearest').values.flatten()
    da_rshp_PERSIANN.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
    da_rshp_PERSIANN.rio.write_crs("epsg:4326", inplace=True)
    # rshp_PERSIANN = rshp_PERSIANN.rio.clip(basin_proj.geometry.apply(mapping), basin_proj.crs, all_touched=True)
    da_rshp_PERSIANN = da_rshp_PERSIANN.rio.clip(basin_proj.geometry.apply(mapping), basin_proj.crs, all_touched=True)
    # Adding the timestamp as a dimension
    timestamp = item[-14:-7]
    timestamp = datetime.strptime(timestamp, '%y%j%H')
    # da_rshp_PERSIANN = da_rshp_PERSIANN.expand_dims(time=[timestamp])
    df = pd.DataFrame(da_rshp_PERSIANN.values.flatten(), dtype='float32').T
    df['timestamp'] = timestamp
    df.set_index('timestamp', inplace=True)
    # dataset['DS'+str(index)] = da_rshp_PERSIANN
    list_arrays.append(da_rshp_PERSIANN) #Each df is saved in the list
    list_df.append(df)

# concat_ds = xr.concat(dataset.values(), dim='time')
# combined_data_array = xr.concat(list_arrays, dim='time') #concatenate the data simulating a time dimension
# list_nan = []
# list_nonan =[]
df_pcp = pd.concat(list_df, axis=0, ignore_index=False)
# i = 0
# for i in range(len(combined_data_array)):
#     print(i+1, 'out of',len(combined_data_array))
#     df = pd.DataFrame(concat_ds[i].values.flatten(), dtype='float32')
#     df.set_index('time', inplace=True)
#     df_no_nan = df.dropna()
#     list_nan.append(df)
#     list_nonan.append(df_no_nan)


end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")
####### Save as pickle variable 
pickle.dump(df_pcp, open(r'Data\Pcp_Dataframes\SPP_DFrames\DATAFRAME_PERSIANN_2023.p', "wb" ))
# pickle.dump(list_nonan, open(r'Data\Pcp_Dataframes\df_nonan_PERSIANN_2022.p', "wb" ))
