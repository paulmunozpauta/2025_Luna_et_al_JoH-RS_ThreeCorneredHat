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
import rasterio
import rasterio.mask
from shapely.geometry import mapping
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
        file_year = int(filename[-31:-27])
        file_month = int(filename[-27:-25])
        file_day = int(filename[-25:-23])
        file_hour = int(filename[-22:-20])

        # Convert the extracted components to a single numeric date
        file_date = int(f"{file_year:04d}{file_month:02d}{file_day:02d}{file_hour:02d}")

        if startdate_int <= file_date <= enddate_int:
            filtered_files.append(filename)

    return filtered_files


# Specify the date range
start_date = pd.Timestamp("2022-01-01 00:00")
end_date = pd.Timestamp("2022-12-31 23:00")

# Specify the base folder containing the data
# Path for data stored in PC: 'C:\Users\patri\OneDrive\Documentos\MAESTRIA_HIDROLOGIA_UCUENCA\DataFusion_TESIS\Datos_Caudal\Data\Test_GSMAP'
folder = os.getcwd()
# base_folder =r'D:\GSMaP'

# # List with all the files in the directory
# zip_files_list = find_zip_files_in_directory(base_folder)

# # Get a list of CSV paths within the specified date range
# csv_paths = filter_files_GSMAP(zip_files_list, start_date, end_date)

# Add the shapefile of Jubones basin
basin_shp = gpd.read_file(r'Data\Shapes_MSF\Jubones\jubonesMSF_catch.shp')
##Get the boundary geometry
basin_geometry = basin_shp.geometry.iloc[0]  # Assuming only one geometry in the shapefile
basin_shp.plot() #To plot the shp
source_crs = CRS.from_string('+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs') #Source CRS: XY coordinates
target_crs = CRS.from_epsg(4326) #To what CRS is going to be projected
basin_proj = basin_shp.to_crs(target_crs)
ext = basin_proj.total_bounds #It creates an array with the bounds in the form (minx, miny, maxx, maxy)

# lon = np.arange(-81.95,-33.85,0.1)
# lat = np.arange(-9.95,13.05,0.1)

# # Initialize an empty DataFrame to store concatenated data
# list_arrays = []
# i = 0
# # sum_data2d = pd.DataFrame(0, index=np.arange(230), columns=np.arange(481))
# # Iterate through each CSV path and concatenate the data
# start_time = time.time()
# # csv_path = csv_paths[1] #Solo para prueba rapida
# for csv_path in csv_paths:
#     i = i+1
#     try:
#         df = pd.read_csv(csv_path)
#         print(i, 'out of', len(csv_paths))
        

#     except (IOError, EOFError) as e:
        
#     #     # Catch specific errors related to file reading or handling
#         continue
    
#     n_lat, n_lon = df[' Lat'].nunique(), df['  Lon'].nunique()
#     data_2d = df.pivot(index=' Lat', columns ='  Lon', values = '  RainRate')
#     # data_3d.index = df[' Lat'].unique()
#     # data_3d.columns = df['  Lon'].unique()
#     # sum_data2d.index = df[' Lat'].unique()
#     # sum_data2d.columns = df['  Lon'].unique()
#     # data_1 = data_2d[:,240:] #for ccs
#     # data_2 = data_2d[:,:240] #for ccs
#     # data_2d_4 = np.hstack((data_1,data_2))
#     # sum_data2d = sum_data2d+data_2d
#     ds = xr.DataArray(data=data_2d, dims=['Latitude', 'Longitude'], coords=[lat, lon]) #Convert to Data Array taking precipitation values
#     #ds = xr.DataArray(data=data_2d, dims=['Latitude', 'Longitude'], coords={'Latitude': df[' Lat'].values[:n_lat], 'Longitude': df['  Lon'].values[:n_lon]}) #Convert to Data Array taking precipitation values
#     ds.rio.set_spatial_dims(x_dim="Longitude", y_dim="Latitude", inplace=True) #Define spatial dimensions based on longitude and latitude
#     ds.rio.write_crs("epsg:4326", inplace=True) #Define the CRS for the data array
#     ds_1 = ds.rio.clip(jubones_shp.geometry.apply(mapping), jubones_shp.crs, all_touched=True) #Clipping the data array to the study area
#     list_arrays.append(ds_1) #Each ds is saved in the list 


# combined_data_array = xr.concat(list_arrays, dim='time') #concatenate the data simulating a time dimension
# #data_non_negative = combined_data_array.where(combined_data_array >= 0)
# cumulative_sum = combined_data_array.sum(dim='time', skipna=False)

# # Save as pickle variable 
# pickle.dump(cumulative_sum, open(folder+"\Data\Test_GSMAP\pickle_data_GSMAP\GSMAP_Accumul_pcp_2022.p", "wb" ) )
cumulative_sum_pick = pickle.load(open( folder + "\Data\pickle_data_GSMAP\GSMAP_Accumul_pcp_2022.p", "rb" ) )

#Plot the accumulated precipitation map
fig, ax = plt.subplots(figsize=(10, 10))
cumulative_sum_pick.plot(ax=ax, cmap = 'GnBu', vmin = 0, vmax = 1750) #Precipitation map
basin_proj.boundary.plot(ax=ax, color='black') #Projected shp of the basin
plt.title('GSMaP Cumulative precipitation map 2022', fontsize=20)
plt.xlabel('Longitude [degrees east]', fontsize = 18)
plt.ylabel('Latitude [degrees north]', fontsize = 18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.show()
print('Precipitación máxima = ',cumulative_sum_pick.max().values,'mm')
print('Precipitación promedio = ',cumulative_sum_pick.mean().values,'mm')
print('Precipitación mínima = ',cumulative_sum_pick.min().values,'mm')



# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Elapsed time: {elapsed_time:.2f} seconds")

fig.savefig('Results/Cumulative_Precipitation_GSMAP/03GSMAP_cumul_pcp_2022.png', bbox_inches='tight', pad_inches = 0.1)

