 # -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 12:44:53 2023

@author: PatricioJavier
"""
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import geopandas as gpd
import pyproj
from pyproj import CRS
import rasterio
import rasterio.mask
from shapely.geometry import mapping
import rioxarray
import gzip
import calendar
import time
import datetime
import pickle

def JulianDate_to_MMDDYYY(y,jd):
    month = 1
    day = 0
    while jd - calendar.monthrange(y,month)[1] > 0 and month <= 12:
        jd = jd - calendar.monthrange(y,month)[1]
        month = month + 1
    return jd, month


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
        file_year = int(filename[7:9])
        file_day = int(filename[9:12])
        file_hour = int(filename[12:14])

        # Convert the extracted components to a single numeric date
        file_date = int(f"{file_year:02d}{file_day:03d}{file_hour:02d}")

        if start_date <= file_date <= end_date:
            filtered_files.append(filename)

    return filtered_files

# Add the shapefile of Jubones basin
jubones_shp = gpd.read_file(r'Data\Shapes_MSF\Jubones\jubonesMSF_catch.shp')
jubones_geometry = jubones_shp.geometry.iloc[0]  #Get the boundary geometry
jubones_shp.plot() #To plot the shp
##Projecting the shapefile to EPSG: 4326
source_crs = CRS.from_string('+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs') #Source CRS: XY coordinates
target_crs = CRS.from_epsg(4326) #To what CRS is going to be projected
jubones_proj = jubones_shp.to_crs(target_crs)
ext = jubones_proj.total_bounds #It creates an array with the bounds in the form (minx, miny, maxx, maxy)
file_list = os.listdir(r'D:\PERSIANN-CCS\hrly\2022') #que archivos son los que se van a enlistar
direccion_principal = r'D:\PERSIANN-CCS\hrly\2022' #colocar el directorio donde estan los archivos como direccion ppal
folder = os.getcwd()

# Example usage
# start_date = 2200100
# end_date = 2236523

# filtered_files = filter_files_PERSIANN(file_list, start_date, end_date)
# list_arrays = []

# #Dimensions
# lon = np.arange(-180,180,0.04)
# lat = np.arange(-60,60,0.04)

# start_time = time.time()
# # index = 1
# # item = 'rgccs1h2100201.bin.gz'
# data_sum = xr.DataArray(data=np.empty((3000, 9000)), dims=["lat", "lon"])
# for index, item in enumerate(filtered_files[:]):
#     #print(index, item)
#     print(index+1, 'out of', len(filtered_files))
#     path = os.path.join(direccion_principal, item) #To get the list of files in the directory
#     f=gzip.GzipFile(path)
#     file_content = f.read()
#     data = np.frombuffer(file_content, dtype=np.dtype('>h'))
#     data = data.reshape((3000,9000))
#     data = data[::-1,:]
#     data_1 = data[:,4500:] #for ccs
#     data_2 = data[:,:4500] #for ccs
#     data = np.hstack((data_1,data_2))
#     data= data/100 #for ccs
#     data[data < 0] = np.nan # set values < 0 to nodata
#     # data = np.flipud(data)
#     data = xr.DataArray(data=data, dims=["lat", "lon"], coords=[lat,lon])
#     mask1 = ~np.isnan(data)
#     data_sum = np.nansum([data_sum, data], axis=0)
#     data_sum = xr.DataArray(data=data_sum, dims=["lat", "lon"], coords=[lat,lon])
#     # data.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
#     # data.rio.write_crs("epsg:4326", inplace=True)
#     # data = data.rio.clip(jubones_shp.geometry.apply(mapping), jubones_shp.crs, all_touched=True)
#     # plt.imshow(data,cmap='magma_r', vmin=0.1,vmax=5)
#     # plt.title("Geospatial Raster Plot")
#     # plt.colorbar(label='Data Value')
#     # plt.xlabel("Longitude")
#     # plt.ylabel("Latitude")
#     # plt.show()
#     # data = data.values.flatten()
#     # date_att = str(path)
#     # year = int('20'+str(date_att[-14:-12]))
#     # julian_day = int(str(date_att[-12:-9]))
#     # day, month = JulianDate_to_MMDDYYY(year,julian_day)
#     # hour = int(str(date_att[-9:-7]))
#     # # Converting date (string format) into date.time format for dataframe indexing
#     # date = datetime.datetime(int(year), int(month), day, hour, 0, 0)
    
#     # list_arrays.append(data) #Each ds is saved in the list
    
#     # data = pd.DataFrame(data)
#     # data = data.dropna() #to retain catchment pixels only, not surroundings in full rectangle area
#     # data = data.T
#     # data = data.set_index(pd.DatetimeIndex([date]))
#     # if index==0:
#     #     dataset=data.copy()
#     # else:       
#     #     dataset = dataset.append( data)


# # dataset = dataset.sort_index()
# # dataset = dataset[~dataset.index.duplicated(keep='first')] #avoiding duplicated values, keeping first ones

# # monthly_mean = dataset.resample('D').sum()

# # monthly_mean = monthly_mean.mean(axis=1)

# # yearly_mean = monthly_mean.groupby(monthly_mean.index.month).mean()
# # Operations of SUM and MEAN for the list of data arrays
# #Operations of SUM for the list of data arrays
# # combined_data_array = np.stack(list_arrays, axis=0) #concatenate the data
# # # data_non_negative = np.where(combined_data_array >= 0, combined_data_array, np.nan)
# # cumulative_pcp = np.cumsum(combined_data_array, axis=0)
# # cumulative_pcp
# data_sum.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True)
# data_sum.rio.write_crs("epsg:4326", inplace=True)
# data_sum = data_sum.rio.clip(jubones_proj.geometry.apply(mapping), jubones_proj.crs, all_touched=True)
# data_sum
# # combined_data_array = xr.concat(list_arrays, dim='time') #concatenate the data simulating a time dimension
# # # data_non_negative = combined_data_array.where(combined_data_array >= 0)
# # cumulative_sum = combined_data_array.sum(dim='time', skipna = True)

# # Save as pickle variable 
# pickle.dump(data_sum, open(folder+"\Data\pickle_PERSIANN\PERSIANN_cumul_pcp_2022.p", "wb" ) )
cumulative_sum_pick = pickle.load( open( folder+"\Data\pickle_PERSIANN\PERSIANN_cumul_pcp_2022.p", "rb" ) )


fig, ax = plt.subplots(figsize=(10, 10))
cumulative_sum_pick.plot(ax=ax, cmap = 'GnBu', vmin = 0, vmax = 1750) #Precipitation map
jubones_proj.boundary.plot(ax=ax, color='black') #Projected shp of the basin
plt.title('PERSIANN-CCS Cumulative Precipitation Map 2022', fontsize=20)
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

fig.savefig(r'Results\Cumulative_Precipitation_PERSIANN\newPERSIANN_cumul_pcp_2022.png', bbox_inches='tight', pad_inches = 0.1)
