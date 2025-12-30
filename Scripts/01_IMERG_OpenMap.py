# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 12:44:53 2023

@author: PatricioJavier
"""
import os
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
import h5py
import numpy as np
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker
import pyproj
from pyproj import CRS
import xarray as xr
import rasterio
import rasterio.mask
from shapely.geometry import mapping
import rioxarray
import time
import pickle


def filter_files_IMERG(files, start_date, end_date):
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
    startdate_int = int(start_date.strftime('%Y%m%d%H%M'))
    enddate_int = int(end_date.strftime('%Y%m%d%H%M'))

    for filename in files:
        # Extract date components from the filename
        # Assuming the format is YYYYMMDD.HH.nc
        file_year = int(filename[23:27])
        file_month = int(filename[27:29])
        file_day = int(filename[29:31])
        file_hour = int(filename[33:37])

        # Convert the extracted components to a single numeric date
        file_date = int(f"{file_year:04d}{file_month:02d}{file_day:02d}{file_hour:04d}")

        if startdate_int <= file_date <= enddate_int:
            filtered_files.append(filename)

    return filtered_files

###Open the Jubones basin shapefile
jubones_shp = gpd.read_file(r'Data\Shapes_MSF\Jubones\jubonesMSF_catch.shp')
jubones_geometry = jubones_shp.geometry.iloc[0]  #Get the boundary geometry
jubones_shp.plot() #To plot the shp
##Projecting the shapefile to EPSG: 4326
source_crs = CRS.from_string('+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs') #Source CRS: XY coordinates
target_crs = CRS.from_epsg(4326) #To what CRS is going to be projected
jubones_proj = jubones_shp.to_crs(target_crs)
jubones_proj.plot() #To plot the shp
ext = jubones_proj.total_bounds #It creates an array with the bounds in the form (minx, miny, maxx, maxy)

#Set of files in the directory
file_list = os.listdir(r'Data\Test_IMERG')
direccion_principal = r'Data\Test_IMERG'
folder = os.getcwd()

# Example usage
# Specify the date range
start_date = pd.Timestamp("2022-04-02 00:00")
end_date = pd.Timestamp("2022-04-02 23:30")

filtered_files = filter_files_IMERG(file_list, start_date, end_date)
list_of_Files = []
list_arrays = []

# start_time = time.time()
## Verification of uncorrupted files for its processing
for index, item in enumerate(filtered_files):
    # index = 0
    # item = list_of_Files[0]
    cpth = os.path.join(direccion_principal, item) #To get the current path of the file
    print(index+1, 'out of', len(filtered_files))
    
    try:
        f = h5py.File(cpth, 'r')
    
    except (IOError, EOFError) as e:
        
    #     # Catch specific errors related to file reading or handling
        continue

    groups = [ x for x in f.keys() ]
    gridMembers = [ x for x in f['Grid'] ]
    # print(groups)
    # print(gridMembers)
    precip = f['Grid/precipitationUncal'][0][:][:]
    precip = np.transpose(precip)
    theLats = f['Grid/lat'][:]
    theLons = f['Grid/lon'][:]
    x, y = np.float32(np.meshgrid(theLons, theLats))
    
    pcp_array = xr.DataArray(precip, dims=('lat', 'lon'), coords={'lat' : theLats, 'lon' : theLons})
    pcp_array.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True) #Define spatial dimensions based on longitude and latitude
    pcp_array.rio.write_crs("epsg:4326", inplace=True) #Define the CRS for the data array
    pcp_array = pcp_array.rio.clip(jubones_proj.geometry.apply(mapping), jubones_proj.crs, 
                                   all_touched=True) #Clipping the data array to the study area
    
    list_arrays.append(pcp_array) #Each ds is saved in the list

#Operations of SUM for the list of data arrays
combined_data_array = np.stack(list_arrays, axis=0) #concatenate the data
data_non_negative = np.where(combined_data_array >= 0, combined_data_array, np.nan)
cumulative_pcp = np.cumsum(data_non_negative, axis=0)
cumul_pcp_mmhr = cumulative_pcp*0.5
cumul_pcp_mmhr = cumul_pcp_mmhr[-1]
cumul_pcp_mmhr = xr.DataArray(cumul_pcp_mmhr, dims=('lat', 'lon'), coords={'lat' : pcp_array.lat, 'lon' : pcp_array.lon})
cumul_pcp_mmhr
# Save as pickle variable 
# pickle.dump(cumul_pcp_mmhr, open(folder+"\Data\pickle_IMERG\IMERG_Accumul_pcp_2019.p", "wb" ) )
cumulative_sum_pick = pickle.load( open( folder+"\Data\pickle_IMERG\IMERG_Accumul_pcp_2022.p", "rb" ) )

fig, ax = plt.subplots(figsize=(10, 10))
cumulative_sum_pick.plot(ax=ax, cmap = 'GnBu', vmin = 0, vmax = 1500) #Precipitation map
jubones_proj.boundary.plot(ax=ax, color='black') #Projected shp of the basin
plt.title('IMERG-ER Cumulative Precipitation Map 2022', fontsize=20)
plt.xlabel('Longitude [degrees east]', fontsize = 18)
plt.ylabel('Latitude [degrees north]', fontsize = 18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.show()
print('Precipitación máxima = ',cumul_pcp_mmhr.max().values,'mm')
print('Precipitación promedio = ',cumul_pcp_mmhr.mean().values,'mm')
print('Precipitación mínima = ',cumul_pcp_mmhr.min().values,'mm')


# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Elapsed time: {elapsed_time:.2f} seconds")

fig.savefig('Results/Cumulative_Precipitation_IMERG/newIMERG_cumul_pcp_2022.png', bbox_inches='tight', pad_inches = 0.1)
