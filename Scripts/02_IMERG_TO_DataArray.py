# -*- coding: utf-8 -*-
"""
Created on Mon Dec 11 08:52:04 2023

@author: PatricioJavier

Code for converting satellite information from IMERG-ER to a data array containing all the SPPs
for each time step
"""

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
        file_ymd = int(filename[-40:-32])
        file_hour = int(filename[-30:-26])
        # Convert the extracted components to a single numeric date
        file_date = int(f"{file_ymd:04d}{file_hour:04d}")

        if startdate_int <= file_date <= enddate_int:
            filtered_files.append(filename)

    return filtered_files


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
rect_ext = box(ext[0], ext[1], ext[2], ext[3])
rect_gdf = gpd.GeoDataFrame(geometry=[rect_ext], crs="EPSG:4326")

####### Reading the satellite precipitation files
#Set of files in the directory
folder_files = r'D:\IMERG\RAWSatellitePrecipitation'
pattern = "*.RT-H5"
file_list = glob.glob(os.path.join(folder_files, pattern), recursive=True)
file_list.sort()
direccion_principal = r'D:\IMERG\RAWSatellitePrecipitation'
folder = os.getcwd()

# Example usage
# Specify the date range
start_date = pd.Timestamp("2023-01-01 00:00")
end_date = pd.Timestamp("2023-12-31 23:30")

filtered_files = filter_files_IMERG(file_list, start_date, end_date)
list_of_Files = []
list_arrays = []

clipped_IMERG = []
list_df = []
# nc_imerg = xr.open_dataset(r'Data\IMERG-ER_Hrly_2019-2023\IMERG-ER_Hrly_2019-2023.nc')
# nc_imerg.variables
#### Reading of IMERG data
# index = 7
# item = filtered_files[7]

start_time = time.time()
total_files = len(filtered_files)
for index, item in enumerate(filtered_files):
    print(index+1, 'out of',total_files)
    try:
        data = h5py.File(item, 'r')
    except (IOError, EOFError) as e:    
        continue
    # groups = [ x for x in data.keys() ]
    # gridMembers = [ x for x in data['Grid'] ]
    # print(groups)
    # print(gridMembers)
    precip = data['/Grid/precipitationUncal'][:]
    precip[precip < 0] = np.nan 
    precip = precip[0,:,:].transpose()
    # precip = np.expand_dims(precip, axis=0)
    # theTime = data['/Grid/time'][:]
    theLats = data['Grid/lat'][:]
    theLons = data['Grid/lon'][:]
    # time_datetime = [datetime.datetime.utcfromtimestamp(t) for t in theTime]
    # time_coord = xr.DataArray(time_datetime, dims=('time'))
    # pcp_data = xr.DataArray(precip, dims=('time', 'lat', 'lon'), coords={'time' : time_coord, 'lat' : theLats, 'lon' : theLons})
    pcp_data = xr.DataArray(precip, dims=('lat', 'lon'), coords={'lat' : theLats, 'lon' : theLons})
    pcp_data.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True) #Define spatial dimensions based on longitude and latitude
    pcp_data.rio.write_crs("epsg:4326", inplace=True) #Define the CRS for the data array
    pcp_data = pcp_data.rio.clip(basin_proj.geometry.apply(mapping), basin_proj.crs, all_touched=True) #Clipping the data array to the study area    
    # pcp_data = pcp_data.rio.clip(rect_gdf.geometry.apply(mapping), rect_gdf.crs, all_touched=True) #Clipping the data array to the study area
    # Adding the timestamp
    file_ymd = int(item[-40:-32])
    file_hour = int(item[-30:-26])
    file_date = str(f"{file_ymd:04d}{file_hour:04d}")
    timestamp = datetime.strptime(file_date, '%Y%m%d%H%M')
    df = pd.DataFrame(pcp_data.values.flatten(), dtype='float32').T
    df['timestamp'] = timestamp
    df.set_index('timestamp', inplace=True)
    list_df.append(df)
    data.close()

df_pcp = pd.concat(list_df, axis=0)
hourly_pcp = df_pcp.resample('H').mean()

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")


####### Save as pickle variable 
pickle.dump(hourly_pcp, open(r'Data\Pcp_Dataframes\SPP_DFrames\DATAFRAME_IMERG_2023.p', "wb" ))
####### Save as pickle variable 
# pickle.dump(dataset, open(r'Data\pickle_IMERG\pcp_Jubones_IMERG.p', "wb" ) )
##Open the pickle variable
# ds1 = pickle.load( open( r'C:\Users\patri\OneDrive\Documentos\MAESTRIA_HIDROLOGIA_UCUENCA\Modelos_de_Pronostico\TFinal_ForecastingModel\Data\Pcp_data_pickle\pcp_H0792_04-06.p', "rb" ) )
####### Generate the csv files for original and resampled dataset
# dataset.to_csv(r'C:\Users\patri\OneDrive\Documentos\MAESTRIA_HIDROLOGIA_UCUENCA\Modelos_de_Pronostico\TFinal_ForecastingModel\Data\Pcp_dataframes\pcp30min_H0792_04-06.csv', index=True)
