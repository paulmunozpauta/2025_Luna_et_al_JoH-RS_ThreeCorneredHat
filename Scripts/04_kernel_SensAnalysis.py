# -*- coding: utf-8 -*-
"""
Created on Thu Apr 18 19:48:27 2024

@author: PatricioJavier
"""
#%% FUNCTIONS
import os
import netCDF4 as nc
import numpy as np
import pandas as pd
import glob
import scipy
from scipy import signal
import pickle
import pyproj
from pyproj import CRS
import xarray as xr
import geopandas as gpd
from shapely.geometry import mapping
import rioxarray
import datetime
from datetime import datetime

#%% To load the basin shapefile
os.chdir(r'C:\Users\patri\OneDrive\Documentos\MAESTRIA_HIDROLOGIA_UCUENCA\DataFusion_TESIS\Datos_Caudal')
####### Reading the satellite precipitation files
#Set of files in the directory
# folder_files = r'D:\IMERG\RAWSatellitePrecipitation'
# pattern = "*.RT-H5"
# file_list = glob.glob(os.path.join(folder_files, pattern), recursive=True)
# file_list.sort()
# direccion_principal = r'D:\IMERG\RAWSatellitePrecipitation'
# folder = os.getcwd()

basin_shp = gpd.read_file(r'Data\Shapes_MSF\Jubones\jubonesMSF_catch.shp')
basin_geometry = basin_shp.geometry.iloc[0]  #Get the boundary geometry
basin_shp.plot() #To plot the shp
##Projecting the shapefile to EPSG: 4326
source_crs = CRS.from_string('+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs') #Source CRS: XY coordinates
target_crs = CRS.from_epsg(4326) #To what CRS is going to be projected
basin_proj = basin_shp.to_crs(target_crs)
basin_proj.plot() #To plot the shp
ext = basin_proj.total_bounds #It creates an array with the bounds in the form (minx, miny, maxx, maxy)

#%% FILTER FILES AND INITIATE VARIABLES
# Specify the date range
start_date = pd.Timestamp("2019-01-01 00:00")
end_date = pd.Timestamp("2023-12-31 23:00")

df_pcp_uncal = pd.DataFrame() 
#For precipitation (calibrated, uncalibrated) and random error for the neighborhood 
df_pcp_uncal_n_neighbors = pd.DataFrame() 
#For storing continuous information for each pixel
pcp_map_Cal = []
#For storing continuous information for each neighborhood
pcp_map_Cal_n_neighbors = []
#Definition of cells to be considered for calculating the average neighborhood value
# kernel = np.array([[1,1,1],
#                    [1,1,1],
#                    [1,1,1]]) 

kernel = np.array([[1,1],
                   [1,1]]) 


df_TCH = pickle.load(open(r"Data\Fused_pcp\TCH_DFrames\LWShrink\caso_10000_TCH_reg_LW_DATAFRAME_complete.p", "rb"))
df_subset_TCH = df_TCH.loc[start_date:end_date]
# Resample to IMERG grid
dsIMERG = pickle.load(open(r'Data\pickle_IMERG\gridref_IMERG.p', "rb"))
lons=dsIMERG.lon
lats=dsIMERG.lat
index_dsIMERG = np.arange(dsIMERG.shape[0]*dsIMERG.shape[1])

#%% KERNEL PROCESS
# Loop for reading all maps and storing pixel infromation as timeseries in a dataframe
# i = 1
for i in range(len(df_subset_TCH)):
    print("Map", i+1, "from", len(df_subset_TCH))
    df_timestep = []
    array_flat = df_subset_TCH.iloc[i].T
    array_flat = array_flat.reindex(index_dsIMERG)
    array_rshp = array_flat.values.reshape(dsIMERG.shape)
    pcp_array = xr.DataArray(array_rshp, coords={'lat': lats, 'lon': lons}, dims=['lat', 'lon'])
    pcp_array.rio.write_crs("epsg:4326", inplace=True) #Define the CRS for the data array
    pcp_array = pcp_array.rio.clip(basin_proj.geometry.apply(mapping), basin_proj.crs, all_touched=True) #Clipping the data array to the study area
    rows = np.size(pcp_array,0)
    columns = np.size(pcp_array,1)      

# Perform 2D convolution with input data and kernel 

    pcp_array_n_neighbors = scipy.signal.convolve2d(pcp_array, kernel, boundary='wrap', mode='valid')
    pcp_array_n_neighbors = pcp_array_n_neighbors[0:rows:3,0:rows:3]/kernel.sum() 

    rows_n_neighbors = np.size(pcp_array_n_neighbors,0)
    columns_n_neighbors = np.size(pcp_array_n_neighbors,1)

    # Retrieving date information from filename attributes
    # date_att = getLine(pcp_array.FileHeader, 10)
    # year = str(date_att[21:25])
    # month = str(date_att[26:28])
    # day = str(date_att[29:31])
    # hour = str(date_att[32:34])
    # minute = str(date_att[35:37])
    # second = str(date_att[38:40])

    # Converting date (string format) into date.time format for dataframe indexing
    date = df_subset_TCH.index[i]

    pcp_map_Cal = []
    for i in range(rows):
        for j in range(columns):
            value = pcp_array[i,j]
            pcp_map_Cal = np.append(pcp_map_Cal,value)

    pcp_map_Cal_n_neighbors = []
    for i in range(rows_n_neighbors):
        for j in range(columns_n_neighbors):
            value = pcp_array_n_neighbors[i,j]
            pcp_map_Cal_n_neighbors = np.append(pcp_map_Cal_n_neighbors,value)


    df_timestep_Cal = pd.DataFrame(pcp_map_Cal.reshape(-1,len(pcp_map_Cal)))
    df_timestep_Cal = df_timestep_Cal.set_index(pd.DatetimeIndex([date]))

    df_timestep_cal_n_neighbors = pd.DataFrame(pcp_map_Cal_n_neighbors.reshape(-1,len(pcp_map_Cal_n_neighbors)))
    df_timestep_cal_n_neighbors = df_timestep_cal_n_neighbors.set_index(pd.DatetimeIndex([date]))

    df_pcp_uncal = df_pcp_uncal.append(df_timestep_Cal)      
    df_pcp_uncal_n_neighbors = df_pcp_uncal_n_neighbors.append(df_timestep_cal_n_neighbors)      

df_pcp_uncal = df_pcp_uncal.sort_index()
# df_pcp_uncal_hourly=df_pcp_uncal.resample('1h').sum()

df_pcp_uncal_n_neighbors = df_pcp_uncal_n_neighbors.sort_index()
# df_pcp_uncal_n_neighbors_hourly=df_pcp_uncal_n_neighbors.resample('1h').sum()       
# df_pcp_uncal_n_neighbors_hourly = df_pcp_uncal_n_neighbors_hourly.sort_index()

pickle.dump(df_pcp_uncal_n_neighbors, open(r'Data\Fused_pcp\Kernel_Balanced_TCH_DF\LWShrink\TCH_Kernel_10000_complete.p', "wb"))
