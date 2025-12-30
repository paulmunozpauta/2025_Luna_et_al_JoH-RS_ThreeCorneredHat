# -*- coding: utf-8 -*-
"""
Created on Mon Jun 26 12:44:53 2023

@author: PatricioJavier
"""
#%% LIBRARIES & FUNCTIONS
import os
import pandas as pd
import netCDF4 as nc
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
import pickle
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

#%% LOADING THE DATA
# Add the shapefile of Jubones basin
jubones_shp = gpd.read_file(r'C:\Users\patri\OneDrive\Documentos\MAESTRIA_HIDROLOGIA_UCUENCA\DataFusion_TESIS\Datos_Caudal\Data\Shapes_MSF\Jubones\jubonesMSF_catch.shp')
##Get the boundary geometry
jubones_geometry = jubones_shp.geometry.iloc[0]  # Assuming only one geometry in the shapefile
jubones_shp.plot() #To plot the shp

#List of NetCDF files in the directory
    #MSWEP v2.2: D:\MSWEP_v220
    #MSWEP v2.8: D:\MSWEP_v280
file_list = os.listdir(r'D:\MSWEP_v280') 
direccion_principal = r'D:\MSWEP_v280'
folder = os. getcwd()
# Example usage
start_date = 20190100 # Numeric format: YYYYJDHH
end_date = 202336521  # Numeric format: YYYYJDHH

filtered_files = filter_files_MSWEP(file_list, start_date, end_date)

##Projecting the shapefile to EPSG: 4326
source_crs = CRS.from_string('+proj=utm +zone=30 +datum=WGS84 +units=m +no_defs') #Source CRS: XY coordinates
target_crs = CRS.from_epsg(4326) #To what CRS is going to be projected
jubones_proj = jubones_shp.to_crs(target_crs)

#%% TO READ, CLIP AND CONCATENATE THE DATA
list_arrays = []

### FOR loop to clip the netcdf file and extract the pixels that intersect the shp
start_time = time.time()
for i in range(0, len(filtered_files)) :
    print(i+1, 'out of', len(filtered_files))
    path = os.path.join(direccion_principal, filtered_files[i]) #To get the list of files in the directory
    ds = xr.open_dataset(path) #Open each file as dataset using xarray
    ds = xr.DataArray(data=ds['precipitation']) #Convert to Data Array taking precipitation values
    ds.rio.set_spatial_dims(x_dim="lon", y_dim="lat", inplace=True) #Define spatial dimensions based on longitude and latitude
    ds.rio.write_crs("epsg:4326", inplace=True) #Define the CRS for the data array
    ds_1 = ds.rio.clip(jubones_shp.geometry.apply(mapping), jubones_shp.crs, all_touched=True) #Clipping the data array to the study area
    
    
    
    list_arrays.append(ds_1) #Each ds is saved in the list

#Accumulation from the files of the list of data arrays
combined_data_array = xr.concat(list_arrays, dim='time') #concatenate the data 
cumulative_sum = combined_data_array.sum(dim='time', skipna=True)
cumsum_w_nan = cumulative_sum.where(cumulative_sum != 0, other=float('nan'))
print(cumsum_w_nan)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

### Save as pickle variable ************** VOLVER A GENERAR EL ARCHIVO 2022 QUE SE SOBREESCRIBIÓ
pickle.dump(cumsum_w_nan, open(folder+"\Data\pickle_MSWEP\MSWEP_Cumul_pcp_2019-2023.p", "wb" ) )
# cumsum_w_nan = pickle.load(open(folder+"\Data\pickle_MSWEP\MSWEP_Cumul_pcp_2023.p", "rb" ) )

#%% PLOTTING THE DATA
### Plotting the accumulation map
fig, ax = plt.subplots(figsize=(12, 12))
cumsum_w_nan.plot(ax=ax, cmap = 'GnBu', vmin=400, vmax=1600) #Precipitation map
jubones_proj.boundary.plot(ax=ax, color='black') #Projected shp of the basin
plt.title('Mapa de precipitación acumulada 2023 (MSWEP)', fontsize=20)
plt.xlabel('Longitud [grados Este]', fontsize = 18)
plt.ylabel('Latitude [grados Norte]', fontsize = 18)
plt.xticks(fontsize = 16)
plt.yticks(fontsize = 16)
plt.show()
print('Precipitación máxima = ',cumsum_w_nan.max().values,'mm')
print('Precipitación promedio = ',cumsum_w_nan.mean().values,'mm')
print('Precipitación mínima = ',cumsum_w_nan.min().values,'mm')


fig.savefig(r'Results\Cumulative_Pcp_MSWEP\Accum_2023\Accum_pcp_map_MSWEP_2023.png', bbox_inches='tight', pad_inches = 0.1)

    ###Trabajando con DATAFRAMES
# mean_precip_time = merged_df.groupby('time').mean() #Promedio de P de toda la cuenca por c/ paso de tiempo
# mean_precip_coord = merged_df.groupby(['lon', 'lat']).mean() #Promedio de P por c/ pixel en todos los pasos de tiempo
# sum_precip_time = merged_df.groupby('time').sum() #Suma de toda la cuenca por c/ paso de tiempo
# sum_precip_coord = merged_df.groupby(['lon', 'lat']).sum() #Suma de c/ pixel en todos los pasos de tiempo
# mean_precip_time.plot(kind='bar', rot=30) #Grafico de precipitacion media de toda la cuenca por c/ paso de tiempo
# mean_precip_time['precipitation'].cumsum().plot() #P acumulada por c/ paso de tiempo en toda la cuenca
# mean_precip_coord.plot(kind='bar', rot=30) #Grafico de precipitacion acumulada de c/ pixel para todos los pasos de tiempo
# mean_precip_coord['precipitation'].cumsum().plot() #P acumulada por c/ pixel para todos los pasos de tiempo
# mean_p_coord_ds = xr.Dataset.from_dataframe(mean_precip_coord)
# sum_p_coord_ds = xr.Dataset.from_dataframe(mean_precip_coord)
