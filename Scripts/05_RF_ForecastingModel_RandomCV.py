# -*- coding: utf-8 -*-
"""
Created on Tue Dec 19 10:01:31 2023

@author: PatricioJavier
"""
#%% IMPORT LIBRARY
import os
import pickle
import matplotlib.pyplot as plt
import numpy as np
import matplotlib.dates as dates
import datetime
import pandas as pd
import scipy
from scipy.stats import norm
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from copy import deepcopy
import seaborn as sns
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import r2_score
import itertools
import statsmodels
import statsmodels.api
from statsmodels.tsa.stattools import acf, pacf
from statsmodels.tsa.stattools import ccf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from scipy.signal import correlate
import time

#%% FUNCTIONS
def lagged_dataset(arr, num_steps, additional_arr, new_num_steps):
    num_columns = arr.shape[1]
    modified_rows = []
    excluded_data = []
    for i in range(num_steps, arr.shape[0]):
        prev_rows = arr[i - num_steps:i]
        current_row = arr[i]
        new_row = np.concatenate((prev_rows.flatten(), current_row))
        modified_rows.append(new_row)
    result_array = np.array(modified_rows)
    # Slicing the result_array to match the number of rows in modified_additional_arr
    if result_array.shape[0] > additional_arr.shape[0]:
        result_array = result_array[result_array.shape[0] - additional_arr.shape[0]:]

    modified_rows = []
    for i in range(new_num_steps, additional_arr.shape[0]):
        prev_rows = additional_arr[i - new_num_steps:i]
        current_row = additional_arr[i]
        excluded_data.append(current_row[-1])  # Store excluded data
        new_row = np.concatenate((prev_rows.flatten(), current_row[:-1]))  # Exclude last column
        modified_rows.append(new_row)

    modified_additional_arr = np.array(modified_rows)

    # Adjust dimensions by removing rows from result_array or modified_additional_arr
    min_rows = min(result_array.shape[0], modified_additional_arr.shape[0])
    result_array = result_array[-min_rows:]
    modified_additional_arr = modified_additional_arr[-min_rows:]
    excluded_data = np.array(excluded_data)[-min_rows:]

    # Concatenate result_array and modified_additional_arr
    final_result = np.concatenate((result_array, modified_additional_arr), axis=1)

    return final_result, np.array(excluded_data)[:, None]

########### Funcion nueva para LAGS
def lagged_dataset_pron(input_output_data, lags_inputs, lags_output, lead_time):
    # Create a copy of the input DataFrame to avoid modifying the original DataFrame
    df = input_output_data.copy()

    # Get the name of the last column
    last_col = df.columns[-1]

    # Create new columns with the previous cell values for each column except the last one
    for col in df.columns[:-1]:
        for i in range(1, lags_inputs + 1):
            new_col_name = f'{col}_Prev{i}'
            df[new_col_name] = df[col].shift(i)

    # Create new columns with the previous cell values for the last column
    for i in range(1, lags_output + 1):
        new_col_name = f'{last_col}_Prev{i}'
        df[new_col_name] = df[last_col].shift(i)

    # Add a new column by shifting the values of the last column
    df[f'{last_col}_Lead{lead_time}'] = df[last_col].shift(-lead_time) #.shift(lead_time)

    df.dropna(inplace=True)
    return df.iloc[:,:-1], pd.DataFrame(df.iloc[:,-1])
###########

def calculate_hydro_metrics(simulations, evaluation):
    sim_mean = np.mean(simulations, axis=0, dtype=np.float64)
    obs_mean = np.mean(evaluation, dtype=np.float64)

    r_num = np.sum((simulations - sim_mean) * (evaluation - obs_mean),
                   axis=0, dtype=np.float64)
    r_den = np.sqrt(np.sum((simulations - sim_mean) ** 2,
                           axis=0, dtype=np.float64)
                    * np.sum((evaluation - obs_mean) ** 2,
                             dtype=np.float64))
    r = r_num / r_den
    
    nse_num = np.sum((evaluation - simulations)**2)
    nse_den = np.sum((evaluation - obs_mean)**2)
    
    # calculate error in spread of flow alpha
    alpha = np.std(simulations, axis=0) / np.std(evaluation, dtype=np.float64)
    # calculate error in volume beta (bias of mean discharge)
    beta = (np.sum(simulations, axis=0, dtype=np.float64)
            / np.sum(evaluation, dtype=np.float64))
    # calculate the Kling-Gupta Efficiency KGE
    kge = 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)
    rmse = np.sqrt(np.mean((evaluation - simulations) ** 2,
                            axis=0, dtype=np.float64))
    pbias = (100 * np.sum(simulations - evaluation, axis=0, dtype=np.float64)
              / np.sum(evaluation))
    nse = 1 - (nse_num / nse_den)
    r2 = 1 - (np.sum((evaluation - simulations)**2) / np.sum((evaluation - np.mean(evaluation))**2))
    return kge, rmse, pbias, nse, r2
np.random.seed(20)
import random
random.seed(20)
#%% IMERG data
'''
Upload the precipitation and runoff data for:
IMERG models
'''
### Select the folder with the data
# os.chdir(r'C:\Users\patri\OneDrive\Documentos\MAESTRIA_HIDROLOGIA_UCUENCA\DataFusion_TESIS\Datos_Caudal')
folder = os.getcwd()
df_pcp = pickle.load(open(r'Data\Pcp_Dataframes\SPP_DFrames\DATAFRAME_IMERG_complete.p', "rb"))
df_pcp = df_pcp.dropna(axis=1, how = 'all')
### Import daily runoff data
df_runoff = pd.read_table(folder+'\Data\Caudal_MSF\PREPROCESSED_RUNOFF_DATA_2019-2023.csv', sep=',', header=0, index_col=0)
df_runoff.index = pd.to_datetime(df_runoff.index)
df_runoff = pd.to_numeric(df_runoff.iloc[:, 0], errors='coerce', downcast='float')

#%% TCH data
'''
Upload the precipitation and runoff data for:
TCH models
'''
### Select the folder with the data
folder = os.getcwd()
# df_pcp = pickle.load(open(r'Data\Fused_pcp\TCH_DFrames\REG02_NoVolumeFilter_DATAFRAME_TCH_complete.p', "rb"))
df_pcp = pickle.load(open(r'Data\Fused_pcp\Kernel_Balanced_TCH_DF\Sensit_Analysis\TCH_Kernel_case2_2023.p', "rb"))
df_pcp = df_pcp.dropna(axis=1, how = 'all')
### Import daily runoff data
df_runoff = pd.read_table(r"Data\Caudal_MSF\Observations_Runoff_2023.csv", sep=',', header=0, index_col=0)
df_runoff.index = pd.to_datetime(df_runoff.index)
df_runoff = pd.to_numeric(df_runoff.iloc[:, 0], errors='coerce', downcast='float')

#%% MSWEP data
'''
Upload the precipitation and runoff data for:
MSWEP models
'''
### Select the folder with the data
folder = os.getcwd()
df_pcp = pickle.load(open(r'Data\Pcp_Dataframes\SPP_DFrames\DATAFRAME_MSWEP_complete.p', "rb"))
df_pcp = df_pcp.dropna(axis=1, how = 'all')
### Import daily runoff data
df_runoff = pd.read_table(r'Data\Caudal_MSF\resampled_3H_RUNOFF_2019-2023.csv', sep=',', header=0, index_col=0)
df_runoff.index = pd.to_datetime(df_runoff.index)
df_runoff = pd.to_numeric(df_runoff.iloc[:, 0], errors='coerce', downcast='float')

# How to resample data
# resampled_df = df_runoff.resample('3H').mean()
# resampled_df.to_csv(r'Data\Caudal_MSF\resampled_3H_RUNOFF_2019-2023.csv', sep=',', index=True)
#%% CONFORMING THE FEATURE SPACE
### Concatenate data in a unique dataframe
complete_df = pd.concat([df_pcp, df_runoff], axis=1)
### Split data into training and testing datasets
# complete_df['Caudal'] = pd.to_numeric(complete_df['Caudal'], errors='coerce')
complete_df = complete_df[~(complete_df.isna().any(axis=1) | (complete_df.lt(0).any(axis=1)))] #Filter for the complete data frame to remove nan or negative values
# input_data_train = np.array(complete_df['2019':'2022'].iloc[:,:-1]) # Input data corresponds to precipitation columns
# input_data_test = np.array(complete_df['2023-01-01 00:00':'2023-12-31 23:00'].iloc[:,:-1])
data_train = complete_df['2023-01-01 00:00':'2023-09-30 23:00']
data_test = complete_df['2023-10-01 00:00':'2023-12-31 23:00']
### Define the output dataframes for training and testing
# output_data_train = np.reshape(np.array(complete_df['2019':'2022'].iloc[:,-1]), # Output data corresponds to the runoff column
#                                 (complete_df['2019':'2022'].shape[0],1))
# output_data_test = np.reshape(np.array(complete_df['2023-01-01 00:00':'2023-12-31 23:00'].iloc[:,-1]),
#                               (complete_df['2023-01-01 00:00':'2023-12-31 23:00'].shape[0],1))
#%% AUTOCORRELATION AND CROSS-CORRELATION ANALYSIS
###### ACF & PACF
lags_acf = 500  # Number of lags to consider
lags_pacf = 20
acf_values = acf(complete_df['Caudal'], nlags=lags_acf)
pacf_values, confint = pacf(complete_df['Caudal'], nlags=lags_pacf, alpha=0.05)

# ccf_values = ccf(sum_pcp_situ, df_runoff['Discharge'].values, adjusted = False)
# Compute confidence intervals
# alpha = 0.05
# nobs = len(sum_pcp_situ)
# z_critical = scipy.stats.norm.ppf(1 - alpha / 2)
# confidence_interval = z_critical / np.sqrt(nobs)

# Plot ACF
fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

plot_acf(complete_df['Caudal'], lags=lags_acf, ax = axes[0], use_vlines=False)
axes[0].set_title(label='', fontsize=18)
axes[0].set_xlabel('Lags (hours)', fontsize=16)
axes[0].set_ylabel('Autocorrelation Function (ACF)', fontsize=15)
axes[0].tick_params(axis='both', labelsize=16)
axes[0].annotate("a)", xy=(0.03, 0.03), xycoords='axes fraction', 
                fontsize=28, fontweight='bold', ha='left', va='bottom', 
                bbox=dict(facecolor='white', edgecolor='none', pad=1.0, alpha=0.5))



# Plot PACF
# plot_pacf(complete_df['Caudal'], lags=lags_pacf, ax = axes[1], alpha=0.05, method='yw')
# axes[1].set_title(label='Partial Autocorrelation Function (PACF)', fontsize=18)
# axes[1].set_xlabel('Lags (hours)', fontsize=16)
# axes[1].set_ylabel('Partial Autocorrelation', fontsize=16)
# axes[1].tick_params(axis='both', labelsize=16)

# Plot PACF manually with confidence intervals
axes[1].plot(range(len(pacf_values)), pacf_values, marker='o', linestyle='', color='C0', label="PACF")
axes[1].fill_between(range(len(pacf_values)), confint[:, 0] - pacf_values, confint[:, 1] - pacf_values,
                      color='C0', alpha=0.3, label="95% confidence interval")
axes[1].set_title(label='', fontsize=18)
axes[1].set_xlabel('Lags (hours)', fontsize=16)
axes[1].set_ylabel('Partial Autocorrelation Function (PACF)', fontsize=15)
axes[1].tick_params(axis='both', labelsize=16)
axes[1].annotate("b)", xy=(0.03, 0.03), xycoords='axes fraction', 
                fontsize=28, fontweight='bold', ha='left', va='bottom', 
                bbox=dict(facecolor='white', edgecolor='none', pad=1.0, alpha=0.5))

plt.tight_layout()
# plt.savefig(folder+'\Results\Runoff_Forecasting\Correlations\REG02_NVF_Kernel_ACF_TCH.png', dpi=300, bbox_inches='tight')
# plt.savefig(r'C:\Users\patri\OneDrive\Documentos\MAESTRIA_HIDROLOGIA_UCUENCA\DataFusion_TESIS\Manuscrito\03_Results\IMG\ACF_PACF_mswep.png', dpi=300, bbox_inches='tight')
plt.show()


###### CROSS-CORRELATION PRECIPITATION vs. RUNOFF
# selected_columns = ['Discharge', 'Pluv_0', 'Pluv_1', 'Pluv_2', 'Pluv_3', 'Pluv_4', 'Pluv_5', 'Pluv_6', 'Pluv_7',
#        'Pluv_8', 'Pluv_9', 'Pluv_10', 'Pluv_11', 'Pluv_12', 'Pluv_13', 'Pluv_14',
#        'Pluv_15', 'Pluv_16', 'Pluv_17', 'Pluv_18', 'Pluv_19', 'Pluv_20', 'Pluv_21',
#        'Pluv_22', 'Pluv_23']

# selected_data = complete_df[selected_columns]

import statsmodels.api as sm
# Calculate cross-correlation for each precipitation column
size_plot_ccr = [10,6]
cross_corr_values = {}
lags = 50 # Adjust the number of lags as needed
confidence_level = 0.95  # Specify the desired confidence level
for column in complete_df.columns[:-1]:  # Skip the first column ('H0792')
    cross_corr_values[column] = sm.tsa.stattools.ccf(complete_df['Caudal'], 
                                                     complete_df[column], 
                                                     adjusted=False)

# Plot cross-correlation for each precipitation column
plt.figure(figsize=size_plot_ccr)
for column, values in cross_corr_values.items():
    plt.plot(values, label=column)
# Customize the appearance
# plt.title('Cross-Correlation between Precipitation and Runoff', fontsize = 20)
plt.xlabel('Lags (hours)', fontsize = 18)
plt.ylabel('Cross-Correlation', fontsize = 18)
plt.tick_params(axis='both', labelsize = 16)
plt.xlim(0, lags)  # Set the x-axis limit to show only the first 100 lags
plt.ylim(0, 0.25)
# plt.legend()
plt.axhline(y=0.15, color='red', linestyle='dashed', label=f'{confidence_level*100}% CI')
plt.grid(True)
# plt.savefig(folder+'\Results\Runoff_Forecasting\Correlations\Reg02_NVF_Kernel_CCF_TCH_redline.png', dpi=300, bbox_inches='tight')
# plt.savefig(r'C:\Users\patri\OneDrive\Documentos\MAESTRIA_HIDROLOGIA_UCUENCA\DataFusion_TESIS\Manuscrito\03_Results\IMG\CCF_tch_kernel_line015.png', dpi=300, bbox_inches='tight')
plt.show()

#############################################

# Calculate cross-correlation for each precipitation column

# lags = 30 # Adjust the number of lags as needed
# confidence_level = 0.95  # Specify the desired confidence level
# cross_corr_mean = sm.tsa.stattools.ccf(complete_df_2['Discharge'], 
#                                                      complete_df_2['Pcp_situ_mean'], 
#                                                      adjusted=False)

# # Plot cross-correlation for each precipitation column
# plt.figure(figsize=size_plot_ccr)
# plt.plot(cross_corr_mean, label=column)

# # Customize the appearance
# plt.title('Cross-Correlation between Precipitation and Discharge', fontsize = 20)
# plt.xlabel('Lag', fontsize = 16)
# plt.ylabel('Cross-Correlation', fontsize = 16)
# plt.xlim(0, lags)  # Set the x-axis limit to show only the first 100 lags
# plt.ylim(0, 0.6)
# plt.tick_params(axis='both', labelsize = 16)
# # plt.legend()
# plt.grid(True)
# # plt.savefig(r'C:\Users\patri\OneDrive\Documentos\MAESTRIA_HIDROLOGIA_UCUENCA\Modelos_de_Pronostico\TFinal_ForecastingModel\Results\CCF_mean_H0326.png', dpi=300, bbox_inches='tight')
# plt.show()
#%% HYPERPARAMETRIZATION

###### For FORECASTINGCASTING MODELS (t>0)
lead_time = 24
input_data_train_lags, output_data_train_lags= lagged_dataset_pron(data_train, 4, 4, lead_time = lead_time)
input_data_test_lags, output_data_test_lags= lagged_dataset_pron(data_test, 4, 4, lead_time = lead_time)
array_output_train_lags = output_data_train_lags.to_numpy()
array_output_test_lags = output_data_test_lags.to_numpy()
df_test_values = pd.DataFrame(data_test.iloc[:, -1]).to_csv(folder + r'\Data\Caudal_MSF\Observations_testing_Runoff.csv', index=True)
### Search of the optimal hyperparameters
start_time = time.time()
param_grid = {
    'min_samples_split': list(range(2, 40, 1)),
    'min_samples_leaf': list(range(1, 30, 1)),
    'max_depth': list(range(5, 100, 1)),
    'n_estimators': list(range(1, 1000, 1)),
    'max_features': ['sqrt', 'log2', 'auto']
}
# Calculate the total number of combinations (Only for grid search)
# total_combinations = len(list(itertools.product(*param_grid.values())))

# total_combinations
# Create a GridSearchCV object
parameter_search = RandomizedSearchCV(estimator=RandomForestRegressor(oob_score=True, n_jobs=-1, warm_start=True),
                                      n_iter=100, param_distributions=param_grid, cv=3, n_jobs=1, scoring='r2', verbose=2)

# Fit the GridSearchCV to your data
parameter_search.fit(input_data_train_lags, output_data_train_lags)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")

# Get the best hyperparameters
best_params = parameter_search.best_params_
best_model = parameter_search.best_estimator_

## Sound to notify the end of the process
# import winsound
# freq = 1000  # Hz
# duration = 3000  # milliseconds
# winsound.Beep(freq, duration)

# Print the best hyperparameters
print("Best Hyperparameters:")
print(best_params)

best_model
simulations_data_train= best_model.predict(input_data_train_lags)
simulations_data_train= np.reshape(simulations_data_train, (-1, 1))
#Prediction on unseen data
simulations_data_test= best_model.predict(input_data_test_lags)
simulations_data_test= np.reshape(simulations_data_test, (-1, 1))
#Nash_Sutcliffe    
r2_test=best_model.score(input_data_test_lags, output_data_test_lags)
r2_train=best_model.score(input_data_train_lags, output_data_train_lags)
print(r2_train,r2_test)

best_model # Best model based on the hyperparametrization process

#%% For MODELING AT TIME t=0 (NO FORECASTING)
# ### Hyperparameter search
# # Reset the input data for training and testing
# input_data_train = np.array(complete_df['2019':'2022'].iloc[:,:-1])
# input_data_test = np.array(complete_df['2023-01-01 00:00':'2023-12-31 23:00'].iloc[:,:-1])

# ###### For modeling at time t = 0
# input_data_train_lags, output_data_train_lags= lagged_dataset(input_data_train, 2, output_data_train,3)
# input_data_test_lags, output_data_test_lags= lagged_dataset(input_data_test, 2, output_data_test,3)
# ### Search of the optimal hyperparameters
# start_time = time.time()
# param_grid = {
#     'min_samples_split': list(range(2, 40, 1)),
#     'min_samples_leaf': list(range(1, 30, 1)),
#     'max_depth': list(range(5, 100, 1)),
#     'n_estimators': list(range(1, 1000, 1)),
#     'max_features': ['sqrt', 'log2', 'auto']
# }
# # Calculate the total number of combinations (Only for grid search)
# # total_combinations = len(list(itertools.product(*param_grid.values())))

# # total_combinations
# # Create a GridSearchCV object
# parameter_search = RandomizedSearchCV(estimator=RandomForestRegressor(oob_score=True, n_jobs=-1, warm_start=True),
#                                       n_iter=20, param_distributions=param_grid, cv=3, n_jobs=1, scoring='r2', verbose=2)

# # Fit the GridSearchCV to your data
# parameter_search.fit(input_data_train_lags, output_data_train_lags)

# end_time = time.time()
# elapsed_time = end_time - start_time
# print(f"Elapsed time: {elapsed_time:.2f} seconds")
# # Get the best hyperparameters
# best_params = parameter_search.best_params_
# best_model = parameter_search.best_estimator_

# ## Sound to notify the end of the process
# import winsound
# freq = 1000  # Hz
# duration = 3000  # milliseconds
# winsound.Beep(freq, duration)

# # Print the best hyperparameters
# print("Best Hyperparameters:")
# print(best_params)

# ### Replace the optimal hyperparameter values
# min_samples_splt=15
# min_samples_lf=6
# max_dpth=30
# n_trees=300
# max_ft='auto'     

# ### Performance metrics for the model
# regr=RandomForestRegressor(bootstrap=True,min_samples_split=min_samples_splt,
#                                max_depth=max_dpth,max_features=max_ft,
#                                min_samples_leaf=min_samples_lf,
#                                n_estimators=n_trees,oob_score=True,n_jobs=-1,
#                                warm_start=True,random_state=22)

# regr=regr.fit(input_data_train_lags, output_data_train_lags)
# simulations_data_train= regr.predict(input_data_train_lags)
# simulations_data_train= np.reshape(simulations_data_train, (-1, 1))
# simulations_data_train

# simulations_data_test= regr.predict(input_data_test_lags)
# simulations_data_test= np.reshape(simulations_data_test, (-1, 1))
# simulations_data_test

# r2_test=regr.score(input_data_test_lags, output_data_test_lags)
# r2_train=regr.score(input_data_train_lags, output_data_train_lags)
# print(r2_train,r2_test)
# kge, rmse, pbias , nse, r2 = calculate_hydro_metrics(simulations_data_test, array_output_test_lags)
# print(f"RMSE: {rmse[0]:.4f}")
# print(f"PBias: {pbias[0]:.4f}")
# print(f"KGE: {kge[0]:.4f}")
# print(f"NSE: {nse:.4f}")
# print(f"R2: {r2:.4f}")



#%% FORECASTING FOR A GIVEN LEAD TIME
min_samples_splt=5
min_samples_lf=29
max_dpth=60
n_trees=112
max_ft='auto'
regr=RandomForestRegressor(bootstrap=True,min_samples_split=min_samples_splt,
                                max_depth=max_dpth,max_features=max_ft,
                                min_samples_leaf=min_samples_lf,
                                n_estimators=n_trees,oob_score=True,n_jobs=-1,
                                warm_start=True,random_state=22)
regr=regr.fit(input_data_train_lags, output_data_train_lags)
#Prediction on training data
simulations_data_train= regr.predict(input_data_train_lags)
simulations_data_train= np.reshape(simulations_data_train, (-1, 1))
#Prediction on unseen data
simulations_data_test= regr.predict(input_data_test_lags)
simulations_data_test= np.reshape(simulations_data_test, (-1, 1))
r2_test=regr.score(input_data_test_lags, output_data_test_lags)
r2_train=regr.score(input_data_train_lags, output_data_train_lags)
print(r2_train,r2_test)
kge, rmse, pbias , nse, r2 = calculate_hydro_metrics(simulations_data_test, array_output_test_lags)
print('LEAD TIME:', lead_time, 'hours')
print(f"RMSE: {rmse[0]:.4f}")
print(f"PBias: {pbias[0]:.4f}")
print(f"KGE: {kge[0]:.4f}")
print(f"NSE: {nse:.4f}")
print(f"R2: {r2:.4f}")

### Visual inspection
simulations_data_test = pd.DataFrame(simulations_data_test, columns=['Pronósticos'], index=complete_df['2023-01-01 00:00':'2023-12-31 23:00'].index[-len(simulations_data_test):])
simulations_data_test

observations_data_test = pd.DataFrame(output_data_test_lags, columns=['Caudal_Lead8'], index=complete_df['2023-01-01 00:00':'2023-12-31 23:00'].index[-len(output_data_test_lags):])
observations_data_test

testing_period = pd.concat([simulations_data_test, observations_data_test], axis=1)
testing_period
# df_test_values = testing_period.to_csv(folder + r'\Results\Runoff_Forecasting\Test_values\IMERG_Runoff_Obs-Sim_24h.csv', index=True)
# df_test_values = testing_period.to_csv(folder + r'\Results\Runoff_Forecasting\Test_values\REG02_NoVolFil_Kernel_TCH_Runoff_Obs-Sim_24h.csv', index=True)
df_test_values = testing_period.to_csv(folder + r'\Results\Runoff_Forecasting\Test_values\MSWEP_Runoff_Obs-Sim_24h.csv', index=True)

# To load the TCH values for graphs
# testing_period = pd.read_csv(folder + r'\Results\Runoff_Forecasting\Test_values\IMERG_Runoff_Obs-Sim_6h.csv', index_col=0, parse_dates=True)
test_period_tch = pd.read_csv(folder + r'\Results\Runoff_Forecasting\Test_values\REG02_NoVolFil_Kernel_TCH_Runoff_Obs-Sim_6h.csv', index_col=0, parse_dates=True)

### Hydrograph comparing observed and simulated values
fig, ax= plt.subplots(figsize=(16, 8))
# Assuming testing_period is a pandas DataFrame with labeled columns
testing_period['Caudal_Lead3'].plot(color='black', linestyle='-', label = 'Observations')
testing_period['Pronósticos'].plot(color='C1', marker='o', linestyle='', markersize=3, alpha = 1, label = 'IMERG Model')
test_period_tch['Pronósticos'].plot(color='C0', marker='o', linestyle='', markersize=3, alpha = 0.5, label = 'TCH Model')
# Adding labels for the legend
plt.legend(fontsize = 20)

# Adding a label to the y-axis
plt.ylabel('Runoff ($m^3/s$)', fontsize = 24)
plt.xticks(fontsize = 22)
plt.yticks(fontsize = 22)
# Adjusting the position of the legend
plt.tight_layout()
plt.show()
# fig.savefig(folder + r'\Results\Runoff_Forecasting\TCH-IMERG_Runoff_timeseries_1h.png')

### Scatter plot Observation vs. Simulation (Replace for a simpler way to represent)
# Assuming testing_period is a pandas DataFrame with labeled columns
fig, ax = plt.subplots(figsize=(9, 9))
# Scatter plot for Observaciones
x = testing_period['Caudal_Lead6']
y = testing_period['Pronósticos']
y_tch = test_period_tch['Pronósticos']
sns.scatterplot(x=x, y=y, color='C1', marker='o', s=50, alpha = 0.5, label='IMERG Model', ax=ax)
sns.scatterplot(x=x, y=y_tch, color='C0', marker='o', s=50, alpha = 0.5, label='TCH Model', ax=ax)
# KDE plot for density
# Assuming x and y are your data arrays
# Concatenate x and y into a single array
data = np.vstack((x, y)).T
# sns.kdeplot(x=x,y=y,cmap='magma', ax=ax, fill=False, thresh=0, levels=10, legend=False)
# sns.kdeplot(x=x,y=y_tch,cmap='magma', ax=ax, fill=False, thresh=0, levels=10, legend=False)
# Add a bisector line (y = x)
min_val = min(x.min(), y.min())
max_val = max(x.max(), y.max())
ax.plot([min_val, max_val], [min_val, max_val], color='black', linestyle='--', label='Bisector line')
# Adding labels for the legend
ax.legend(fontsize = 20)
# Adding a label to the y-axis
ax.set_ylabel('Predicted runoff ($m^3/s$)', fontsize = 24)
ax.set_xlabel('Observed runoff ($m^3/s$)', fontsize = 24)
ax.tick_params(labelsize = 22)
# Show the plot
plt.tight_layout()
plt.show()
# fig.savefig(folder + r'\Results\Runoff_Forecasting\TCH-IMERG_Runoff_scatterplot_Hyper_1h.png')

#%% ACCUMULATION CURVES
# Flow data frames
df_3h = pd.read_csv(folder + r'\Results\Runoff_Forecasting\Test_values\REG02_NoVolFil_Kernel_TCH_Runoff_Obs-Sim_3h.csv', index_col=0, parse_dates=True)
df_6h = pd.read_csv(folder + r'\Results\Runoff_Forecasting\Test_values\REG02_NoVolFil_Kernel_TCH_Runoff_Obs-Sim_6h.csv', index_col=0, parse_dates=True)
df_12h = pd.read_csv(folder + r'\Results\Runoff_Forecasting\Test_values\REG02_NoVolFil_Kernel_TCH_Runoff_Obs-Sim_12h.csv', index_col=0, parse_dates=True)
df_24h = pd.read_csv(folder + r'\Results\Runoff_Forecasting\Test_values\REG02_NoVolFil_Kernel_TCH_Runoff_Obs-Sim_24h.csv', index_col=0, parse_dates=True)



fig, ax = plt.subplots(figsize=(12, 6))
# Valores observados acumulados
observed_cumulative = np.cumsum(output_data_test_lags, axis=0)
ax.plot(observed_cumulative, label='Observed', linewidth=2, )


# Calcular valores simulados acumulados
df1_sim = np.cumsum(df_3h, axis=0)
df6_sim = np.cumsum(df_6h, axis=0)
df11_sim = np.cumsum(df_12h, axis=0)
df24_sim = np.cumsum(df_24h, axis=0)

# Plot
ax.plot(df1_sim, label=f'Simulated (Lead Time 3h)')
ax.plot(df6_sim, label=f'Simulated (Lead Time 6h)')
ax.plot(df11_sim, label=f'Simulated (Lead Time 12h)')
ax.plot(df24_sim, label=f'Simulated (Lead Time 24h)')

# Configurar etiquetas y leyenda
ax.set_xlabel('Time step')
ax.set_ylabel('Accumulated volume (m^3)')
ax.legend()

# Mostrar el gráfico
plt.show()


#%% Testing functions
# BORRAR LUEGO DE PROBAR
########### Funcion nueva para LAGS
def lagged_dataset_pron(input_output_data, lags_inputs, lags_output, lead_time):
    
    #Prameters of the function
    input_output_data = data_test
    lags_inputs = 4
    lags_output = 4
    
    # Create a copy of the input DataFrame to avoid modifying the original DataFrame
    df = input_output_data.copy()

    # Get the name of the last column
    last_col = df.columns[-1]

    # Create new columns with the previous cell values for each column except the last one
    for col in df.columns[:-1]:
        for i in range(1, lags_inputs + 1):
            new_col_name = f'{col}_Prev{i}'
            df[new_col_name] = df[col].shift(i)

    # Create new columns with the previous cell values for the last column
    for i in range(1, lags_output + 1):
        new_col_name = f'{last_col}_Prev{i}'
        df[new_col_name] = df[last_col].shift(i)

    # Add a new column by shifting the values of the last column
    df[f'{last_col}_Lead{lead_time}'] = df[last_col].shift(-lead_time) #.shift(lead_time)

    df.dropna(inplace=True)
    return df.iloc[:,:-1], pd.DataFrame(df.iloc[:,-1])


