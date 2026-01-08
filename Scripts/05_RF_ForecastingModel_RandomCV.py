# -*- coding: utf-8 -*-
"""
Random Forest Runoff Forecasting Model.

Uses lagged precipitation and runoff features to predict future runoff.
Includes hyperparameter tuning (RandomizedSearchCV).

Author: Patricio Luna Abril
"""

import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import RandomizedSearchCV
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.getcwd()

# Select Input Source: 'IMERG', 'MSWEP', or 'TCH'
INPUT_SOURCE = 'TCH' 

# Run Modes
RUN_HYPERPARAM_SEARCH = False
RUN_FORECAST = True

# Forecasting Params
LAGS_INPUT = 4
LAGS_OUTPUT = 4
LEAD_TIME = 3

# Paths
PATH_RUNOFF_OBS = os.path.join(BASE_DIR, 'Data', 'Caudal_MSF', 'runoff_2019_GIT.csv') # Verify filename!

# ==========================================
# FUNCTIONS
# ==========================================
def create_lagged_features(df_input, lags_in, lags_out, lead_time):
    """Creates lagged features for supervised learning."""
    df = df_input.copy()
    target_col = df.columns[-1] # Assumes last col is target (Runoff)
    
    # Input Lags (Precipitation columns)
    for col in df.columns[:-1]:
        for i in range(1, lags_in + 1):
            df[f'{col}_Prev{i}'] = df[col].shift(i)
            
    # Target Lags (Autoregression)
    for i in range(1, lags_out + 1):
        df[f'{target_col}_Prev{i}'] = df[target_col].shift(i)
        
    # Future Target (The label to predict)
    df[f'{target_col}_Lead{lead_time}'] = df[target_col].shift(-lead_time)
    
    df.dropna(inplace=True)
    
    X = df.iloc[:, :-1] # Features
    y = df.iloc[:, -1]  # Target (Lead)
    return X, y

def calculate_metrics(sim, obs):
    """Calculates KGE, RMSE, NSE, PBias, R2."""
    obs = np.array(obs).flatten()
    sim = np.array(sim).flatten()
    
    mean_obs = np.mean(obs)
    mean_sim = np.mean(sim)
    
    # Correlation
    r = np.corrcoef(obs, sim)[0, 1]
    # Alpha (Spread)
    alpha = np.std(sim) / np.std(obs)
    # Beta (Bias)
    beta = mean_sim / mean_obs
    
    kge = 1 - np.sqrt((r - 1)**2 + (alpha - 1)**2 + (beta - 1)**2)
    rmse = np.sqrt(np.mean((obs - sim)**2))
    pbias = 100 * np.sum(sim - obs) / np.sum(obs)
    nse = 1 - (np.sum((obs - sim)**2) / np.sum((obs - mean_obs)**2))
    r2 = r**2
    
    return kge, rmse, pbias, nse, r2

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    
    # 1. Load Data based on Configuration
    print(f"Loading data for source: {INPUT_SOURCE}")
    
    if INPUT_SOURCE == 'IMERG':
        p_path = os.path.join(BASE_DIR, 'Data', 'Pcp_Dataframes', 'SPP_DFrames', 'DATAFRAME_IMERG_2019_GIT.p')
        df_pcp = pickle.load(open(p_path, "rb")).dropna(axis=1, how='all')
        
    elif INPUT_SOURCE == 'MSWEP':
        p_path = os.path.join(BASE_DIR, 'Data', 'Pcp_Dataframes', 'SPP_DFrames', 'DATAFRAME_MSWEP_2019_GIT.p')
        df_pcp = pickle.load(open(p_path, "rb")).dropna(axis=1, how='all')
        
    elif INPUT_SOURCE == 'TCH':
        # Ensure this path matches your Kernel Output if using Kernel TCH
        p_path = os.path.join(BASE_DIR, 'Data', 'Fused_pcp', 'Kernel_Balanced_TCH_DF', 'TCH_Kernel_2019_GIT.p')
        # If file doesn't exist, try the one we just generated in Block 3
        if not os.path.exists(p_path):
             p_path = os.path.join(BASE_DIR, 'Data', 'Fused_pcp', 'Kernel_Balanced_TCH_DF', 'TCH_Kernel_2019_GIT.p')
        
        df_pcp = pickle.load(open(p_path, "rb")).dropna(axis=1, how='all')

    # Load Runoff
    df_runoff = pd.read_csv(PATH_RUNOFF_OBS, index_col=0, parse_dates=True)
    df_runoff = pd.to_numeric(df_runoff.iloc[:, 0], errors='coerce').to_frame(name='Caudal')
    
    # Concatenate & Clean
    complete_df = pd.concat([df_pcp, df_runoff], axis=1)
    complete_df = complete_df.dropna() # Remove NaNs
    
    # Split Train/Test
    data_train = complete_df['2019-01-01 00:00:00':'2019-01-01 16:00:00']
    data_test = complete_df['2019-01-01 17:00:00':'2019-01-01 23:00:00']
    
    # Prepare Lags
    X_train, y_train = create_lagged_features(data_train, LAGS_INPUT, LAGS_OUTPUT, LEAD_TIME)
    X_test, y_test = create_lagged_features(data_test, LAGS_INPUT, LAGS_OUTPUT, LEAD_TIME)
    
    # 2. Hyperparameter Search (Optional)
    if RUN_HYPERPARAM_SEARCH:
        print("Running Hyperparameter Search...")
        param_grid = {
            'n_estimators': [100, 300, 500],
            'max_depth': [10, 30, 50],
            'min_samples_leaf': [1, 5, 10],
            'max_features': ['auto', 'sqrt', 'log2']
        }
        rf = RandomForestRegressor(n_jobs=-1, random_state=22)
        search = RandomizedSearchCV(rf, param_grid, n_iter=10, cv=3, verbose=1, scoring='r2')
        search.fit(X_train, y_train)
        best_model = search.best_estimator_
        print(f"Best Params: {search.best_params_}")
        
    else:
        # 3. Training with Fixed Params (provided by user)
        print("Training Random Forest...")
        best_model = RandomForestRegressor(
            n_estimators=112,
            max_depth=60,
            min_samples_leaf=29,
            min_samples_split=5,
            n_jobs=-1,
            random_state=22
        )
        best_model.fit(X_train, y_train)
        
    # 4. Forecasting & Evaluation
    if RUN_FORECAST:
        print("Forecasting...")
        # Train metrics
        y_pred_train = best_model.predict(X_train)
        kge_tr, rmse_tr, _, nse_tr, r2_tr = calculate_metrics(y_pred_train, y_train)
        print(f"TRAIN -> R2: {r2_tr:.3f}, NSE: {nse_tr:.3f}")
        
        # Test metrics
        y_pred_test = best_model.predict(X_test)
        kge, rmse, pbias, nse, r2 = calculate_metrics(y_pred_test, y_test)
        
        print("-" * 30)
        print(f"TEST RESULTS (Lead Time: {LEAD_TIME}h)")
        print("-" * 30)
        print(f"R2:    {r2:.4f}")
        print(f"NSE:   {nse:.4f}")
        print(f"KGE:   {kge:.4f}")
        print(f"RMSE:  {rmse:.4f}")
        print(f"PBias: {pbias:.4f}")
        print("-" * 30)
        
        # Plotting
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.index, y_test.values, label='Observed', color='black')
        plt.plot(y_test.index, y_pred_test, label=f'Simulated ({INPUT_SOURCE})', color='orange', alpha=0.7)
        plt.title(f'Runoff Forecasting - Lead Time {LEAD_TIME}h', fontsize=16)
        plt.ylabel('Runoff [m3/s]')
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.show()
