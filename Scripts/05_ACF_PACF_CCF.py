# -*- coding: utf-8 -*-
"""
Hydrological Correlation Analysis (ACF, PACF, CCF).

Performs statistical analysis on Precipitation and Runoff time series:
1. Autocorrelation (ACF) of Runoff.
2. Partial Autocorrelation (PACF) of Runoff.
3. Cross-Correlation (CCF) between Precipitation inputs and Runoff.

Author: Patricio Luna Abril
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.stattools import acf, pacf

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.getcwd()

# Select Input Source: 'IMERG', 'MSWEP', or 'TCH'
INPUT_SOURCE = 'TCH' 

# Analysis Parameters
LAGS_ACF = 500   # For long-term memory check
LAGS_PACF = 20   # For immediate autoregressive identification
LAGS_CCF = 50    # For precip-runoff delay identification
CONFIDENCE_LEVEL = 0.95

# Paths
PATH_RUNOFF_OBS = os.path.join(BASE_DIR, 'Data', 'Caudal_MSF', 'runoff_2019_GIT.csv')
OUTPUT_IMG_DIR = os.path.join(BASE_DIR, 'Results', 'Runoff_Forecasting', 'Correlations')

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # 1. Load Data
    print(f"Loading data for source: {INPUT_SOURCE}...")
    
    if INPUT_SOURCE == 'IMERG':
        p_path = os.path.join(BASE_DIR, 'Data', 'Pcp_Dataframes', 'SPP_DFrames', 'DATAFRAME_IMERG_2019_GIT.p')
        df_pcp = pickle.load(open(p_path, "rb")).dropna(axis=1, how='all')
        
    elif INPUT_SOURCE == 'MSWEP':
        p_path = os.path.join(BASE_DIR, 'Data', 'Pcp_Dataframes', 'SPP_DFrames', 'DATAFRAME_MSWEP_2019_GIT.p')
        df_pcp = pickle.load(open(p_path, "rb")).dropna(axis=1, how='all')
        
    elif INPUT_SOURCE == 'TCH':
        # Default to Kernel processed TCH if available, else raw TCH
        p_path = os.path.join(BASE_DIR, 'Data', 'Fused_pcp', 'Kernel_Balanced_TCH_DF', 'TCH_Kernel_2019_GIT.p')
        if not os.path.exists(p_path):
             p_path = os.path.join(BASE_DIR, 'Data', 'Fused_pcp', 'Kernel_Balanced_TCH_DF', 'TCH_Kernel_2019_GIT.p')
        df_pcp = pickle.load(open(p_path, "rb")).dropna(axis=1, how='all')

    # Load Runoff
    df_runoff = pd.read_csv(PATH_RUNOFF_OBS, index_col=0, parse_dates=True)
    df_runoff = pd.to_numeric(df_runoff.iloc[:, 0], errors='coerce').to_frame(name='Caudal')
    
    # Concatenate & Clean
    complete_df = pd.concat([df_pcp, df_runoff], axis=1).dropna()
    print(f"Data ready. Time steps: {len(complete_df)}")

    # Ensure output directory exists
    os.makedirs(OUTPUT_IMG_DIR, exist_ok=True)

    # ==========================================
    # 2. ACF & PACF Analysis (Runoff Memory)
    # ==========================================
    print("Calculating ACF and PACF...")
    
    # Calculate stats
    acf_values = acf(complete_df['Caudal'], nlags=LAGS_ACF)
    pacf_values, confint = pacf(complete_df['Caudal'], nlags=LAGS_PACF, alpha=1-CONFIDENCE_LEVEL)
    
    # Plotting
    fig, axes = plt.subplots(1, 2, figsize=(15, 5), sharey=True)

    # A) Plot ACF
    sm.graphics.tsaplots.plot_acf(complete_df['Caudal'], lags=LAGS_ACF, ax=axes[0], use_vlines=False)
    axes[0].set_title('')
    axes[0].set_xlabel('Lags (hours)', fontsize=16)
    axes[0].set_ylabel('Autocorrelation Function (ACF)', fontsize=15)
    axes[0].tick_params(axis='both', labelsize=16)
    axes[0].annotate("a)", xy=(0.03, 0.03), xycoords='axes fraction', 
                    fontsize=28, fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))

    # B) Plot PACF (Manual plot for better control)
    axes[1].plot(range(len(pacf_values)), pacf_values, marker='o', linestyle='', color='C0', label="PACF")
    # Confidence Interval Fill
    axes[1].fill_between(range(len(pacf_values)), 
                         confint[:, 0] - pacf_values, 
                         confint[:, 1] - pacf_values,
                         color='C0', alpha=0.3, label="95% Confidence Interval")
    
    axes[1].set_xlabel('Lags (hours)', fontsize=16)
    axes[1].set_ylabel('Partial Autocorrelation Function (PACF)', fontsize=15)
    axes[1].tick_params(axis='both', labelsize=16)
    axes[1].annotate("b)", xy=(0.03, 0.03), xycoords='axes fraction', 
                    fontsize=28, fontweight='bold', bbox=dict(facecolor='white', edgecolor='none', alpha=0.5))
    
    plt.tight_layout()
    
    # Save ACF/PACF
    out_acf = os.path.join(OUTPUT_IMG_DIR, f'{INPUT_SOURCE}_ACF_PACF_GIT.png')
    plt.savefig(out_acf, dpi=300, bbox_inches='tight')
    print(f"ACF/PACF plot saved to: {out_acf}")
    plt.show()

    # ==========================================
    # 3. Cross-Correlation Analysis (Precip vs Runoff)
    # ==========================================
    print("Calculating Cross-Correlation (CCF)...")
    
    cross_corr_values = {}
    
    # Iterate over all precipitation columns (excluding the last column which is Runoff)
    # Note: complete_df.columns[:-1] assumes 'Caudal' is the last column
    for column in complete_df.columns[:-1]:
        cross_corr_values[column] = sm.tsa.stattools.ccf(
            complete_df['Caudal'], 
            complete_df[column], 
            adjusted=False
        )

    # Plot CCF
    plt.figure(figsize=(10, 6))
    
    for column, values in cross_corr_values.items():
        plt.plot(values[:LAGS_CCF], label=column, alpha=0.6, linewidth=1)

    plt.xlabel('Lags (hours)', fontsize=18)
    plt.ylabel('Cross-Correlation', fontsize=18)
    plt.tick_params(axis='both', labelsize=16)
    plt.xlim(0, LAGS_CCF)
    plt.ylim(0, 0.3) # Adjusted for typical correlation values, change if needed
    
    # Significance Line (e.g., 0.15 or 1.96/sqrt(N))
    # Using fixed 0.15 as per original script logic for visual threshold
    plt.axhline(y=0.15, color='red', linestyle='dashed', linewidth=2, label='Significance Threshold')
    
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    
    # Save CCF
    out_ccf = os.path.join(OUTPUT_IMG_DIR, f'{INPUT_SOURCE}_CCF_Analysis_GIT.png')
    plt.savefig(out_ccf, dpi=300, bbox_inches='tight')
    print(f"CCF plot saved to: {out_ccf}")
    plt.show()
