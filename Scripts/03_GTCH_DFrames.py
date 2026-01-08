# -*- coding: utf-8 -*-
"""
Generalized Three-Cornered Hat (GTCH) Fusion.

Main script for fusing 3 precipitation products.
Includes Uncertainty Analysis (Eq. 8 Ferreira et al. 2016).

Based on the implementation by Leviathan19931111 
Source: https://github.com/Leviathan19931111/Generalized-Three-Cornaer-Hat/blob/main/GTHC

Author: Patricio Luna Abril
"""

import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy import optimize
from numpy.linalg import LinAlgError, eig, inv, det, multi_dot

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, 'Data', 'Pcp_Dataframes', 'SPP_DFrames')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Data', 'Fused_pcp', 'TCH_DFrames')

# ==========================================
# TCH FUNCTIONS
# ==========================================
def calculate_uncertainty_R(arr_diffs):
    """
    Calculates Covariance Matrix R using Kuhn-Tucker minimization (Ferreira et al. 2016).
    """
    M, N = arr_diffs.shape # M rows (time), N cols (datasets)
    
    # 1. Compute Difference Covariance (S)
    ref = arr_diffs[:, N-1]
    tar = arr_diffs[:, 0:N-1]
    Y = tar - np.repeat(np.reshape(ref, (M, 1)), N-1, axis=1)
    
    # Safety: Handle NaNs in differences (assume 0 diff if NaN)
    if np.any(np.isnan(Y)):
        Y = np.nan_to_num(Y, nan=0.0)
        
    S_cov = np.cov(Y.T)
    
    # Regularization (Tikhonov default for stability)
    reg_factor = 0.01
    S = S_cov + reg_factor * np.eye(N-1)
    
    # 2. Optimization Setup
    u = np.ones((1, N-1))
    R = np.zeros((N, N))
    
    # Initial Guess (Eq. 10)
    R[N-1, N-1] = 2 * multi_dot([u, inv(S), u.T])
    x0 = R[:, N-1]
    
    # Constants for Constraints
    u_vec = np.ones((1, len(S)))
    det_S = det(S)
    inv_S = inv(S)
    denom = np.power(det_S, 2/len(S))
    
    # Objective Function (Eq. 8)
    def objective_func(r):
        f_val = 0
        r_nn = r[len(S)]
        for j in range(len(S)):
            f_val += r[j]**2
            for k in range(len(S)):
                if j < k:
                    f_val += (S[j,k] - r_nn + r[j] + r[k])**2
        return f_val / np.power(det_S, 2*len(S))
    
    # Constraint (Eq. 9)
    cons = {'type': 'ineq', 'fun': lambda r: (r[len(S)] - np.dot(
            np.dot(np.reshape(r[0:-1], (1, len(r)-1)) - r[len(S)]*u_vec, inv_S),
            (np.reshape(r[0:-1], (1, len(r)-1)) - r[len(S)]*u_vec).T)) / denom}
    
    # Minimize
    res = optimize.minimize(objective_func, x0, method='COBYLA', tol=2e-10, constraints=cons)
    R[:, N-1] = res.x
    
    # Fill R Matrix
    for ii in range(N-1):
        for jj in range(N-1):
            if ii <= jj:
                R[ii, jj] = S[ii, jj] - R[N-1, N-1] + R[ii, N-1] + R[jj, N-1]
    
    # Symmetrize
    R += R.T - np.diag(R.diagonal())
    for ii in range(N):
        for jj in range(N):
            R[jj, ii] = R[ii, jj]
            
    return R

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    
    # 1. Load DataFrames
    print("Loading data...")
    # Helper to load and clean
    def load_df(name):
        path = os.path.join(INPUT_DIR, name)
        df = pickle.load(open(path, "rb"))
        return df.dropna(axis=1, how='all')

    df1 = load_df('DATAFRAME_PERSIANN_2019_GIT.p')
    df2 = load_df('DATAFRAME_GSMAP_2019_GIT.p')
    df3 = load_df('DATAFRAME_IMERG_2019_GIT.p')
    
    # Align Indices (Intersection of available dates)
    common_idx = df1.index.intersection(df2.index).intersection(df3.index)
    df1, df2, df3 = df1.loc[common_idx], df2.loc[common_idx], df3.loc[common_idx]
    
    print(f"Processing {len(common_idx)} time steps...")
    
    list_concat_dfs = [] # To store combined frames
    df_fusion_complete = pd.DataFrame()
    
    start_time = time.time()
    
    # 2. Fusion Loop
    for i in range(len(df1)):
        if (i+1) % 500 == 0: print(f"Step {i+1}...")
        
        # Get pixels for current time step
        p1 = df1.iloc[i].values
        p2 = df2.iloc[i].values
        p3 = df3.iloc[i].values
        
        # Form Matrix (Pixels x 3 Products)
        pcp_concat = np.column_stack((p1, p2, p3))
        
        pcp_weighted = np.zeros_like(p1)
        
        try:
            # Calculate Uncertainty Matrix R
            R = calculate_uncertainty_R(pcp_concat)
            
            # Calculate Weights W = R^-1
            W = inv(R)
            
            # Fusion (Eq. 16 Xu et al. 2020)
            J = np.ones((1, 3))
            term1 = multi_dot([J, W, J.T])
            term2 = multi_dot([J, W, pcp_concat.T])
            pcp_weighted = (inv(term1) * term2).T
            
            # Quality Control (Singularity/Stability Check)
            e_val, _ = eig(R)
            if abs(e_val.max())/1000 > abs(e_val.min()) or e_val.min() <= -1:
                # Fallback to IMERG if unstable
                pcp_weighted = p3.astype(np.float32)
                
        except (LinAlgError, ValueError):
            # Fallback
            pcp_weighted = p3.astype(np.float32)
            
        # Post-process thresholds
        pcp_weighted = np.where(pcp_weighted < 0.1, 0, pcp_weighted)
        
        # Store Results
        # Save fusion column
        df_step = pd.DataFrame(pcp_weighted, index=df3.columns, columns=['Fusion']).T
        df_step['Date'] = df3.index[i]
        df_fusion_complete = pd.concat([df_fusion_complete, df_step])
        
        # Optional: Save full comparison
        # df_comp = pd.DataFrame(pcp_concat, columns=['PERSIANN','GSMAP','IMERG'])
        # df_comp['Fusion'] = pcp_weighted
        # list_concat_dfs.append(df_comp)

    # 3. Finalize and Save
    df_fusion_complete.set_index('Date', inplace=True)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    out_path = os.path.join(OUTPUT_DIR, 'TCH_Fusion_Results_GIT.p')
    with open(out_path, "wb") as f:
        pickle.dump(df_fusion_complete, f)
        
    print(f"Done in {time.time()-start_time:.2f}s. Saved to {out_path}")
