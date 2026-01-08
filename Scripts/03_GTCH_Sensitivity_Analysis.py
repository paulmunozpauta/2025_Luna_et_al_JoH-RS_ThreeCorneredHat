# -*- coding: utf-8 -*-
"""
GTCH Sensitivity Analysis.

Compares regularization methods for the Covariance Matrix:
1. Tikhonov Regularization (Ridge)
2. Ledoit-Wolf Shrinkage

Author: Patricio Luna Abril
"""

import os
import time
import pickle
import warnings
import numpy as np
import pandas as pd
from scipy import optimize
from numpy.linalg import LinAlgError, inv, det, multi_dot, eig
from sklearn.covariance import LedoitWolf

warnings.filterwarnings('ignore')

# ==========================================
# CONFIGURATION
# ==========================================
# SELECT METHOD HERE: 'TIKHONOV' or 'LEDOIT_WOLF'
METHOD = 'LEDOIT_WOLF' 

BASE_DIR = os.getcwd()
INPUT_DIR = os.path.join(BASE_DIR, 'Data', 'Pcp_Dataframes', 'SPP_DFrames')
OUTPUT_DIR = os.path.join(BASE_DIR, 'Data', 'Fused_pcp', 'TCH_DFrames', 'LWShrink')

# ==========================================
# CORE LOGIC
# ==========================================
def get_regularized_covariance(Y, method):
    """Calculates S matrix based on selected method."""
    
    # SAFETY NET: Handle NaNs (Critical for Ledoit-Wolf)
    if np.any(np.isnan(Y)):
        Y = np.nan_to_num(Y, nan=0.0)
        
    if method == 'LEDOIT_WOLF':
        lw = LedoitWolf()
        lw.fit(Y)
        S = lw.covariance_
        # Micro-regularization for inversion stability
        S = S + 1e-4 * np.eye(S.shape[0])
        return S
    
    elif method == 'TIKHONOV':
        S_cov = np.cov(Y.T)
        reg_factor = 0.01 # As per original script
        S = S_cov + reg_factor * np.eye(S_cov.shape[0])
        return S
    
    else:
        raise ValueError("Unknown Method")

def calculate_R_sensitivity(arr_diffs, method):
    M, N = arr_diffs.shape
    ref = arr_diffs[:, N-1]
    tar = arr_diffs[:, 0:N-1]
    
    # Difference Matrix Y
    Y = tar - np.repeat(np.reshape(ref, (M, 1)), N-1, axis=1)
    
    # GET COVARIANCE (The only difference between the two original scripts)
    S = get_regularized_covariance(Y, method)
    
    # --- Optimization (Identical for both) ---
    u = np.ones((1, N-1))
    R = np.zeros((N, N))
    R[N-1, N-1] = 2 * multi_dot([u, inv(S), u.T])
    x0 = R[:, N-1]
    
    u_vec = np.ones((1, len(S)))
    det_S = det(S)
    inv_S = inv(S)
    denom = np.power(det_S, 2/len(S))
    
    def obj_fun(r):
        f = 0
        r_nn = r[len(S)]
        for j in range(len(S)):
            f += r[j]**2
            for k in range(len(S)):
                if j < k:
                    f += (S[j,k] - r_nn + r[j] + r[k])**2
        return f / np.power(det_S, 2*len(S)) # Note: Original code used power(K, 2*len(S)) or 2/len(S) depending on version. Check Eq. 8.
    
    cons = {'type': 'ineq', 'fun': lambda r: (r[len(S)] - np.dot(
            np.dot(np.reshape(r[0:-1], (1, len(r)-1)) - r[len(S)]*u_vec, inv_S),
            (np.reshape(r[0:-1], (1, len(r)-1)) - r[len(S)]*u_vec).T)) / denom}
    
    res = optimize.minimize(obj_fun, x0, method='COBYLA', tol=2e-10, constraints=cons)
    R[:, N-1] = res.x
    
    for ii in range(N-1):
        for jj in range(N-1):
            if ii <= jj:
                R[ii,jj] = S[ii,jj] - R[N-1,N-1] + R[ii,N-1] + R[jj,N-1]
    
    R += R.T - np.diag(R.diagonal())
    for ii in range(N):
        for jj in range(N):
            R[jj,ii] = R[ii,jj]
            
    return R

# ==========================================
# MAIN
# ==========================================
if __name__ == "__main__":
    print(f"Running Sensitivity Analysis with: {METHOD}")
    
    # 1. Load Data (Simplified loading logic)
    print("Loading pickles...")
    df1 = pickle.load(open(os.path.join(INPUT_DIR, 'DATAFRAME_PERSIANN_2019_GIT.p'), "rb")).dropna(axis=1, how='all')
    df2 = pickle.load(open(os.path.join(INPUT_DIR, 'DATAFRAME_GSMAP_2019_GIT.p'), "rb")).dropna(axis=1, how='all')
    df3 = pickle.load(open(os.path.join(INPUT_DIR, 'DATAFRAME_IMERG_2019_GIT.p'), "rb")).dropna(axis=1, how='all')
    
    common = df1.index.intersection(df2.index).intersection(df3.index)
    df1, df2, df3 = df1.loc[common], df2.loc[common], df3.loc[common]
    
    df_results = pd.DataFrame()
    
    # 2. Loop
    for i in range(len(df1)):
        if (i+1)%500==0: print(f"Step {i+1}...")
        
        p_concat = np.column_stack((df1.iloc[i].values, df2.iloc[i].values, df3.iloc[i].values))
        
        try:
            R = calculate_R_sensitivity(p_concat, METHOD)
            
            # Standard Fusion
            W = inv(R)
            J = np.ones((1, 3))
            term1 = multi_dot([J, W, J.T])
            term2 = multi_dot([J, W, p_concat.T])
            pcp_w = (inv(term1) * term2).T
            
            # Stability Check
            e_val, _ = eig(R)
            cond = abs(e_val.max())/abs(e_val.min())
            if cond > 1e4 or e_val.min() <= -1:
                pcp_w = df3.iloc[i].values.astype(float)
                
        except:
            pcp_w = df3.iloc[i].values.astype(float)
            
        pcp_w = np.where(pcp_w < 0.1, 0, pcp_w)
        
        df_step = pd.DataFrame(pcp_w, index=df3.columns, columns=['Fusion']).T
        df_step['Date'] = df3.index[i]
        df_results = pd.concat([df_results, df_step])
        
    # 3. Save
    df_results.set_index('Date', inplace=True)
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    fname = f"TCH_Sensitivity_{METHOD}_Results_GIT.p"
    pickle.dump(df_results, open(os.path.join(OUTPUT_DIR, fname), "wb"))
    print(f"Saved: {fname}")
