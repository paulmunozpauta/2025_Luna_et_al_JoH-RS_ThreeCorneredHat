# -*- coding: utf-8 -*-
"""
Created on Thu Oct 12 15:58:33 2023

@author: Leviathan19931111
Link: https://github.com/Leviathan19931111/Generalized-Three-Cornaer-Hat/blob/main/GTHC
"""
#%% FUNCTIONS
import os
import numpy as np
import pandas as pd
from scipy import optimize
from numpy.linalg import LinAlgError
from numpy.linalg import eig
import matplotlib.pyplot as plt
import time
import pickle
# Array used for the function has M rows (for each time step) and N columns (for each dataset)
def cal_uct(arr): #Function to calculate uncertainty and relative uncertainty
    def my_fun(r): 
        """
        my_fun() is a function that corresponds to the equation 8 of Ferreira et al. (2016)
        The aim is to get a unique solution for matrix R (N x N covariance matrix of individual noises)
        It is based on the Kuhn-Tucker theorem
        """
        f = 0 #Objective function F begins in 0
        for j in range(0, len(S)):
            f = f + np.power(r[j], 2)
            for k in range(0, len(S)):
                if j < k:
                    f = np.power((S[j, k] - r[len(S)] + r[j] + r[k]), 2) + f
        K = np.linalg.det(S) #Value of K based in the formula given by Ferreira et al. (2016)
        F = f / np.power(K, 2 * len(S)) #Calculation of F
        return F

# Array M x N with the differences between datasets
# M: Number of timesteps (rows)
# N: Number of datasets. For TCH -> N = 3
# N-1 represents the last product since Python take index from 0
    M, N = np.shape(arr)

    ref_arr = arr[:, N - 1] # Reference array
    tar_arr = arr[:, 0:N - 1] # Target array

    Y = tar_arr - np.repeat(np.reshape(ref_arr, (M, 1)), N - 1, axis=1) # Calculation of the differences matrix Y

# --- RED DE SEGURIDAD FINAL (CRÍTICO PARA LEDOIT-WOLF) ---
    # ¿Qué pasa si la Referencia TAMBIÉN era NaN en esos 25 puntos?
    # La resta anterior habrá dado (NaN - NaN) = NaN.
    # Ledoit-Wolf fallará si no limpiamos esto.
    if np.any(np.isnan(Y)):
        # Reemplazamos cualquier NaN remanente con 0.0
        # (Asume diferencia cero/perfecta concordancia ante falta de datos)
        Y = np.nan_to_num(Y, nan=0.0)

    S_cov = np.cov(Y.T) # Calculation of the covariance matrix of the series differences (eq. 4)
    ## Regularization AR = A + k*I
    reg_factor = 10000
    S = S_cov + reg_factor*np.eye(N-1)
    u = np.ones((1, N - 1)) # Vector of ones with N-1 columns

    R = np.zeros((N, N)) # An empty R matrix is created 
    R[N - 1, N - 1] = 2 * np.dot(np.dot(u, np.linalg.inv(S)), u.T) # The last element of R matrix, which is rNN (the variance of reference DS) 
                                                                            # based on eq. 10
    # initializing conditions
    x0 = R[:, N - 1]

    # define constraints
    u_ = np.ones((1, len(S))) # S is a square matrix of (N-1)x(N-1)
    det_S = np.linalg.det(S) # Determinant of S
    inv_S = np.linalg.inv(S) # Inverse matrix of S
    Denominator = np.power(det_S, 2 / len(S)) # Denominator of the F function (eq. 8)

    # Constraints of the minimization function
    cons = {'type': 'ineq', 'fun': lambda r: (r[len(S)] - np.dot(
        np.dot(np.reshape(r[0:-1], (1, len(r) - 1)) - r[len(S)] * u_, inv_S),
        (np.reshape(r[0:-1], (1, len(r) - 1)) - r[len(S)] * u_).T)) / Denominator}

    #Minimization of the function
    x = optimize.minimize(my_fun, x0, method='COBYLA', tol=2e-10, constraints=cons)

    R[:, N - 1] = x.x

    for ii in range(0, N - 1):
        for jj in range(0, N - 1):
            if ii < jj or ii == jj:
                R[ii, jj] = S[ii, jj] - R[N - 1, N - 1] + R[ii, N - 1] + R[jj, N - 1]
    R += R.T - np.diag(R.diagonal())
    for ii in range(0, N):
        for jj in range(0, N):
            R[jj, ii] = R[ii, jj]

    uct = np.sqrt(R.diagonal())  # uncertainty
    r_uct = uct / np.mean(abs(arr), axis=0) * 100  # relative uncertainty

    return R

folder_main = os.getcwd()


#%% UPLOADING DATA
os.chdir(r'C:\Users\patri\OneDrive\Documentos\MAESTRIA_HIDROLOGIA_UCUENCA\DataFusion_TESIS\Datos_Caudal')
folder_main = r'C:\Users\patri\OneDrive\Documentos\MAESTRIA_HIDROLOGIA_UCUENCA\DataFusion_TESIS\Datos_Caudal'
### Upload the array containing the all the satellite precipitation products in hourly resolution
# Array with dimensions MxN
# M rows: Time steps
# N columns: each pixel of the basin for each product
folder = folder_main + r'\Data\Pcp_Dataframes'
pcp_array_1 = pickle.load(open(folder + r'\SPP_DFrames\DATAFRAME_PERSIANN_complete.p', "rb" ))
### Fill the gaps with NaN in case they exist
date_range = pd.date_range(start=pcp_array_1.index.min(), end=pcp_array_1.index.max(), freq='H')
pcp_array_1 = pcp_array_1.reindex(date_range)
pcp_array_1 = pcp_array_1.dropna(axis=1, how = 'all')
# pcp_array_1['Date'] = pcp_array_1.Date.apply(lambda x: pd.to_datetime(x,dayfirst=True))
# pcp_array_1.set_index(pcp_array_1['Date'],inplace=True)
# pcp_array_1 = pcp_array_1.drop('Date',1)

pcp_array_2 = pickle.load(open(folder + r'\SPP_DFrames\DATAFRAME_GSMAP_complete.p', "rb" ))
# ### Fill the gaps with NaN in case they exist
# date_range = pd.date_range(start=pcp_array_2.index.min(), end=pcp_array_2.index.max(), freq='H')
# pcp_array_2 = pcp_array_2.reindex(date_range)
pcp_array_2 = pcp_array_2.dropna(axis=1, how = 'all')
####Solo para eliminar datos faltantes en 2023 NO USAR EN CONDICIONES NORMALES!!!!!!
# elements_to_drop = [7625, 7627, 7628, 7637]
# elements_to_drop.sort(reverse=True)
# for index in elements_to_drop:
#     del pcp_array_2[index]
# pcp_array_2['Date'] = pcp_array_2.Date.apply(lambda x: pd.to_datetime(x,dayfirst=True))
# pcp_array_2.set_index(pcp_array_2['Date'],inplace=True)
# pcp_array_2 = pcp_array_2.drop('Date',1)

pcp_array_3 = pickle.load(open(folder + r'\SPP_DFrames\DATAFRAME_IMERG_complete.p', "rb" ))
# ### Fill the gaps with NaN in case they exist
# date_range = pd.date_range(start=pcp_array_3.index.min(), end=pcp_array_3.index.max(), freq='H')
# pcp_array_3 = pcp_array_3.reindex(date_range)
pcp_array_3 = pcp_array_3.dropna(axis=1, how = 'all')
####Solo para eliminar datos faltantes en 2023 NO USAR EN CONDICIONES NORMALES!!!!!!
# elements_to_drop = [7625, 7627, 7628, 7637]
# elements_to_drop.sort(reverse=True)
# for index in elements_to_drop:
#     del pcp_array_3[index]
# pcp_array_3['Date'] = pcp_array_3.Date.apply(lambda x: pd.to_datetime(x,dayfirst=True))
# pcp_array_3.set_index(pcp_array_3['Date'],inplace=True)
# pcp_array_3 = pcp_array_3.drop('Date',1)
#### Create the date range for the data
# start_date = pd.Timestamp("2023-04-01 00:00")
# end_date = pd.Timestamp("2023-04-30 23:00")
# date_range = pd.date_range(start=start_date, end=end_date, freq='1H')
# df_daterange = pd.DataFrame(index=date_range)
##############################################################################
####### Testing the function to calculate uncertainty
# pcp_arrayay M x N with the differences between datasets
# M: Number of timesteps (rows)
# N: Number of datasets. For TCH -> N = 3
# N-1 represents the last product since Python take index from 0
# i = 1
# pcp_1 = pcp_array_1[i]
# pcp_1 = np.where(pcp_1 < 1, 0, pcp_1)
# pcp_2 = pcp_array_2[i]
# pcp_2 = np.where(pcp_2 < 1, 0, pcp_2)
# pcp_3 = pcp_array_3[i]
# pcp_3 = np.where(pcp_3 < 1, 0, pcp_3)
# pcp_concat = np.column_stack((pcp_1, pcp_2, pcp_3))
# M, N = np.shape(pcp_concat)

# ref_pcp_concat = pcp_concat[:, N - 1] # Reference pcp_concatay
# tar_pcp_concat = pcp_concat[:, 0:N - 1] # Target pcp_concatay

# Y = tar_pcp_concat - np.repeat(np.reshape(ref_pcp_concat, (M, 1)), N - 1, axis=1) # Calculation of the differences matrix Y

# S = np.cov(Y.T) # Calculation of the covariance matrix of the series differences (eq. 4)
# u = np.ones((1, N - 1)) # Vector of ones with N-1 columns

# R = np.zeros((N, N)) # An empty R matrix is created 
# R[N - 1, N - 1] = 2 * np.dot(np.dot(u, np.linalg.inv(S)), u.T) # The last element of R matrix, which is rNN (the variance of reference DS) 
#                                                                         # based on eq. 10
# # initial conditions according to eq. 10 from Ferreira et al. (2016)
# x0 = R[:, N - 1]

# # define constraints
# u_ = np.ones((1, len(S))) # S is a square matrix of (N-1)x(N-1)
# det_S = np.linalg.det(S) # Determinant of S
# inv_S = np.linalg.inv(S) # Inverse matrix of S
# Denominator = np.power(det_S, (2 / len(S))) # Denominator of the F function (eq. 8)

# # Constraints of the minimization function according to Eq. 9 from Ferreira et al. (2016)
# cons = {'type': 'ineq', 'fun': lambda r: (r[len(S)] - np.dot(
#     np.dot(np.reshape(r[0:-1], (1, len(r) - 1)) - r[len(S)] * u_, inv_S),
#     (np.reshape(r[0:-1], (1, len(r) - 1)) - r[len(S)] * u_).T)) / Denominator}


# def my_fun(r): 
#     """
#     my_fun() is a function that corresponds to the equation 8 of Ferreira et al. (2016)
#     The aim is to get a unique solution for matrix R (N x N covariance matrix of individual noises)
#     It is based on the Kuhn-Tucker theorem
#     """
    
#     f = 0 #Objective function F begins in 0
#     for j in range(0, len(S)):
#         print(f'j: {j}')
#         f = f + np.power(r[j], 2)
#         for k in range(0, len(S)):
#             print(f'k: {k}')
#             if j < k:
#                 f = np.power((S[j, k] - r[len(S)] + r[j] + r[k]), 2) + f
#     K = np.linalg.det(S) #Value of K based in the formula given by Ferreira et al. (2016)
#     F = f / np.power(K, 2 / len(S)) #Calculation of F
#     return F


# # Minimization of the function, using the function 'my_fun'
# # x0 are the initial conditions according to Eq. 10 from Ferreira et al. (2016)
# x = optimize.minimize(my_fun, x0, method='COBYLA', tol=2e-10, constraints=cons) 

# R[:, N - 1] = x.x

# for ii in range(0, N - 1):
#     for jj in range(0, N - 1):
#         if ii < jj or ii == jj:
#             R[ii, jj] = S[ii, jj] - R[N - 1, N - 1] + R[ii, N - 1] + R[jj, N - 1]
# R += R.T - np.diag(R.diagonal())
# for ii in range(0, N):
#     for jj in range(0, N):
#         R[jj, ii] = R[ii, jj]

# # e_val, e_vec = eig(W)
# uct = np.sqrt(R.diagonal())  # uncertainty
# r_uct = uct / np.mean(abs(pcp_concat), axis=0) * 100  # relative uncertainty


#%% TCH FUSION PROCESS
list_concat_df = []
list_R = []
list_W = []
list_eig = []
list_cond_num = []
df_fusion_complete = pd.DataFrame()
### TCH-based fusion with conditionals for:
    # 1. Matrix singularity
    # 2. Noisy precipitation data
# i = 14017
start_time = time.time()
for i in range(len(pcp_array_1)):
    print (i+1, 'out of', len(pcp_array_1))
    pcp_1 = pcp_array_1.iloc[i].values
    # pcp_1 = np.where(pcp_1 < 0.1, 0, pcp_1)
    # pcp_1 = np.expand_dims(pcp_1, axis=1)
    pcp_2 = pcp_array_2.iloc[i].values
    # pcp_2 = np.where(pcp_2 < 0.1, 0, pcp_2)
    # pcp_2 = np.expand_dims(pcp_2, axis=1)
    pcp_3 = pcp_array_3.iloc[i].values
    # pcp_3 = np.where(pcp_3 < 0.1, 0, pcp_3)
    # pcp_3 = np.expand_dims(pcp_3, axis=1)
    pcp_concat = np.column_stack((pcp_1, pcp_2, pcp_3))
    # try:
    #     pcp_concat = np.column_stack((pcp_1, pcp_2, pcp_3))
    # except ValueError as e:
    #     continue
    
    pcp_weighted = np.zeros_like(pcp_1) # Initialize pcp_weighted
    R = np.identity(pcp_concat.shape[1])
    W = np.identity(pcp_concat.shape[1])
    
    
    ### Fusion thorugh TCH assuming no singularities are found
    R = cal_uct(pcp_concat)
    #### Fusion process
    M, N = np.shape(pcp_concat)
    J = np.ones((1, N)) # Vector of ones corresponding to each product
    W = np.linalg.inv(R) #  Weight matrix: Inverse of variance-covariance matrix
    
    # if np.isnan(W).any():  # Check if W contains NaNs
    #     raise ValueError("Array must not contain infs or NaNs")
    
    ## Weighted precipitation calculation (According to Eq. 16 from Xu et al. (2020))
    term_1 = np.linalg.multi_dot([J, W, J.T]) # First term of the equation (J*W*J.T)
    term_2 = np.linalg.multi_dot([J, W, pcp_concat.T]) # Second term of the equation (J*W*P)
    pcp_weighted = (np.linalg.inv(term_1)*term_2).T # Fused pcp for a specific time step
    pcp_weighted = np.where(pcp_weighted < 0.1, 0, pcp_weighted) #Threshold of 1 mm as minimum precipitation volume
    # pcp_weighted = np.where(pcp_weighted > 30, pcp_3, pcp_weighted) #Threshold of 1 mm as minimum precipitation volume
    e_val, e_vec = eig(R)
    condit_num = e_val.max() / e_val.min()
    # if abs(e_val.max())/1000 > abs(e_val.min()) or e_val.min() <= -1:
    #     pcp_weighted = pcp_3.astype(np.float32)
    
    
    # # Conditional for matrix singularity
    # try:
    #     R = cal_uct(pcp_concat)
    #     #### Fusion process
    #     M, N = np.shape(pcp_concat)
    #     J = np.ones((1, N)) # Vector of ones corresponding to each product
    #     W = np.linalg.inv(R) #  Weight matrix: Inverse of variance-covariance matrix
        
    #     # if np.isnan(W).any():  # Check if W contains NaNs
    #     #     raise ValueError("Array must not contain infs or NaNs")
        
    #     ## Weighted precipitation calculation (According to Eq. 16 from Xu et al. (2020))
    #     term_1 = np.linalg.multi_dot([J, W, J.T]) # First term of the equation (J*W*J.T)
    #     term_2 = np.linalg.multi_dot([J, W, pcp_concat.T]) # Second term of the equation (J*W*P)
    #     pcp_weighted = (np.linalg.inv(term_1)*term_2).T # Fused pcp for a specific time step
    #     pcp_weighted = np.where(pcp_weighted < 0.1, 0, pcp_weighted) #Threshold of 1 mm as minimum precipitation volume
    #     # pcp_weighted = np.where(pcp_weighted > 30, pcp_3, pcp_weighted) #Threshold of 1 mm as minimum precipitation volume
    #     e_val, e_vec = eig(R)
    #     condit_num = e_val.max() / e_val.min()
    #     if abs(e_val.max())/1000 > abs(e_val.min()) or e_val.min() <= -1:
    #         pcp_weighted = pcp_3.astype(np.float32)
    #             # pcp_weighted = pcp_weighted.reshape(-1, 1)
    # except LinAlgError as e:        
    #     pcp_weighted = pcp_3.astype(np.float32)
    #     condit_num = np.nan
    #     e_val = np.array([np.nan])
    #     e_vec = np.array([np.nan])
    #     # pcp_weighted = pcp_weighted.reshape(-1, 1)
    
    df_SPP = pd.DataFrame(pcp_concat, columns=['PERSIANN', 'GSMAP', 'IMERG'], dtype='float32')
    df_tch = pd.DataFrame(pcp_weighted, columns=['Fusion'], dtype='float32')
    df_pcp = pd.concat([df_SPP, df_tch], axis=1)
    df_pcp.index = pcp_array_3.columns
    df_tch.index = pcp_array_3.columns
    df_tch_tposed = df_tch.T 
    df_tch_tposed['Date'] = pcp_array_3.index[i]
    df_fusion_complete = df_fusion_complete.append(df_tch_tposed)
    list_concat_df.append(df_pcp)
    list_R.append(R)
    list_W.append(W)
    list_eig.append(e_val)
    list_cond_num.append(condit_num)
    # WP = np.dot(W, pcp_concat.T)
    # print(WP)

df_concat_spps = np.stack(list_concat_df, axis=0)
df_fusion_complete.set_index('Date', drop=True, inplace=True)

end_time = time.time()
elapsed_time = end_time - start_time
print(f"Elapsed time: {elapsed_time:.2f} seconds")


#%% Save results as pickle variables
pickle.dump(df_concat_spps, open(r'Data\Fused_pcp\TCH_DFrames\Tikhonov_Sens_Analys\TCH_Tikh_newcase7_SPP_results_complete.p', "wb" ))
pickle.dump(df_fusion_complete, open(r'Data\Fused_pcp\TCH_DFrames\Tikhonov_Sens_Analys\TCH_Tikh_newcase7_DATAFRAME_complete.p', "wb" ))


#%% TESTING THE cal_uct function
def my_fun(r): 
        """
        my_fun() is a function that corresponds to the equation 8 of Ferreira et al. (2016)
        The aim is to get a unique solution for matrix R (N x N covariance matrix of individual noises)
        It is based on the Kuhn-Tucker theorem
        """
        f = 0 #Objective function F begins in 0
        for j in range(0, len(S)):
            f = f + np.power(r[j], 2)
            for k in range(0, len(S)):
                if j < k:
                    f = np.power((S[j, k] - r[len(S)] + r[j] + r[k]), 2) + f
        K = np.linalg.det(S) #Value of K based in the formula given by Ferreira et al. (2016)
        F = f / np.power(K, 2 * len(S)) #Calculation of F
        return F

# Array M x N with the differences between datasets
# M: Number of timesteps (rows)
# N: Number of datasets. For TCH -> N = 3
# N-1 represents the last product since Python take index from 0
M, N = np.shape(pcp_concat)

ref_arr = pcp_concat[:, N - 1] # Reference array
tar_arr = pcp_concat[:, 0:N - 1] # Target array

Y = tar_arr - np.repeat(np.reshape(ref_arr, (M, 1)), N - 1, axis=1) # Calculation of the differences matrix

S_cov = np.cov(Y.T) # Calculation of the covariance matrix of the series differences (eq. 4)
## Regularization AR = A + k*I
reg_factor = 0.001
S = S_cov + reg_factor*np.eye(N-1)
u = np.ones((1, N - 1)) # Vector of ones with N-1 columns

R = np.zeros((N, N)) # An empty R matrix is created 
R[N - 1, N - 1] = 2 * np.dot(np.dot(u, np.linalg.inv(S)), u.T) # The last element of R matrix, which is rNN (the variance of reference DS) 
                                                                        # based on eq. 10
# initializing conditions
x0 = R[:, N - 1]

# define constraints
u_ = np.ones((1, len(S))) # S is a square matrix of (N-1)x(N-1)
det_S = np.linalg.det(S) # Determinant of S
inv_S = np.linalg.inv(S) # Inverse matrix of S
Denominator = np.power(det_S, 2 / len(S)) # Denominator of the F function (eq. 8)

# Constraints of the minimization function
cons = {'type': 'ineq', 'fun': lambda r: (r[len(S)] - np.dot(
    np.dot(np.reshape(r[0:-1], (1, len(r) - 1)) - r[len(S)] * u_, inv_S),
    (np.reshape(r[0:-1], (1, len(r) - 1)) - r[len(S)] * u_).T)) / Denominator}

#Minimization of the function
x = optimize.minimize(my_fun, x0, method='COBYLA', tol=2e-10, constraints=cons)

R[:, N - 1] = x.x

for ii in range(0, N - 1):
    for jj in range(0, N - 1):
        if ii < jj or ii == jj:
            R[ii, jj] = S[ii, jj] - R[N - 1, N - 1] + R[ii, N - 1] + R[jj, N - 1]
R += R.T - np.diag(R.diagonal())
for ii in range(0, N):
    for jj in range(0, N):
        R[jj, ii] = R[ii, jj]

uct = np.sqrt(R.diagonal())  # uncertainty
r_uct = uct / np.mean(abs(pcp_concat), axis=0) * 100  # relative uncertainty
