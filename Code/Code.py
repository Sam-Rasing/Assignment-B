# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 16:03:29 2025

@author: Sam Rasing
"""

#%% Import Modules

import pandas as pd
import geopandas as gpd
import numpy as np

#%% processing data

df_RI = pd.read_excel('Data/ER_RSK_LST.xlsx', index_col=6)
df_SA = pd.read_excel('Data/AG_LND_SUST_PRXCSS.xlsx', index_col=6)

#%%

# find most recent year 
for i in df_SA.columns[-5:]:
    if i in df_RI.columns[-6:]:
        recent_year = i
    else: 
        break

df_SA = df_SA[[recent_year, 'GeoAreaCode']]
df_RI = df_RI[[recent_year, 'GeoAreaCode']]

#%% Sanity check
print('SANITY CHECKS:\n\n')

print('index:\n\n', df_RI.index,
      '\n\ncolumns:\n\n', df_RI.columns,
      '\n\nshape:\n\n', df_RI.shape,
      '\n\ndata types:\n\n', df_RI.dtypes,
      '\n\nno missing values:\n\n', pd.notna(df_RI).all(),
      '\n\nmissing values:\n\n', pd.isna(df_RI).sum())

print('index:\n\n', df_SA.index,
      '\n\ncolumns:\n\n', df_SA.columns,
      '\n\nshape:\n\n', df_SA.shape,
      '\n\ndata types:\n\n', df_SA.dtypes,
      '\n\nno missing values:\n\n', pd.notna(df_SA).all(),
      '\n\nmissing values:\n\n', pd.isna(df_SA).sum())

check = df_SA.index.isin(df_RI.index)
print('country list is identifal:', np.unique(check) == True)


#%%

df_SA.dropna()
df_RI.dropna()
