# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 16:03:29 2025

@author: Sam Rasing
"""

#%% Import Modules

import pandas as pd
import geopandas as gpd

#%% processing data

df_RI = pd.read_excel('ER_RSK_LST.xlsx')

df_SA = pd.read_excel('AG_LND_SUST_PRXCSS.xlsx')

print('INDEX:\n\n', df.index,
      '\n\nCOLUMNS:\n\n', df.columns,
      '\n\nSHAPE:\n\n', df.shape,
      '\n\nDTYPES:\n\n', df.dtypes,
      '\n\n Not Missing values:\n\n', pd.notna(df).all(),
      '\n\nNo missing values:\n\n', pd.isna(df).sum())

#%%

