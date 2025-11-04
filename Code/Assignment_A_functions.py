# -*- coding: utf-8 -*-
"""
Created on Sun Oct 19 13:57:40 2025

@author: Sam
"""

import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import lilliefors
from scipy.spatial.distance import mahalanobis
from scipy.stats import spearmanr, pearsonr, chi2, norm
import matplotlib.pyplot as plt

#%% Importing Data

def importing_data(data_folder: str, file_name: str, index: str, columns: list, var_name: str):
    """
    Import SDG data from an Excel file and run sanity check.
    
    Parameters
    ----------
    data_folder : str
        The name of the folder in which the data file is placed
    file_name : str
        The name of the file
    index : str
        The name of the index header
    columns : list of str
        A list of used column headers.
    Returns
    -------
    DataFrame
        A DataFrame with all the necessary columns, and the right index.
    Examples
    --------
    >>> importing_data([data, file, countries, columns])
    DataFrame
    """        
    
    df = None
    df = pd.read_excel(data_folder + file_name, 
                       index_col=index, 
                       usecols=lambda col: col in columns)

    print(f'\n=== SANITY CHECK: {var_name} ===\n',
          '\nNUMBER OF ROWS:', len(df),
          '\nNUMBER OF COLUMNS:', len(df.columns),
          '\nCOLUMNS:', df.columns.tolist(),
          '\nMISSING VALUES:', df.isna().sum().sum())
    
    return df

#%% Filtering Data

def filter_data(var1: pd.DataFrame, var2: pd.DataFrame):
    '''
    Filtering two SDG datasets for most recent year for which data is available.
    
    Parameters
    ----------
    var1 : DataFrame
        The first indicator variable 
    var2 : DataFrame 
        The second indicator variable
    Returns
    -------
    dataframe 1
        A dataframe with all the necessary columns, and the right index.
    dataframe 2
        A dataframe with all the necessary columns, and the right index
    recent_year
        The most recent common year in both datasets
    countries
        A list of common countries
    Examples
    --------
    >>> importing_data([data1, data2])
    data1_filtered
    data2_filtered
    2024
    China, The Netherlads, Belgium
    '''
    
    var1.dropna(inplace=True)
    var1.sort_index(inplace=True)
    
    var2.dropna(inplace=True)
    var2.sort_index(inplace=True)
     
    common_index = var1.index.intersection(var2.index)
    var1 = var1.loc[common_index]
    var2 = var2.loc[common_index]
    
    common_columns = var1.columns.intersection(var2.columns)
    var1 = var1[common_columns]
    var2 = var2[common_columns]
       
    recent_year = min(var1.columns[-1], var2.columns[-1])
    
    common_countries = var1['GeoAreaCode']
        
    var1 = var1[['GeoAreaCode', recent_year]]
    var2 = var2[['GeoAreaCode', recent_year]]  
    
    print('\n=== RESULTS FILTER ===\n',
          '\nCOMMON COUNTRIES: ' + str(len(common_countries)) if var1.index.equals(var2.index) 
                                                              else 'No common countries',
                                                              
          '\nRECENT YEAR: ' + str(recent_year) if var1.columns.equals(var2.columns) 
                                               else 'No common years',
                                                              
          '\nMISSING VALUES:', var1.isna().sum().sum() + var2.isna().sum().sum())
    
    
    return var1, var2, recent_year, common_countries

#%% Data analysis

#lillifors test
def normality(var: pd.Series, var_name: str, var_range: list, alpha: float):
    '''
    Checking for normal distribution of a variable, by visual plot and a lilliefors test.
    
    Parameters
    ----------
    numbers : list of int or float
    A sequence of numeric values.
    Returns
    -------
    float or None
    The mean of all even numbers in the list.
    Returns None if the list contains no even numbers.
    Examples
    --------
    >>> average_of_evens([1, 2, 3, 4, 5])
    3.0
    >>> average_of_evens([1, 3, 5])
    None
    '''
    stat, p_value = lilliefors(var)
       
    mu, sigma = np.mean(var), np.std(var)
    x = np.linspace(min(var), max(var), 200)
    pdf = norm.pdf(x, mu, sigma)   
    
    plt.figure(figsize=(10, 7))
    plt.hist(var, 
             bins=20, 
             density=True, 
             color='skyblue', 
             edgecolor='black', 
             alpha=0.7, 
             label="Data histogram")
    
    plt.plot(x, 
             pdf, 
             'r-', 
             lw=2, 
             label=f'Fitted normal\n(mean={mu:.2f}, sd={sigma:.2f})')
    
    plt.xlim(var_range)
    plt.title(f"Lilliefors Test: stat={stat:.3f}, p={p_value:.3f}")
    plt.xlabel("Data values")
    plt.ylabel("Density")
    plt.legend()
    plt.savefig('../Results/liliefors_test.png')
    plt.grid(alpha=0.3)
    plt.show()
     
    print(f'\n=== LILLIEFORS TEST: {var_name} ===\n',
          '\nSTAT: ', stat, 
          '\nP-VALUE: ', p_value,
          '\nP <= 0.05: data is not normally distributed' if p_value <= alpha 
                                                          else 'p > 0.05: data is normally distributed')
    
    return p_value 















#%%
def Mahalanobis_test(var_1: pd.Series, var_2: pd.Series, alpha: float):
    '''
    **Calculating Mahalanobis Distance**
    
    Compute the average of even numbers in a list. 
    The code is taken from `here <https://stackoverflow.com/questions/46827580/multivariate-outlier-removal-with-mahalanobis-distance>`_ .

    
    Parameters
    ----------
    numbers : list of int or float
    A sequence of numeric values.
    Returns
    -------
    float or None
    The mean of all even numbers in the list.
    Returns None if the list contains no even numbers.
    Examples
    --------
    >>> average_of_evens([1, 2, 3, 4, 5])
    3.0
    >>> average_of_evens([1, 3, 5])
    None
    '''
     
    # Inverse matrix
    covariance_matrix = np.cov(var_1, var_2, rowvar=False)  
    inv_covariance_matrix = np.linalg.inv(covariance_matrix)
    # mean vector 
    mean_vector = np.array([var_1.mean(), var_2.mean()])

    mahalanobis_dist = []
    for i in var_1.index:
        i_vector = (var_1.loc[i], var_2.loc[i])
        distance = mahalanobis(i_vector, mean_vector, inv_covariance_matrix)
        mahalanobis_dist.append(distance)
     
    threshold = chi2.ppf((1 - alpha), 2)
    
    outliers_mask = mahalanobis_dist < threshold
    
    print('\n=== MAHALANOBIS TEST ===\n',
          '\nOUTLIER THRESHOLD: ', threshold,
          '\nMAX MAHALANOBIS DISTANCE:', np.max(mahalanobis_dist),
          '\nTOTAL OUTLIERS:', np.sum(outliers_mask == False))
    
    # Create the plot
    plt.figure(figsize=(10, 6))
    plt.plot(mahalanobis_dist, 
             color='royalblue', 
             marker='o',
             linestyle='',
             label='Mahalanobis Distances')  # 'bo-' means blue circles connected by lines
    
    plt.axhline(y=threshold, 
                color='red', 
                linestyle='--', 
                label='Threshold')
    
    plt.xlim(0, len(mahalanobis_dist))
    plt.xlabel('Observation Number')
    plt.ylabel('Mahalanobis Distance')
    plt.title('Mahalanobis Distances with Threshold')
    plt.legend()
    plt.grid(True)
    plt.savefig('../Results/Mahalanobis_distances.png')
    plt.show()

    # remove outliers based on distance
    return mahalanobis_dist, threshold, outliers_mask

    

#%%
def analysing_data(var1: pd.Series, var2: pd.Series,
                   var1_range: list, var2_range: list,
                   var1_name: str, var2_name: str,
                   alpha: float):   
    '''
    Compute the average of even numbers in a list.
    
    Parameters
    ----------
    numbers : list of int or float
    A sequence of numeric values.
    Returns
    -------
    float or None
    The mean of all even numbers in the list.
    Returns None if the list contains no even numbers.
    Examples
    --------
    >>> average_of_evens([1, 2, 3, 4, 5])
    3.0
    >>> average_of_evens([1, 3, 5])
    None
    '''
    # Testing for normality
    p_values = [normality(var1, var1_name, var1_range, alpha),
                normality(var2, var2_name, var2_range, alpha)] 
       
    # Outlier removal
    mahalanobis_dist, threshold, outliers_mask = Mahalanobis_test(var1, var2, alpha) 
      
    var1_no_outliers = var1[outliers_mask]
    var2_no_outliers = var2[outliers_mask]
     
    # Testing for correlation        
    if p_values[0] <= 0.05 or p_values[1] <= 0.05:  # One or both variables are normal distributed, therefore non-parametric test is selected 
    
        correlation, p_value = spearmanr(var1_no_outliers, var2_no_outliers)
        print('\n=== SPEARMAN TEST ===\n',
              '\nCORRELATION:', correlation,
              '\P-VALUE:', p_value,
              '\nP <= 0.05: Correlation is significant' if p_value <= 0.05 
              else '\nP > 0.05: Correlation is insignificant')
        
    else :
        correlation, p_value = pearsonr(var1_no_outliers, var2_no_outliers) # Both variables are normal distributed, therefore parametric test is selected
        print('\n=== PEARSON TEST ===\n',
              '\nPEARSON Rho:', correlation,
              '\P-value:', p_value,
              '\np <= 0.05: Correlation is significant' if p_value <= 0.05 
              else '\np > 0.05: Correlation is insignificant')
    
    # Fit linear regression (slope and intercept)
    slope, intercept = np.polyfit(var1, var2, 1)
    y_pred = slope * var1 + intercept
    
    plt.scatter(var1, var2, 
                color='royalblue', 
                label='Data points',
                alpha=0.5)
    
    plt.plot(var1, y_pred, 
             color='red', 
             label=f'Linear fit: y={slope:.2f}x+{intercept:.2f}')
    
    plt.xlabel("var 1")
    plt.ylabel("var 2")
    plt.title(f"Spearman correlation: rho = {correlation:.2f}, p = {p_value:.3f}")
    plt.xlim(var1_range)
    plt.ylim(var2_range)
    plt.legend()
    plt.savefig('../Results/scatter.png')
    plt.show()
  
    return p_values, correlation
