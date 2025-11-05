# -*- coding: utf-8 -*-
"""
**Assignment_A_function.py**
Created on Sun Oct 19 13:57:40 2025

@author: Sam
"""

# Importing the required modules
import pandas as pd
import numpy as np
from statsmodels.stats.diagnostic import lilliefors
from scipy.stats import spearmanr, pearsonr, chi2, norm
import matplotlib.pyplot as plt

'''
In this code file you find all the code for the functions used in Assignment_A_main.py. 
For each function code documentation is added to provide the user a better understanding of the code 
'''

#%% Importing Data

def importing_data(data_folder: str, file_name: str, index: str, columns: list, var_name: str):
    """
    Import SDG Data

    Load the SDG data from an Excel file obtained from
    `the UN Database <https://unstats.un.org/sdgs/dataportal/database>`_
    and perform basic sanity checks.
    
    Parameters
    ----------
    data_folder : str
        Path to the folder containing the data file.
    file_name : str
        The name of the Excel file.
    index : str
        Column to use as the DataFrame index.
    columns : list of str
        list of column headers to include.
    Returns
    -------
    pd.DataFrame
        A DataFrame containing the selected columns with the specified index.
    Examples
    --------
    >>> importing_data(data_folder, file_name, index, columns])
    Country SDG_score
    CountryA 15
    CountryB 20
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
    Filter SDG data

    Filtering the two SDG datasets for most recent year for which both indicators are available. Countries with missing values for one or
    both indicator are removed.
    
    Parameters
    ----------
    var1 : pd.DataFrame
        The first indicator dataset 
    var2 : pd.DataFrame 
        The second indicator dataset
    Returns
    -------
    dataframe 1: pd.DataFrame
        Filtered version of the first dataset based on most recent year and correct index.
    dataframe 2: pd.DataFrame
        Filtered version of the second dataset based on most recent year and correct index.
    recent_year: int
        The most recent common year present in both datasets.
    countries: list of str
        List of countries present in both datasets.
    Examples
    --------
    >>> filter_data(var1, var2)
    filtered_var1, filtered_var2, recent_year, common_countries = filter_sdg_data(data1, data2)
    recent_year
    2024
    common_countries
    'China', 'The Netherlands', 'Belgium'
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

#%% Normality check

def normality(var: pd.Series, var_name: str, var_range: list, alpha: float):
    '''
    Normality Check

    Assess whether a variable follows a normal distribution using a visual plot and a Lilliefors test.
    The assessment is based on a specified significance level (alpha).
    
    Parameters
    ----------
    var: pd.Series
        The country values of SDG indicator to test for normality.
    var_name: str
        The name of the variable.
    var_range: list
        The minimum and maximum possible value of the SDG indicator.
    alpha: float
        The significance level for the lillieforst test.
    
    Returns
    -------
    p_value: float
        The obeserved p-value from the lilliefors test.
    
    Examples
    --------
    >>> p_value = normality(variable_1, 'name_1', [0, 10], 0.5)
    p_value
    0.019
    >>> p_value = normality(variable_2, 'name_2', [1, 2], 0.01)
    p_value
    0.59
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
             label=f'Data histogram (Lilliefors Test: stat={stat:.3f}, p={p_value:.3f})')
    
    plt.plot(x, 
             pdf, 
             'r-', 
             lw=2, 
             label=f'Fitted normal curve\n(mean={mu:.2f}, sd={sigma:.2f})')
    
    plt.xlim(var_range)
    plt.xlabel(var_name)
    plt.ylabel('Density')
    plt.legend()
    plt.savefig(f'../Results/liliefors_test_{var_name}.png') # How to save it under the right name!!!!!
    plt.show()
    plt.close()
     
    print(f'\n=== LILLIEFORS TEST: {var_name} ===\n',
          '\nSTAT: ', stat, 
          '\nP-VALUE: ', p_value,
          '\nP <= 0.05: data is not normally distributed' if p_value <= alpha 
                                                          else 'p > 0.05: data is normally distributed')
    
    return p_value 

#%% Outlier treatment based on Mahalanobis distances


def Mahalanobis_test(var1: pd.Series, var2: pd.Series, alpha: float):
    '''
    Calculating the Mahalanobis Distance
  
    Compute the Mahalanobis distance for two variables to identify multivariate outliers.
    An outlier threshold is determined based on the specified significance level (alpha) and used to generate an outlier mask.
    The implementation is adapted from `Stackoverflow <https://stackoverflow.com/questions/46827580/multivariate-outlier-removal-with-mahalanobis-distance>`_.

    Parameters
    ----------
    var_1: pd.Series
        First variable for the Mahalanobis distance calculation.
    var_2: pd.Series
        Second variable for the Mahalanobis distance calculation
    alpha: float
        The significance level to determine the outlier treshold
    Returns
    -------
    mahalanobis_dist: list of str
        The Mahalanobis distances for each observation.
    threshold: float
        The distance threshold for which an observation is considered an outlier
    outliers_mask: list of bool
        Boolean mask indicating which observations are outliers (True) and which are not (False)
    Examples
    --------
    >>> mahalanobis_dist, threshold, outliers_mask = Mahalanobis_test(var1, var2, 0.05)
    mahalanobis_dist
    [2.3, 2.4, 3.1]
    threshold
    2.9
    outliers_mask
    [False, False, True]
    '''
    
    data = pd.concat([var1, var2], axis=1)
    covariance_matrix = np.cov(data, rowvar=False)  
    
    if is_pos_def(covariance_matrix):
        inv_covariance_matrix = np.linalg.inv(covariance_matrix)
        if is_pos_def(inv_covariance_matrix):
            vars_mean = data.mean(axis=0)
            diff = data - vars_mean
            mahalanobis_dist = []
            for i in range(len(diff)):
                mahalanobis_dist.append(np.sqrt(diff.iloc[i].dot(inv_covariance_matrix).dot(diff.iloc[i])))
                
            degree_of_freedom = 2 
            md_square = np.square(mahalanobis_dist)
            
            p_values = 1 - chi2.cdf(md_square, degree_of_freedom)

            outlier_mask = p_values < alpha
            non_outlier_mask = p_values >= alpha
            
            plt.figure(figsize=(6,6))
            plt.scatter(data.iloc[non_outlier_mask, 0], data.iloc[non_outlier_mask, 1], color='blue', label='Non-outliers')
            plt.scatter(data.iloc[outlier_mask, 0], data.iloc[outlier_mask, 1], color='red', label='Outliers')
            plt.xlabel('Variable 1')
            plt.ylabel('Variable 2')
            plt.legend()
            plt.savefig('../Results/outlier.png')
            plt.show()
            plt.close()
     
            
            print('\n=== MAHALANOBIS TEST ===\n',
                  '\nOUTLIER THRESHOLD: ', p_values,
                  '\nMAX MAHALANOBIS DISTANCE:', np.max(mahalanobis_dist),
                  '\nTOTAL OUTLIERS:', np.sum(outlier_mask == True))
            return mahalanobis_dist, p_values, non_outlier_mask
        else:
            print("Error: Inverse of Covariance Matrix is not positive definite!")
    else:
        print("Error: Covariance Matrix is not positive definite!")


#%% Positive definite check

def is_pos_def(A):
    """
    Positive definite check

    Check if a given square matrix is symmetric and positive definite.
    This function is obtained from `Stackoverflow <https://stackoverflow.com/questions/46827580/multivariate-outlier-removal-with-mahalanobis-distance>`_.
    Parameters
    ----------
    A : pd.DataFrame
        A square matrix to check.
        
    Returns
    -------
    bool
        True if the matrix is symmetric and positive definite, False otherwise.

    Example
    -------
    >>> A = pd.DataFrame([[2, -1], [-1, 2]])
    >>> is_pos_def(A)
    True
    >>> B = pd.DataFrame([[1, 2], [2, 1]])
    >>> is_pos_def(B)
    False
    """
    if np.allclose(A, A.T):
        try:
            np.linalg.cholesky(A)
            return True
        except np.linalg.LinAlgError:
            return False
    else:
        return False



    

#%% Statistical analysis
def analysing_data(var1: pd.Series, var2: pd.Series,
                   var1_range: list, var2_range: list,
                   var1_name: str, var2_name: str,
                   alpha: float):   
    '''
    Data Analysis
   
    Calculate the correlation between two SDG indicators based.
    - If both variables are normally distributed, a parametric test (Pearson correlation) is applied.
    - If not, a non-parametric test (Spearman correlation) is used.
    Outliers are removed based on the Mahalanobis distance before performing the correlation analysis.
    
    Parameters
    ----------
    var1: pd.Series
        The first SDG indicator series.
    var2: pd.Series
        The second SDG indicator series.
    var1_range: list 
        Theoretical range of first SDG indicator.
    var2_range: list
        Theoretical range of second SDG indicator.
    var1_name: str 
        Name of first SDG indicator.
    var2_name: str
        Name of second SDG indicator.
    alpha: float
        Significance level used for normality, outlier detection and correlation test.
    Returns
    -------
    norm_p_values: list of float
        P-values from the Lilliefors normality test for both variables.
    correlation: float
        Correlation coefficient calculated using Pearson or Spearman test.
    p_value:
        P-value associated with the correlation test.
    Examples
    --------
    >>> norm_p_values, correlation, p_value = analysing_data(var1, var2, var1_range, var2_range, 'SDG 1', 'SDG 12', 0.05)
    norm_p_values
    [0.12, 0.08]
    correlation
    0.72
    p_value
    0.003
    '''
    # Testing for normality
    norm_p_values = [normality(var1, var1_name, var1_range, alpha),
                normality(var2, var2_name, var2_range, alpha)] 
       
    # Outlier removal
    mahalanobis_dist, threshold, outliers_mask = Mahalanobis_test(var1, var2, alpha) 
      
    var1_no_outliers = var1[outliers_mask]
    var2_no_outliers = var2[outliers_mask]
     
    # Testing for correlation        
    if norm_p_values[0] <= 0.05 or norm_p_values[1] <= 0.05:  # One or both variables are normal distributed, therefore non-parametric test is selected 
    
        correlation, p_value = spearmanr(var1_no_outliers, var2_no_outliers)
        print('\n=== SPEARMAN TEST ===\n',
              '\nCORRELATION:', correlation,
              '\nP-VALUE:', p_value,
              '\nP <= 0.05: Correlation is significant' if p_value <= 0.05 
              else '\nP > 0.05: Correlation is insignificant\n')
        
    else :
        correlation, p_value = pearsonr(var1_no_outliers, var2_no_outliers) # Both variables are normal distributed, therefore parametric test is selected
        print('\n=== PEARSON TEST ===\n',
              '\nPEARSON Rho:', correlation,
              '\nP-value:', p_value,
              '\np <= 0.05: Correlation is significant' if p_value <= 0.05 
              else '\np > 0.05: Correlation is insignificant\n')
    
    plt.scatter(var1, var2, 
                color='royalblue', 
                label='Data points',
                alpha=0.5)

    plt.xlabel("var 1")
    plt.ylabel("var 2")
    plt.xlim(var1_range)
    plt.ylim(var2_range)
    plt.legend()
    plt.savefig('../Results/scatter.png')
    plt.show()
  
    return norm_p_values, correlation, p_value
#%% Conclusion
def conclusion(p_value: float, alpha: float, correlation: float, positive: bool):
    '''
    Conclusion
   
    State the conclusion on the signficance of the correlation based on the p-value and alpha.
    Parameters
    ----------
    p_value: float
        The calculated p-value.
    alpha: float
        the defined significance level.
    Returns
    -------
    conclusion: str
       A statement with a conclusion based on p-value and alpha.
    --------
    >>> conclusion(0.03, 0.05)
    'Since the identified p-value (0.03) is below the significance level (0.05),
    we can conclude that the observed the correlation is statistically significant.'
    '''
    
    if correlation > 0 and positive==True or correlation < 0 and positive==False:
        relation_statement = (f'The correlation coefficient ({correlation}),'
                              ' indicates a synergy between the two SDG indicators.')
        
    elif correlation > 0 and positive==False or correlation < 0 and positive==True:
        relation_statement = (f'The correlation coefficient ({correlation}),'
                              ' indicates a trade-off between the two SDG indicators.')    
    else:
        relation_statement = ('A correlation coefficient of 0 indicates no trade-offs or synergys between'
                              'the two SDG indicators.')
    
    if p_value > alpha:
        conclusion_statement = (f'\nSince the identified p-value ({p_value}) is above the significance level ({alpha}),'
                                ' we can conclude that the observed the correlation is statistically insignificant.')    
    else:
        conclusion_statement = (f'\nSince the identified p-value ({p_value}) is below the significance level ({alpha}),'
                                ' we can conclude that the observed the correlation is statistically significant.') 
    
    
     
    return relation_statement + conclusion_statement
