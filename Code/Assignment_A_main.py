# -*- coding: utf-8 -*-
"""
**Assignment_A_main.py**
Created on Thu Sep 18 16:03:29 2025

@author: Sam Rasing
"""

# Importing the required modules
import cProfile, pstats 
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import Assignment_A_functions as af
import numpy as np

# Code profiler for optimization
pr=cProfile.Profile() 
pr.enable() 

if __name__ == "__main__":
        
    # Add your path to the root folder of this project called Assignment-A/
    root_path = 'C:/Users/Sam/OneDrive/Documents/Industrial Ecology/SAPY/Assignment-A/'
    
#%% Importing and processing of required data.

    '''
    Importing and processing of the time series data for two SDG indicators 
    across multiple countries. The Data is obtained from the UN SDG Data Portal: 
        - https://unstats.un.org/sdgs/dataportal/database. 
    
    For this analysis the following two SDG indicators are selected:
        - SDG 2.4.1: Sustainable Agriculture
        - SDG 15.5.1: Red list index
    '''
    
    # Define the name of both SDGs:
    var1_name = 'SDG 2.4.1, Sustainable Agriculture' 
    var2_name = 'SDG 15.5.1, Red list index' 
    
    # Define the theoretical ranges for both SDGs:
    var1_range = [1, 5] 
    var2_range = [0, 1] 
    
    # Define the relationship
    positive = False  # The SDGs have opposite directions in terms of better outcomes.
    
    # Define the file names for both SDG datasets 
    data_folder = root_path + 'Data/'
    file_name = ['AG_LND_SUST_PRXCSS.xlsx',
                 'ER_RSK_LST.xlsx']
    index = 'GeoAreaName'
    time_period = list(map(str, range(1800, 2026)))
    columns = ['GeoAreaName', 'GeoAreaCode'] + time_period
    
    var1 = af.importing_data(data_folder, file_name[0], index, columns, var1_name)
    var2 = af.importing_data(data_folder, file_name[1], index, columns, var2_name)
     
    # Filtering the two SDG datasets for most recent year and common countries.
    var1, var2, recent_year, common_countries = af.filter_data(var1, var2)

#%% Statistial analysis

    '''
    Analyse the correlation between the two SDG variables. 
    For this the following steps needs to be done:
        - For each variable select the SDG country data based on recent_year.
        - Define your alpha for the desired confidence level of the analysis
    
    The statistical analysis involves:
        - Normality check for each variable using the Lilliefors test.
        - Visual check for homoscedasticity through a scatter plot.
        - Outlier detection and removal using the Mahalanobis distance.
        - Correlation test based on outcomes of normality test
    '''  
    
    alpha = 0.05 # 0.01 -> 99% confidence, 0.05 -> 95% confidence, 0.1 -> 90% confidence 
    
    # Select variable column based on recent_year
    norm_p_values, p_value, correlation_result = af.analysing_data(var1[recent_year], var2[recent_year],
                                                                    var1_range, var2_range,
                                                                    var1_name,  var2_name,
                                                                    alpha)
    
#%% Plotting results
    
    '''
    Visualisation of the results of this analysis
        - Bubble plot: countries sized by population, top 10 highlighted.
            Population data: https://population.un.org/wpp/downloads?folder=Standard%20Projections&group=CSV%20format.
            
        -  World map: country colours based on SDG performance compared to global average.
            Spatial data ('naturalearth_lowres'): https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip
    '''
    
    # Importing and processing population data to make it compatible to SDG data.
    usecols = ['Location code', 'ISO3 Alpha-code']
    area_codes = pd.read_excel(data_folder + 'WPP2024_F01_LOCATIONS.xlsx', 
                               skiprows=16, 
                               usecols=usecols)
    
    usecols = ['ISO3_code', 'Time', 'PopTotal', 'LocTypeName', 'Location']
    pop = pd.read_csv(data_folder + 'WPP2024_TotalPopulationBySex.csv.gz', 
                      usecols=usecols,
                      low_memory=False)

    pop = pop[(pop['LocTypeName'] == 'Country/Area') & (pop['Time'] == int(recent_year))]  
    pop = pop.merge(area_codes, left_on='ISO3_code', right_on='ISO3 Alpha-code', how='left')
    pop = pop[pop['Location code'].isin(common_countries)]
    pop.set_index('Location', inplace=True)
    pop.sort_index(inplace=True)
    
    # Identifying the 10 populous countries.
    top_10 = pop.nlargest(10, 'PopTotal').index.tolist()
    colors = ['royalblue' if country in top_10 else 'cornflowerblue' for country in pop.index]
    
    
    # Bubble plot:
    x = var1.loc[var1['GeoAreaCode'].isin(pop['Location code']), recent_year]
    y = var2.loc[var2['GeoAreaCode'].isin(pop['Location code']), recent_year] 
    size = pop['PopTotal']/1e3
    
    plt.scatter(x, y, size, c=colors, alpha=0.5)
   
    for country in top_10: # label top 10 populous countries
        plt.annotate(country, (var1.loc[country, recent_year], var2.loc[country, recent_year]), 
                         fontsize=9, ha='center', va='center',)
    
    plt.xlim(var1_range)
    plt.ylim(var2_range)
    plt.xlabel(var1_name)
    plt.ylabel(var2_name)   
    plt.savefig('../Results/bubble.png')   
    plt.show()
    plt.close()

    # Processing and importing data to make world map.
    map_data = pd.merge(var1, var2, on='GeoAreaCode', how='outer')  
    map_data['UN_A3'] = map_data['GeoAreaCode'].astype(str).str.zfill(3)
    
    # Import country polygons
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    
    world_map = gpd.read_file(url)
    world_map = world_map.set_index('UN_A3')
    world_map = pd.merge(world_map, map_data, on='UN_A3', how='left')
    world_map = world_map[world_map['CONTINENT'] != 'Antarctica'] # Exclude antartica
     
    # Plot countries on a world map based on their first SDG score.
    world_map.plot(column=world_map.columns[-2], 
                   legend=True, 
                   cmap='OrRd_r', # Switch to 'OrRd' if higher values represent worse outcomes
                   vmin=var1_range[0], # Min value for SDG indicator.
                   vmax=var1_range[1], # Max value for SDG indicator.
                   missing_kwds={'color': 'lightgrey',  "label": "Missing values"}, 
                   legend_kwds={"label": var1_name, "orientation": "horizontal"})
    
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.xlim([-180, 180]) 
    plt.ylim([-60, 90])
    plt.savefig(f'../Results/map_{var1_name}.png')
    plt.show()
    plt.close()
       
    # Plot countries on a world map based on their second SDG score.
    world_map.plot(column=world_map.columns[-1], 
                   legend=True, 
                   cmap='OrRd', # Switch to 'OrRd_r' if lower values represent worse outcomes
                   vmin=var2_range[0], # Min value for SDG indicator.
                   vmax=var2_range[1], # Max value for SDG indicator.
                   missing_kwds={'color': 'lightgrey',  "label": "Missing values"}, 
                   legend_kwds={"label": var2_name, "orientation": "horizontal"})
    
    plt.tight_layout()
    plt.xticks([])
    plt.yticks([])
    plt.xlim([-180, 180]) 
    plt.ylim([-60, 90])
    plt.savefig(f'../Results/map_{var2_name}.png')
    plt.show()
    plt.close()

    
    # Score countries based on global average on each SDG indicator.
    average1 = map_data.iloc[:, 1].mean()
    average2 = map_data.iloc[:, 2].mean() 
    
    # Adjust < > depending on whether higher or lower values indicate worse outcomes!!!
    world_map['Trade_offs'] = None   
    for i in range(len(world_map.index)):
        if pd.isna(world_map.iloc[i, -3]) or pd.isna(world_map.iloc[i, -2]):
            continue
        if world_map.iloc[i, -3] > average1 and world_map.iloc[i, -2] < average2:
            world_map.loc[i,'Trade_offs'] = 2 # Both targets above average
        elif world_map.iloc[i, -3] > average1 or world_map.iloc[i, -2] < average2:
            world_map.loc[i,'Trade_offs'] = 1 # One target above average
        else:
            world_map.loc[i,'Trade_offs'] = 0 # Both targets below average
                
    cmap = ListedColormap(['crimson', 'gold', 'green'])
    
    # Plot countries on a world map to visualize SDG trade-offs. 
    fig, ax = plt.subplots(1, 1, figsize=(15, 10))
    
    world_map.plot(column='Trade_offs',
                   ax=ax, 
                   cmap=cmap, 
                   legend=False, 
                   legend_kwds={'loc': 'lower left', 'title': ''}, 
                   missing_kwds={'color': 'lightgrey'})
    
    legend_elements = [
    Patch(facecolor='crimson', label='Both targets below average'),
    Patch(facecolor='gold', label='One target above average'),
    Patch(facecolor='green', label='Both SDGs above average'),
    Patch(facecolor='lightgrey', label='No data')
    ]
    
    ax.set_xlim([-180, 180]) 
    ax.set_ylim([-60, 90])
    
    ax.legend(handles=legend_elements,
          loc='lower left',
          fontsize=12,
          framealpha=1)  
    
    ax.set_xticks([]) 
    ax.set_yticks([]) 
    
    plt.savefig('../Results/map.png')
    plt.show()
    plt.close()
    
    #%% Exporting to txt file 
    '''
    Exporting all the results to a txt file 
    
    '''
    f = open('../Results/sdg_correlation.txt', 'w')
    
    print('=== CONCLUSION ===\n\n',
          'This analysis used inferential statistics to answer the research question whether there is a synergy or correlation between:',
          f'\n - {var1_name}',
          f'\n - {var2_name}',
          '\n\nThe results obtained from the', 'Spearnmanr test' if (np.array(norm_p_values) > alpha).any() else 'Pearsonr test', 'are the following:',
          f'\n - Correlation coefficient: {correlation_result}',
          f'\n - P-value: {p_value}',
          '\n\n', af.conclusion(p_value, alpha, correlation_result, positive), file=f)
    
    f.close()

    
#%% Disable code profiler
    
    pr.disable() 
    
    ps=pstats.Stats(pr).strip_dirs().sort_stats('cumulative') 
    
    ps.print_stats(10) # 
    ps.print_callers(10) # 
    




