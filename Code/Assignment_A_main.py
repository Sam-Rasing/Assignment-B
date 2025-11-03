# -*- coding: utf-8 -*-
"""
Created on Thu Sep 18 16:03:29 2025

@author: Sam Rasing
"""

#%% Import Modules
import cProfile, pstats 
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import Assignment_A_functions as af

pr=cProfile.Profile() 
pr.enable() 

if __name__ == "__main__":
    
    root_path = 'C:/Users/Sam/OneDrive/Documents/Industrial Ecology/SAPY/Assignment-A/'# Add here your own root path to Assignment-A
    
#%% Importing Data
    
    '''
    Process of importing, Sanity checks and filtering the data
    
    The social indicator is SDG2 > Sustinable agriculture. THis is obtained from
    sdsdsdsd    
    The environmental indicator is SDG15 > Sustinable agriculture. THis is obtained from ....

    '''
    # Step 1: Loading in the SDG data
    var1_name = 'SDG 2.4.1: Sustainable Agriculture'
    var2_name = 'SDG 15.5.1: Red list index'
    
    var1_range = [1, 5]
    var2_range = [0, 1]
    
    data_folder = root_path + 'Data/'
    file_name = ['AG_LND_SUST_PRXCSS.xlsx',
                 'ER_RSK_LST.xlsx']
    
    index = 'GeoAreaName'
    time_period = list(map(str, range(1800, 2026)))
    columns = ['GeoAreaName', 'GeoAreaCode'] + time_period
    
    var1 = af.importing_data(data_folder, file_name[0], index, columns, var1_name)
    var2 = af.importing_data(data_folder, file_name[1], index, columns, var2_name)
     
    # Filtering the data for most recent year for variables
    var1, var2, recent_year, common_countries = af.filter_data(var1, var2)
    
#%%
    '''
    Doing the analysis 
    
    Select the alpha for this study
    
    Outliers are thrown based on mahalanobis distance and a shi quare reflecting the degree of freedoms of 2
    
    The outcome is the p_values for the normality test for both variables, and the correlation test for parametric 
    or non-parametric test depending on the distrubtion of the variables 
    '''
    alpha = 0.05 # select alpha for test: 0.01 -> 99% confidence, 0.05 -> 95% confidence, 0.1 -> 90% confidence 
    
    p_values, correlation_result = af.analysing_data(var1[recent_year], 
                                                     var2[recent_year], 
                                                     alpha,
                                                     var1_range,
                                                     var2_range,
                                                     var1_name,
# %%
                                                     var2_name)

    
    #%% Plotting results
    '''
    Plotting the results of the analysis
    XX
    
    
    dfsd
    
    
    sdfdsfdsfdsf
    
    '''
    # Importing additional population data
   
    usecols = ['Location code', 'ISO3 Alpha-code']
    area_codes = pd.read_excel(data_folder + 'WPP2024_F01_LOCATIONS.xlsx', 
                               skiprows=16, 
                               usecols=usecols)

    usecols = ['ISO3_code', 'Time', 'PopTotal', 'LocTypeName', 'Location']
    pop = pd.read_csv(data_folder + 'WPP2024_TotalPopulationBySex.csv.gz', 
                      usecols=usecols)
    
    pop = pop[(pop['LocTypeName'] == 'Country/Area') & (pop['Time'] == int(recent_year))] 
 
    pop = pop.merge(area_codes, left_on='ISO3_code', right_on='ISO3 Alpha-code', how='left')
    pop = pop[pop['Location code'].isin(common_countries)]
    
    pop.set_index('Location', inplace=True)
    pop.sort_index(inplace=True)
    
    # Bubble plot
    top_10 = pop.nlargest(10, 'PopTotal').index.tolist()
    colors = ['royalblue' if country in top_10 else 'cornflowerblue' for country in pop.index]

    plt.scatter(var1[recent_year], var2[recent_year], s=pop['PopTotal']/1e3, c=colors, alpha=0.5)
   
    for country in top_10:
        plt.annotate(country, (var1.loc[country, recent_year], var2.loc[country, recent_year]), 
                         fontsize=9, ha='center', va='center',)
    
    
    plt.xlim(1, 5)
    plt.ylim(0, 1)
    plt.xlabel(var1_name)
    plt.ylabel(var2_name)   
    plt.savefig('../Results/bubble_plot.png')   
    plt.show()
    plt.close()

    
#%%
    '''
    Plotting the results of the analysis
    XX
    XXX
    
    dfsd
    
    
    sdfdsfdsfdsf
    
    '''    
    map_data = pd.merge(var1, var2, on='GeoAreaCode', how='outer')  
    map_data['UN_A3'] = map_data['GeoAreaCode'].astype(str).str.zfill(3)
    
    url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
    
    world_map = gpd.read_file(url)
    world_map = world_map.set_index('UN_A3')

    world_map = pd.merge(world_map, map_data, on='UN_A3', how='left')
    world_map = world_map[world_map['CONTINENT'] != 'Antarctica']
       
    world_map.plot(column=world_map.columns[-2], 
                   legend=True, 
                   cmap='OrRd_r',
                   vmin=1,                # Min value for SDG indicator
                   vmax=5,                # Max value for SDG indicator
                   missing_kwds={'color': 'lightgrey',  "label": "Missing values"}, 
                   legend_kwds={"label": var1_name, "orientation": "horizontal"})
    
    plt.xticks([])
    plt.yticks([])
    plt.xlim([-180, 180]) 
    plt.ylim([-60, 90])
    plt.savefig('../Results/map_1.png')
    plt.show()
    plt.close()
       
    world_map.plot(column=world_map.columns[-1], 
                   legend=True, 
                   cmap='OrRd',
                   vmin=0,                # Min value for SDG indicator
                   vmax=1,                # Max value for SDG indicator
                   missing_kwds={'color': 'lightgrey',  "label": "Missing values"}, 
                   legend_kwds={"label": var2_name, "orientation": "horizontal"})
    
    plt.tight_layout()
    
    plt.xticks([])
    plt.yticks([])
    plt.xlim([-180, 180]) 
    plt.ylim([-60, 90])
    plt.savefig('../Results/map_2.png')
    plt.show()
    plt.close()

    
    
#%%
    average_1 = map_data.iloc[:, 1].mean()
    average_2 = map_data.iloc[:, 2].mean()
    
    world_map['Trade_offs'] = None
    
    for i in range(len(world_map.index)):
        if pd.isna(world_map.iloc[i, -3]) or pd.isna(world_map.iloc[i, -2]):
            continue
        if world_map.iloc[i, -3] > average_1 and world_map.iloc[i, -2] < average_2:
            world_map.loc[i,'Trade_offs'] = 2
        elif world_map.iloc[i, -3] > average_1 or world_map.iloc[i, -2] < average_2:
            world_map.loc[i,'Trade_offs'] = 1  
        else:
            world_map.loc[i,'Trade_offs'] = 0
                
    cmap = ListedColormap(['crimson', 'gold', 'green'])

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
    
    #%% Exporting to txt file 
    '''
    Exporting the results to a txt file
    
    '''
    f = open('../Results/my_results.txt', 'w')
    
    print('SDG indicator #1:', var1_name,'\n'
          'SDG indicator #2:', var2_name,'\n',
          #'\nCorrelation Coefficient:', correlation_result, 'Obtained by Spearmanr test' if p_values <= 0.05 or p_values <= 0.05 else 'Obtained by Pearsonr test',
          #'\nP-value 1:', p_values[0] if p_values[0] <= 0.05 or p_values[1] <= 0.05 else 'Obtained by Pearsonr test',
          #'\nP-value 2:', p_values[1] if p_values[0] <= 0.05 or p_values[1] <= 0.05 else 'Obtained by Pearsonr test',
          file=f)
    
    f.close()
    
    #%%
    
    pr.disable() 
    
    ps=pstats.Stats(pr).strip_dirs().sort_stats('cumulative') 
    
    ps.print_stats(10) # 
    ps.print_callers(10) # 
    




