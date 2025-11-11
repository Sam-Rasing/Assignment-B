### Title: Analysing the correlation between two SDGs

*S.E. Rasing, Sustainability Analysis in Python, MSc Industrial Ecology, Leiden*

This repository contains my submission for Assignment A for the course Sustainability Analysis in Python. 
The code in this assignment can be used to analyze the relationship between a social and an environmental Sustainable Development Goal (SDG) of the United Nations (UN). 
For this specific project, I have selected the following two SDGs:
- SDG 2.4.1: Sustainable Agriculture Index
- SDG 15.5.1: Red list Index

Within this repository you can find all necessary folders to run the analysis. 

1) Create a new environment based on the fil named environment.yaml, placed in the Environment folder.
2) If you want to analyse different SDGs, download your data from here: https://unstats.un.org/sdgs/dataportal/database, and place it in the Data folder.
3) Run the Assignment_A_main.py placed in the Code folder, and follow the instruction given within the document. 
4) Open the results in the Results folder.

**Data**

The data for this project is taken from several sources, which are all saved in the Data folder. 

Data 1: AG_LND_SUST_PRXCSS.xlsx
- SDG data is sourced from the UN Stats Data Portal. This file includes country-level data series for SDG 2.4.1 across all available years.
- https://unstats.un.org/sdgs/dataportal/database.

Data 2: ER_RSK_LST.xlsx
- SDG data is sourced from the UN Stats Data Portal. This file includes country-level data series for SDG 2.4.1 across all available years.
- https://unstats.un.org/sdgs/dataportal/database.

Data 3: WPP2024_F01_LOCATIONS.xlsx
- Country codes are obtained from the UN World Population Prospects Documentation.
- https://population.un.org/wpp/downloads?folder=Documentation&group=Documentation.

Data 4: WPP2024_TotalPopulationBySex.csv.gz
- Total Population data for all countries is obtained from the UN World Population Prospects.
- https://population.un.org/wpp/downloads?folder=Standard%20Projections&group=CSV%20format. 

**Results**

The output of running the code is saved in the results folder, which includes the following: 
Result 1: bubble.png
Result 2: liliefors_test_SDG 2.4.1, Sustainable Agriculture Index.png
Result 3: liliefors_test_SDG 15.5.1, Red List Index.png
Result 4: map.png
Result 5: map_SDG 2.4.1, Sustainable Agriculture Index.png
Result 6: map_SDG 15.5.1, Red List Index.png
Result 7: outlier.png
Result 8: scatter.png
Result 9: sdg_correlation.png

**AI disclaimer**
I acknowledge the use of Large language models (LLMs) during the development of this analysis. 
Specifically, AI was used to:
- Explain and troubleshoot errors
- Enhance data visualizations and plots
- Suggest optimization options
- Brainstorming
- Code improvements

Within both Python files, comments have been added to indicate which parts of the code were generated or assisted by AI.
 


