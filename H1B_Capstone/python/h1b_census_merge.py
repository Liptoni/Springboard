import pandas as pd
pd.options.mode.chained_assignment = None

"""
This script merges the cleaned H1B data with the cleaned census and county population data
Outputs one csv to use for analysis
"""

#import cleaned H1B dataset
#full dataset can be found: https://www.kaggle.com/nsharan/h-1b-visa/data
#A description of cleanup can be found: https://github.com/Liptoni/Springboard/blob/master/H1B_Capstone/H1B_Data_Wrangling.docx
hb_data = pd.read_csv('Z:/Springboard/H1B_Capstone/data/h1b_clean.csv')

#need to create these in order to join with census data
hb_data['lat_round'] = hb_data['lat'].apply(lambda x: round(x, 4))
hb_data['lon_round'] = hb_data['lon'].apply(lambda x: round(x, 4))


#import census data
census_cols = ['ID1', 'Census_ID', 'lat', 'lon', 'block_fips', 'block_pop_2015', 'county_fips', 'county_name', 'state_code', 'state_fips', 'state_name', 'county_pop', 'GEONAME']
census_data = pd.read_csv('Z:/Springboard/H1B_Capstone/data/county_pop.csv', header=0, names=census_cols)
census_data['lat_round'] = census_data['lat'].apply(lambda x: round(x, 4))
census_data['lon_round'] = census_data['lon'].apply(lambda x: round(x, 4))
census_data = census_data[['Census_ID', 'lat_round', 'lon_round', 'block_fips', 'block_pop_2015', 'county_fips', 'county_name', 'county_pop', 'state_code', 'state_fips', 'state_name']]


#merge hb_data and census data
hb_census = hb_data.merge(census_data, how='left', left_on=['lat_round', 'lon_round'], right_on=['lat_round', 'lon_round'])


hb_census.to_csv('Z:/Springboard/H1B_Capstone/data/h1b_census_full.csv', index=False)
