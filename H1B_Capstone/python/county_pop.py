import pandas as pd
import requests
import numpy as np

#import FCC census information
#This was generated with the fcc_census.py script
census_cols = ['ID', 'lat', 'lon', 'block_fips', 'block_pop_2015', 'county_fips', 'county_name', 'state_code', 'state_fips', 'state_name']
census_data = pd.read_csv('fcc_census_info.csv', header=0, names=census_cols, dtype={5:np.object, 8:np.object})

#subset only the county_fips and state fips, calculate county only code
census_county = census_data[['county_fips', 'state_fips']].drop_duplicates()
census_county['county_short'] = census_county['county_fips'].apply(lambda x: x[2:])

#will add this back when I need to re-run
api_key = ''

#create a list of api calls
county_urls = []
for index, row in census_county.iterrows():
    county= row['county_short']
    state = row['state_fips']
    url = 'https://api.census.gov/data/2016/pep/population?get=POP,GEONAME&for=county:%s&in=state:%s&key=%s' % (county, state, api_key)
    county_urls.append(url)

#call api using created list of urls, store each dataframe in a list
county_frames = []
for url in county_urls:
    r = requests.get(url)
    if r.status_code == 204:
        pass
    else:
        json_data = r.json()
        headers = json_data[0]
        df = pd.DataFrame(json_data[1:], columns=headers)
        county_frames.append(df)

#concatenate all dataframes into one
all_counties = pd.concat(county_frames)

#join dataframes from api calls with census data
pop_county_join = census_county.merge(all_counties, how='inner', left_on=['state_fips', 'county_short'], right_on=['state', 'county'])
pop_county_join = pop_county_join[['county_fips', 'POP', 'GEONAME']]

all_joined = census_data.merge(pop_county_join, how='left', left_on='county_fips', right_on='county_fips')

#export final csv
all_joined.to_csv('county_pop.csv')
