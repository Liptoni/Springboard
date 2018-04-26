import pandas as pd
import requests

#import h1b data, subset lat/lon
hb_data = pd.read_csv('h1b_clean.csv', index_col='CASE_NUMBER')
lat_long = hb_data[['lat', 'lon']].drop_duplicates().dropna()

#create a list of urls for api calls
url_list = []
for index, row in lat_long.iterrows():
    lat = row['lat']
    lon = row['lon']
    url = "https://geo.fcc.gov/api/census/area?lat=%s&lon=%s&format=json" % (str(lat), str(lon))
    url_list.append(url)

#loop over urls to make api calls, store each in a dataframe and store all dataframes in a list
dfs=[]
for url in url_list:
    r = requests.get(url)
    json_data = r.json()

    input = pd.io.json.json_normalize(json_data['input'])
    results = pd.io.json.json_normalize(json_data['results'])
    together = input.merge(results, left_index=True, right_index=True)
    dfs.append(together)

#concatenate dataframes from api calls, export to csv
all_locs = pd.concat(dfs)
all_locs.reset_index(inplace=True)
all_locs = all_locs[['lat', 'lon', 'block_fips', 'block_pop_2015', 'county_fips', 'county_name', 'state_code', 'state_fips', 'state_name']]
print(all_locs.info())

all_locs.to_csv('fcc_census_info.csv')
