import pandas as pd
import json
from pandas.io.json import json_normalize

def most_projects(df, group_column):
    """
    This takes a dataframe and a column to group by.
    It returns a list of the tuples.
    The tuples contain top 10 most common value in that column and how many times they occur.
    """

    df_group = df.groupby(group_column)
    df_count = [(k, v[group_column].count()) for k, v in df_group]
    df_count.sort(key = lambda x: x[1], reverse=True)
    return df_count[:10]

def get_cleaned_proj_names(projects_data):
    """
    This takes the raw imported json data. It normalizes the mjtheme_namecode
    field, and then fills in blank name values based on the unique combination
    of code and name where name is not blank.
    This returns a pandas dataframe with all name-column blanks filled in.
    """
    proj_themes = json_normalize(projects_data, 'mjtheme_namecode')
    proj_themes = proj_themes.set_index('code')
    unique_themes = proj_themes[proj_themes.name != ''].drop_duplicates()
    proj_themes = unique_themes.join(proj_themes, how = 'inner', rsuffix='_x')
    proj_themes = proj_themes['name'].reset_index()
    return proj_themes

#set the filename and import the data
filename = 'data/world_bank_projects.json'
with open(filename) as json_file:
    projects_data = json.load(json_file)

#create the two dataframes to go through most_projects()
world_bank_projects = pd.DataFrame(projects_data)
proj_themes = get_cleaned_proj_names(projects_data)

#get the top 10 countries and major project themes
most_countries = most_projects(world_bank_projects, 'countryname')
most_proj_theme = most_projects(proj_themes, 'name')

#print results
print(most_countries)
print(most_proj_theme)
