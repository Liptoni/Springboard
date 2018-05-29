import pandas as pd
import numpy as np
pd.options.mode.chained_assignment = None

def get_status(status):
    """
    Takes the Status of a visa application
    Returns either Certified of Not Certifed
    Used to create the column CERTIFIED
    """
    if status == 'CERTIFIED' or status == 'CERTIFIED-WITHDRAWN':
        return 'certified'
    else:
        return 'denied'

def case_status(hb):
    """
    Takes the hb dataframe.
    Drops records from CASE_STATUS, creates CERTIFIED
    Returns the new hb dataframe
    """
    valid_status_values = ['CERTIFIED', 'CERTIFIED-WITHDRAWN', 'DENIED']
    hb = hb[hb.CASE_STATUS.isin(valid_status_values)]
    hb['CERTIFIED'] = hb.CASE_STATUS.apply(get_status)
    return hb

#read in data
hb = pd.read_csv('Z:/Springboard/H1B_Capstone/data/h1b_kaggle.csv')

#rename missing column header. This is from the data dict website
#https://www.foreignlaborcert.doleta.gov/docs/Performance_Data/Disclosure/FY15-FY16/H-1B_FY16_Record_Layout.pdf
hb.rename(columns={'Unnamed: 0': 'CASE_NUMBER'}, inplace = True)
hb.set_index('CASE_NUMBER', inplace=True)

hb = case_status(hb)

#Convert year to an integer
hb.YEAR = hb.YEAR.astype(np.int64)

#Split WORKSITE into City and State
hb['CITY'], hb['STATE'] = hb['WORKSITE'].str.split(', ', 1).str
hb['CITY'] = hb['CITY'].str.strip()
hb['STATE'] = hb['STATE'].str.strip()

#log-transform wage into new column
hb['LOG_WAGE'] = np.log10(hb['PREVAILING_WAGE'])

#drop obviously erroneous records
hb = hb[hb.PREVAILING_WAGE < 1e+09]

hb.to_csv('Z:/Springboard/H1B_Capstone/data/h1b_clean.csv')
