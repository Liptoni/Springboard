import pandas as pd

df = pd.read_excel('WineKMC.xlsx', sheet_name=0)

print(df.head())
