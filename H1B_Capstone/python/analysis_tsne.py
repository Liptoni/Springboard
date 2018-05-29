# -*- coding: utf-8 -*-
"""
Created on Fri May 25 21:48:56 2018

@author: Ian
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from sklearn.manifold import TSNE
pd.options.mode.chained_assignment = None

start_time = datetime.now()
print(start_time)
    
#import cleaned H1B dataset
#full dataset can be found: https://www.kaggle.com/nsharan/h-1b-visa/data
#A description of cleanup can be found: https://github.com/Liptoni/Springboard/blob/master/H1B_Capstone/H1B_Data_Wrangling.docx
hb_data = pd.read_csv('Z:/Springboard/H1B_Capstone/data/h1b_census_full.csv', index_col='CASE_NUMBER',
                        dtype={'block_fips':np.object, 'county_fips':np.object, 'state_fips':np.object})#, nrows=1000

#select random 10,000 and drop NAs
hb_data = hb_data.sample(n=10000, random_state = 24)
hb_data.dropna(inplace=True)

print('data cleaned', datetime.now()-start_time)

#get labels and convert to ints for plotting
labels = hb_data[['CERTIFIED']]
labels.CERTIFIED = pd.Categorical(labels.CERTIFIED)
labels['code'] = labels.CERTIFIED.cat.codes
labels = labels['code']

#get dummies for features
hb_data = pd.get_dummies(hb_data, drop_first=True)

#fit the t-SNE model
model = TSNE(learning_rate = 100)
transformed = model.fit_transform(hb_data)


#plot data
unique = list(set(labels))
colors = ['blue', 'red']
legend_labels = ['Certified', 'Denied']
fig = plt.figure(figsize=(10, 8))

xs = transformed[:, 0]
ys = transformed[:, 1]

#create plot
for i, u in enumerate(unique):
    xi = [xs[j] for j in range(len(xs)) if labels.iloc[j] == u]
    yi = [ys[j] for j in range(len(ys)) if labels.iloc[j] == u]    
    plt.scatter(xi, yi, c=colors[i], label=legend_labels[i])

plt.legend()
plt.title('T-SNE Plot Certified and Denied (n=10,000)')
plt.savefig('TSNE_10000_title.png')
plt.show()



