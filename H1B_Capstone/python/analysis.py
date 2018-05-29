import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.kernel_approximation import RBFSampler, Nystroem
pd.options.mode.chained_assignment = None


start_time = datetime.now()
print(start_time)


def get_chunks(data_len, chunksize):
    chunkstarter = 0
    while chunkstarter < data_len:
        print("chunk",chunkstarter, datetime.now()-start_time)
        chunkender = chunkstarter + chunksize
        x_chunk, y_chunk = x_train.iloc[chunkstarter:chunkender], y_train.iloc[chunkstarter:chunkender]
        #x_chunk, y_chunk = x_train[chunkstarter:chunkender:1], y_train[chunkstarter:chunkender:1]

        
        yield x_chunk, y_chunk
        chunkstarter += chunksize
    
#import cleaned H1B dataset
#full dataset can be found: https://www.kaggle.com/nsharan/h-1b-visa/data
#A description of cleanup can be found: https://github.com/Liptoni/Springboard/blob/master/H1B_Capstone/H1B_Data_Wrangling.docx
hb_data = pd.read_csv('Z:/Springboard/H1B_Capstone/data/h1b_census_full.csv', index_col='CASE_NUMBER',
                        dtype={'block_fips':np.object, 'county_fips':np.object, 'state_fips':np.object})

#drop NAs
hb_data = hb_data[hb_data.YEAR == 2016]
hb_data.dropna(inplace=True)

print('data cleaned', datetime.now()-start_time)

#split data into labels and features
labels = hb_data['CERTIFIED']
features = hb_data[['FULL_TIME_POSITION', 'PREVAILING_WAGE','SOC_NAME','lon', 'lat', 'county_fips', 'county_pop', 'state_code']]
features= pd.get_dummies(features, drop_first=True)

print('dummies created', datetime.now()-start_time)

#get training and testing data
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.25, random_state=24)
data_len = len(x_train)

print('data split', datetime.now()-start_time)

#split data into batches to meet memory requirements
batcher = get_chunks(data_len, 10000)

#define SGD
sgd = SGDClassifier(alpha=1, loss='hinge', max_iter=5, penalty='l2', tol=None)

scaler = StandardScaler()

#Scale and partial fit classifier for each chunk
for x_chunk, y_chunk in batcher:
    x_scaled = scaler.fit_transform(x_chunk)
    sgd.partial_fit(x_scaled, y_chunk, classes=np.unique(labels))


#Predict on test set, print results
print('predicting', datetime.now()-start_time)

test_scaled = scaler.fit_transform(x_test)
y_pred = sgd.predict(test_scaled)

print(sgd.score(test_scaled, y_test))
print(classification_report(y_test, y_pred))

print("Done!")
print(datetime.now()-start_time)




