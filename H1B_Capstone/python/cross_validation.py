import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import classification_report
#from sklearn.kernel_approximation import RBFSampler, Nystroem

from datetime import datetime
pd.options.mode.chained_assignment = None


start_time = datetime.now()
print(start_time)
#import cleaned H1B dataset
#full dataset can be found: https://www.kaggle.com/nsharan/h-1b-visa/data
#A description of cleanup can be found: https://github.com/Liptoni/Springboard/blob/master/H1B_Capstone/H1B_Data_Wrangling.docx
hb_data = pd.read_csv('Z:/Springboard/H1B_Capstone/data/h1b_census_full.csv', index_col='CASE_NUMBER',
                        dtype={'block_fips':np.object, 'county_fips':np.object, 'state_fips':np.object})


#select a random 10,000 records and drop NAs
hb_data = hb_data.sample(n=10000, random_state = 24)
hb_data.dropna(inplace=True)

#split data into labels and features
labels = hb_data['CERTIFIED']
features = hb_data[['FULL_TIME_POSITION', 'PREVAILING_WAGE','SOC_NAME','lon', 'lat', 'county_fips', 'county_pop', 'state_code']]# , 'block_fips', 'block_pop_2015'
features= pd.get_dummies(features, drop_first=True)

# =============================================================================
# #pick Kernel
# kernel_ = Nystroem(random_state =24)
# =============================================================================


#setup pipeline
steps = [('scaler', StandardScaler()), ('sgd',SGDClassifier(tol=None))] #, ('kernel',kernel_)
pipeline = Pipeline(steps)

#identify hyperparameters to test in GridSearchCV
loss_options = ['hinge', 'log', 'modified_huber', 'squared_hinge', 'perceptron']
alpha_options =  [0.01, 0.1, 1, 10]#[0.00001, 0.0001, 0.001, 0.01, 0.1, 1]
max_iter_options = [5, 10, 100]#, 1000]
penalty_options = ['l2', 'l1']

parameters = dict(sgd__loss = loss_options,
                sgd__alpha = alpha_options, 
                sgd__max_iter = max_iter_options,
                sgd__penalty = penalty_options)


#get training and testing data
x_train, x_test, y_train, y_test = train_test_split(features, labels, test_size = 0.25, random_state=24)

#run cross validation
cv=GridSearchCV(pipeline, param_grid=parameters, cv=5, verbose=10)
cv.fit(x_train, y_train)
y_pred = cv.predict(x_test)

#print results of best-fit model
print(cv.best_params_)
print(cv.score(x_test, y_test))
print(classification_report(y_test, y_pred))


print(datetime.now()-start_time)

