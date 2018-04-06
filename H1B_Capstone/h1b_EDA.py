from matplotlib import pyplot as plt
import pandas as pd
import seaborn as sns
import numpy as np
pd.options.mode.chained_assignment = None

hb_data = pd.read_csv('h1b_clean.csv', index_col='CASE_NUMBER')

#print(hb_data.info())


clean_wage = hb_data[hb_data.PREVAILING_WAGE != 0]['PREVAILING_WAGE']
clean_wage.dropna(inplace=True)
clean_log = hb_data[hb_data.PREVAILING_WAGE != 0]['LOG_WAGE']
clean_log.dropna(inplace=True)
print(clean_log.describe())


x = np.percentile(clean_wage, [25, 50, 75])
print(x[0], x[1], x[2])
#print(hb_data['PREVAILING_WAGE'].describe())

# x_cert = np.sort(hb_data[hb_data.CERTIFIED =='certified']['LOG_WAGE'])
# y_cert = np.arange(1, (len(x_cert)+1)) / len(x_cert)
#
# x_den = np.sort(hb_data[hb_data.CERTIFIED =='denied']['LOG_WAGE'])
# y_den = np.arange(1, (len(x_den)+1)) / len(x_den)
#
#
# sns.set()
# _ = plt.plot(x_cert,y_cert,marker='.', linestyle='none')
# _ = plt.plot(x_den,y_den,marker='.', linestyle='none')
# plt.margins(0.02)
# plt.legend(('certified', 'denied'), loc="lower right")
# plt.show()
#

# _ = sns.swarmplot(x='CASE_STATUS', y='PREVAILING_WAGE', data=hb_data)
# plt.show()
