

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

phoneData = pd.read_csv("train.csv"); #print(phoneData)
feat = list(phoneData.columns)
target, feat  = feat.pop(), feat

features = pd.DataFrame(data=phoneData, columns=feat)
features = features.drop(['talk_time', 'sc_w', 'n_cores', 'fc','m_dep', 'clock_speed',
 'pc', 'blue', 'four_g', 'sc_h', 'three_g', 'dual_sim', 'touch_screen', 'wifi', 'int_memory'], axis=1)
prices = phoneData['price_range']


# phoneStats = features.mean().values
# print(phoneStats)

regr = LinearRegression().fit(features, prices)


fittedVals = regr.predict(features)
sns.jointplot(prices.values.reshape(2000), fittedVals.reshape(2000))
plt.show()


testData = pd.read_csv("test.csv")
testData.set_index('id', inplace = True)
testData = testData.drop(['talk_time', 'sc_w', 'n_cores', 'fc','m_dep', 'clock_speed',
 'pc', 'blue', 'four_g', 'sc_h', 'three_g', 'dual_sim', 'touch_screen', 'wifi', 'int_memory'], axis=1)

testVals = np.round(regr.predict(testData))
testData['Price Prediction'] = testVals
testData.loc[testData['Price Prediction'] > 3, 'Price Prediction'] = 3;
testData.loc[testData['Price Prediction'] < 0, 'Price Prediction'] = 0;
