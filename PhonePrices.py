

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
print(feat)

#print(pd.isnull(phoneData).any())

# print(phoneData['clock_speed'].value_counts())
# print(phoneData['clock_speed'].mean())

### VISUALIZING HISTOGRAM OF CLOCK SPEED
"""plt.figure(figsize=(10,6))
plt.hist(phoneData['clock_speed'], bins = 24, ec='white', color='forestgreen')
plt.xlabel('Average Clock Speed')
plt.ylabel('Nbr of iters')
#sns.distplot(phoneData['clock_speed'], bins=7)
plt.show()
"""

### CORRELATION
print(phoneData.describe())
mask = np.zeros_like(phoneData.corr())
triangle_indices = np.triu_indices_from(mask)
mask[triangle_indices] = True


"""
plt.figure(figsize=(13,8))
sns.heatmap(phoneData.corr(), mask=mask, annot=True)
#plt.show()"""


features = pd.DataFrame(data=phoneData, columns=feat)
prices = phoneData['price_range']

xTrain, xTest, yTrain, yTest = train_test_split(features, prices,
 test_size=.2, random_state=653)

regr = LinearRegression().fit(xTrain,yTrain)
print("Intercept :", regr.intercept_)
panda = pd.DataFrame(data=regr.coef_, index=xTrain.columns,columns=['coef'])
#print(panda)
xAndConst = sm.add_constant(xTrain)

print('R² Training Data: ', regr.score(xTrain,yTrain))
print('R² Test Data: ', regr.score(xTest,yTest))

print("********   P values   *******")

model = sm.OLS(yTrain, xAndConst)
results = model.fit()
pandaPvalues = pd.DataFrame({'coef':results.params, 'p-values':round(results.pvalues,3)})
print(pandaPvalues.sort_values(by=['p-values']))



#ORIGNAL MODEL BIC ,RSquared & MSE

print('\nBIC :', results.bic)
print('R-Squared :', results.rsquared)
print("MSE for Normal Model :", round(results.mse_resid, 5))

#REMOVING  FEATURES TO SIMPLIFY MODEL

print("\n********   NEW MODEL   *******")

xAndConst = xAndConst.drop(['talk_time', 'sc_w', 'n_cores', 'fc','m_dep', 'clock_speed',
 'pc', 'blue', 'four_g', 'sc_h', 'three_g', 'dual_sim', 'touch_screen', 'wifi', 'int_memory'], axis=1)
model = sm.OLS(yTrain, xAndConst)
results = model.fit()
newCoef= pd.DataFrame({'coef':results.params, 'p-values':round(results.pvalues,3)})
print(newCoef.sort_values(by=['p-values']))
print('\nBIC :', results.bic)
print('R-Squared :', results.rsquared)
print("MSE for Reduced Model :", round(results.mse_resid, 5))


### Checking possible coef changes
print(pd.concat([pandaPvalues, newCoef], axis=1))

"""
corr = yTrain.corr(results.fittedvalues)
print('Correlation :',corr)
roundCorr = yTrain.corr(round(results.fittedvalues))
print('Corr for rounded values :',roundCorr)
"""

plt.plot(yTrain,yTrain)
plt.scatter(x=yTrain, y=results.fittedvalues, c='darkred', alpha = .6)
#plt.show()

plt.scatter(x=results.fittedvalues, y=results.resid, c='darkred', alpha = .6)
#plt.show()

print('Fitted Values: \n',round(results.fittedvalues))


print("\n********    V I F    *******")
print(variance_inflation_factor(exog=xAndConst.values, exog_idx=0))
### Variance Inflation Factor
vif = []
for i in range(xAndConst.shape[1]) :
    vif.append(variance_inflation_factor(exog=xAndConst.values, exog_idx=i))

vifData = pd.DataFrame({'coef_name':xAndConst.columns, 'vif':np.around(vif, 2)})
print(vifData)
