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
prices = phoneData['price_range']


def bestRandomState(i) :

    xTrain, xTest, yTrain, yTest = train_test_split(features, prices,
     test_size=.2, random_state=i)

    """print("\nORIGINAL MODEL")"""

    regr = LinearRegression().fit(xTrain,yTrain)
    xAndConst = sm.add_constant(xTrain)
    """print('R² Training Data: ', regr.score(xTrain,yTrain))
    print('R² Test Data: ', regr.score(xTest,yTest))"""

    #print("********   P values   *******")

    model = sm.OLS(yTrain, xAndConst)
    results = model.fit()
    pandaPvalues = pd.DataFrame({'coef':results.params, 'p-values':round(results.pvalues,3)})
    #print(pandaPvalues)


    #ORIGNAL MODEL BIC ,RSquared & MSE

    """print('\nBIC :', results.bic)
    print('R-Squared :', results.rsquared)
    print("MSE for Normal Model :", round(results.mse_resid, 5))"""

    #REMOVING  FEATURES TO SIMPLIFY MODEL

    """print("\nNEW MODEL")"""

    xAndConst = xAndConst.drop(['talk_time', 'sc_w', 'n_cores', 'fc','m_dep', 'clock_speed',
     'pc', 'blue', 'four_g', 'sc_h', 'three_g', 'dual_sim', 'touch_screen', 'wifi'], axis=1)
    model = sm.OLS(yTrain, xAndConst)
    results = model.fit()
    newCoef= pd.DataFrame({'coef':results.params, 'p-values':round(results.pvalues,3)})
    #print(newCoef.sort_values(by=['p-values']))
    """print('\nBIC :', results.bic)
    print('R-Squared :', results.rsquared)
    print("MSE for Reduced Model :", round(results.mse_resid, 5))"""

    return results.bic,results.rsquared,results.mse_resid

bic, rsq, mse = [], [], [] #600-700
for i in range(600,700):
    #print('\nRandom State : ',i)

    bic.append(bestRandomState(i)[0])

    #print("*"*50)
print(bic.index(min(bic)))
