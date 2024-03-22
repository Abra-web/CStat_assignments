#I did not manage to get notebooks to work with git, so i am making a simple file for now

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression as lm
from sklearn.metrics import r2_score
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import random

random.seed(10)



data = pd.read_csv("OnlineNewsPopularity.csv")

#drop unnecessary columns
# data = data.drop(['car_ID', 'CarName'], axis=1)


# def ohe(data):
#
#     cat_cols = ['symboling', 'fueltype', 'aspiration', 'doornumber', 'carbody', 'drivewheel', 'enginelocation',
#             'enginetype', 'cylindernumber', 'fuelsystem']
#
#     to_encode = list(set(cat_cols).intersection(set(data.columns)))
#
#     data = pd.get_dummies(data, columns=to_encode)
#
#     return data


def calc_adjusted_r2(n,p,r2):
    adjusted_r2 = 1 - ((1 - r2) * (n - 1) / (n - p - 1))
    return adjusted_r2


# FIT THE MODEL AND CALCULATE R2 AND ADJUSTED R2

def build_model(X_train, y_train, X_test, y_test):
    model = lm(fit_intercept=True)
    model.fit(X=X_train, y=y_train)
    y_fitted = model.predict(X_train)
    y_predicted = model.predict(X_test)
    r2 = r2_score(y_test, y_predicted)
    r2_adjusted = calc_adjusted_r2(n=len(X_train), p=len(X_train.columns), r2=r2)
    return r2, r2_adjusted

r2_scores = []


# EXPERIMENT WITH THE NUMBER OF PARAMETERS

# print(len(data))
# data= data.sample(n=10000)
data.rename(columns = {' shares':'shares'}, inplace = True)
data = data.drop(['url'], axis=1)

train, test = train_test_split(data, test_size=0.2)
X_train = train.drop(['shares'], axis=1)
y_train = train['shares']
X_test = test.drop(['shares'], axis=1)
y_test = test['shares']


random.seed(10)
for i in range(1,len(X_train.columns)):
    X_sliced_train = X_train.iloc[:, :i]
    X_sliced_test = X_test.iloc[:, :i]


    r2, r2_adjusted = build_model(X_sliced_train, y_train, X_sliced_test, y_test)

    r2_scores.append([i, r2, r2_adjusted])


# VISUALIZE RESULTS

r2_df = pd.DataFrame(r2_scores, columns=['num_variables', 'r2','adjusted_r2'])

sns.lineplot(data=r2_df, x='num_variables', y='r2', label='R2')
sns.lineplot(data=r2_df, x='num_variables', y='adjusted_r2', label='Adjusted R2')

plt.title('R2 vs adjusted R2 on real data')
plt.xlabel('Number of Predictors')
plt.ylabel('Score')

plt.legend(title='Score Type')

plt.show()






