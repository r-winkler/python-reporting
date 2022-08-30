import sklearn

import numpy as np
import scipy as sp
import pandas as pd
from sklearn.datasets import load_breast_cancer, load_boston
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score, accuracy_score

pd.set_option('display.max_colwidth', None)  # die komplette Spalte anzeigen
pd.set_option('display.max_columns', None)  # alle Spalten anzeigen

print(sklearn.__version__)
print(np.__version__)
print(sp.__version__)
print(pd.__version__)


# example dataset
cancer_dataset = load_breast_cancer()
df_cancer_features = pd.DataFrame(cancer_dataset.data, columns=cancer_dataset.feature_names)
df_cancer_target = pd.DataFrame(cancer_dataset.target, columns=["cancer"])
df_cancer = pd.concat([df_cancer_features, df_cancer_target], axis=1)


### Linear regression ###
housing = pd.read_csv('datasets/housing.csv')

fig, ax = plt.subplots(figsize=(12,8))

plt.scatter(housing['total_rooms'], housing['median_house_value'])
plt.xlabel('Total rooms')
plt.ylabel('Median house value')
plt.show()

plt.scatter(housing['housing_median_age'], housing['median_house_value'])
plt.xlabel('Median age')
plt.ylabel('Median house value')
plt.show()

plt.scatter(housing['median_income'], housing['median_house_value'])
plt.xlabel('Median income')
plt.ylabel('Median house value')
plt.show()

housing_corr = housing.corr()
# print(housing_corr)

sns.heatmap(housing_corr, annot=True)
plt.show()

housing = housing.dropna()
housing = housing.drop(housing.loc[housing['median_house_value'] == 500001].index)

housing = pd.get_dummies(housing, columns=['ocean_proximity'])

X = housing.drop(['median_house_value'], axis=1)
Y = housing['median_house_value']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

linear_model = LinearRegression(normalize=True).fit(x_train, y_train)

print("Training Score: ", linear_model.score(x_train, y_train))

predicators = x_train.columns

coef = pd.Series(linear_model.coef_, predicators).sort_values()
print(coef)

y_pred = linear_model.predict(x_test)

df_pred_actual = pd.DataFrame({'predicted': y_pred, 'actual': y_test})
df_pred_actual['diff'] = df_pred_actual['predicted'] - df_pred_actual['actual']
print(df_pred_actual.head(10))

print("Testing score: ", r2_score(y_test, y_pred))

plt.scatter(y_test, y_pred)
plt.show()

df_pred_actual_sample = df_pred_actual.sample(100)
df_pred_actual_sample = df_pred_actual_sample.reset_index()

plt.figure(figsize=(20,10))
plt.plot(df_pred_actual_sample['predicted'], label='Predicted')
plt.plot(df_pred_actual_sample['actual'], label='Actual')
plt.ylabel('median_house_value')
plt.legend()
plt.show()



### Logistic regression ###

median = housing['median_house_value'].median()

housing['above_median'] = (housing['median_house_value'] - median) > 0

X = housing.drop(['median_house_value', 'above_median'], axis=1)
Y = housing['above_median']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

logistic_model = LogisticRegression(solver='liblinear').fit(x_train, y_train)

print("Training Score: ", logistic_model.score(x_train, y_train))

y_pred = logistic_model.predict(x_test)

df_pred_actual = pd.DataFrame({'predicted': y_pred, 'actual': y_test})

print(df_pred_actual.head(10))

print("Testing score: ", accuracy_score(y_test, y_pred))
