import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
import datetime
import numpy as np

pd.set_option('display.max_colwidth', None)  # die komplette Spalte anzeigen
pd.set_option('display.max_columns', None)  # alle Spalten anzeigen

automobile_df = pd.read_csv('datasets/auto-mpg.csv')

automobile_df = automobile_df.replace('?', np.nan)
automobile_df = automobile_df.dropna()
automobile_df = automobile_df.drop(['origin', 'car name'], axis=1)

automobile_df['model year'] = '19' + automobile_df['model year'].astype(str)

automobile_df['age'] = datetime.datetime.now().year - pd.to_numeric(automobile_df['model year'])
automobile_df = automobile_df.drop(['model year'], axis=1)

automobile_df['horsepower'] = pd.to_numeric(automobile_df['horsepower'], errors='coerce')

print(automobile_df.dtypes)
print(automobile_df.sample(5))

fig, ax = plt.subplots(figsize=(12, 8))

plt.scatter(automobile_df['age'], automobile_df['mpg'])
plt.xlabel('age')
plt.ylabel('mpg')
plt.show()

plt.scatter(automobile_df['acceleration'], automobile_df['mpg'])
plt.xlabel('acceleration')
plt.ylabel('mpg')
plt.show()

plt.scatter(automobile_df['weight'], automobile_df['mpg'])
plt.xlabel('weight')
plt.ylabel('mpg')
plt.show()

plt.scatter(automobile_df['displacement'], automobile_df['mpg'])
plt.xlabel('displacement')
plt.ylabel('mpg')
plt.show()

plt.scatter(automobile_df['horsepower'], automobile_df['mpg'])
plt.xlabel('horsepower')
plt.ylabel('mpg')
plt.show()

plt.scatter(automobile_df['cylinders'], automobile_df['mpg'])
plt.xlabel('cylinders')
plt.ylabel('mpg')
plt.show()


automobile_df_corr = automobile_df.corr()
sns.heatmap(automobile_df_corr, annot=True)
plt.show()

#shuffle dataset
automobile_df = automobile_df.sample(frac=1).reset_index(drop=True)

automobile_df.to_csv('datasets/auto-mpg-processed.csv', index=False)
