import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

pd.set_option('display.max_colwidth', None)  # die komplette Spalte anzeigen
pd.set_option('display.max_columns', None)  # alle Spalten anzeigen


titanic_df = pd.read_csv('datasets/titanic_train.csv')

titanic_df.drop(['PassengerId', 'Name', 'Ticket', 'Cabin'], inplace=True, axis=1)

# print(titanic_df[titanic_df.isnull().any(axis=1)].count())

titanic_df = titanic_df.dropna()

fig, ax = plt.subplots(figsize=(12, 8))

plt.scatter(titanic_df['Age'], titanic_df['Survived'])
plt.xlabel('Age')
plt.ylabel('Survived')
plt.show()

plt.scatter(titanic_df['Fare'], titanic_df['Survived'])
plt.xlabel('Fare')
plt.ylabel('Survived')
plt.show()

titanic_df_corr = titanic_df.corr()
# print(titanic_df)

sns.heatmap(titanic_df_corr, annot=True)
plt.show()

print(pd.crosstab(titanic_df['Sex'], titanic_df['Survived']))
print(pd.crosstab(titanic_df['Pclass'], titanic_df['Survived']))

label_encoding = preprocessing.LabelEncoder()
titanic_df['Sex'] = label_encoding.fit_transform(titanic_df['Sex'].astype(str))

# print(label_encoding.classes_)
titanic_df = pd.get_dummies(titanic_df, columns=['Embarked'])

# alle Datensätze werden gemischt zurückgegeben
titanic_df = titanic_df.sample(frac=1).reset_index(drop=True)

titanic_df.to_csv('datasets/titanic_processed.csv', index=False)

X = titanic_df.drop(['Survived'], axis=1)
Y = titanic_df['Survived']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

logistic_model = LogisticRegression(penalty='l2', C=1.0, solver='liblinear').fit(x_train, y_train)

print("Training Score: ", logistic_model.score(x_train, y_train))

y_pred = logistic_model.predict(x_test)

pred_results = pd.DataFrame({'y_test': y_test, 'y_pred': y_pred})

print(pred_results.head(10))

titanic_crosstab = pd.crosstab(pred_results.y_pred, pred_results.y_test)
print(titanic_crosstab)

print("Accuracy score: ", accuracy_score(y_test, y_pred))
print("Precision score: ", precision_score(y_test, y_pred))  # Wieviele von der vorherhergesagten "survived", waren wirklich "survived" (40/(40+11))
print("Recall score: ", recall_score(y_test, y_pred))  # Wieviele von der wirklichen "survived" hat das Modell richtig vorausgesagt (40/(40+21))
