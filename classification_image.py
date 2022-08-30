import pandas as pd
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import RadiusNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
import matplotlib.pyplot as plt

pd.set_option('display.max_colwidth', None)  # die komplette Spalte anzeigen
pd.set_option('display.max_columns', None)  # alle Spalten anzeigen

fashion_mnist_df = pd.read_csv('datasets/fashion-mnist_train.csv')

LOOKUP = {0: 'T-Shirt',
          1: 'Trouser',
          2: 'Pullover',
          3: 'Dress',
          4: 'Coat',
          5: 'Sandal',
          6: 'Shirt',
          7: 'Sneaker',
          8: 'Bag',
          9: 'Ankle boot'}


def display_image(features, actual_label):
    print('Actual label:', LOOKUP[actual_label])
    plt.imshow(features.reshape(28, 28))
    plt.show()


X = fashion_mnist_df[fashion_mnist_df.columns[1:]]
Y = fashion_mnist_df['label']

display_image(X.loc[567].values, Y.loc[567])

X = X / 255  # scaling

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)


def summarize_classification(y_test, y_pred, avg_method='weighted'):
    acc = accuracy_score(y_test, y_pred, normalize=True)
    num_acc = accuracy_score(y_test, y_pred, normalize=False)

    prec = precision_score(y_test, y_pred, average=avg_method)
    recall = recall_score(y_test, y_pred, average=avg_method)

    print('Test data count', len(y_test))
    print('accurracy:', acc)
    print('precsion:', prec)
    print('recall:', recall)
    print('accuracy_count:', num_acc)


# this takes a couple of minutes (~5min)
logistic_model = LogisticRegression(solver='sag', multi_class='auto', max_iter=10000).fit(x_train, y_train)

y_pred = logistic_model.predict(x_test)

print('start')

summarize_classification(y_test, y_pred)

print('finished')
