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

pd.set_option('display.max_colwidth', None)  # die komplette Spalte anzeigen
pd.set_option('display.max_columns', None)  # alle Spalten anzeigen

titanic_df = pd.read_csv('datasets/titanic_processed.csv')

FEATURES = list(titanic_df.columns[1:])

result_dict = {}


def summarize_classification(y_test, y_pred):
    acc = accuracy_score(y_test, y_pred, normalize=True)
    num_acc = accuracy_score(y_test, y_pred, normalize=False)

    prec = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    return {'accurracy': acc,
            'precsion': prec,
            'recall': recall,
            'accuracy_count': num_acc}


def build_model(classifier_fn,
                name_of_y_col,
                names_of_x_cols,
                dataset,
                test_frac=0.2):
    X = dataset[names_of_x_cols]
    Y = dataset[name_of_y_col]

    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=test_frac)
    model = classifier_fn(x_train, y_train)

    y_pred = model.predict(x_test)
    y_pred_train = model.predict(x_train)

    train_summary = summarize_classification(y_train, y_pred_train)
    test_summary = summarize_classification(y_test, y_pred)

    pred_results = pd.DataFrame({'y_test': y_test,
                                 'y_pred': y_pred})

    model_crosstab = pd.crosstab(pred_results.y_pred, pred_results.y_test)

    return {'training': train_summary,
            'test': test_summary,
            'confusion_matrix': model_crosstab}


def compare_results():
    for key in result_dict:
        print('Classification: ', key)

        print()
        print('Training data')
        for score in result_dict[key]['training']:
            print(score, result_dict[key]['training'][score])

        print()
        print('Test data')
        for score in result_dict[key]['test']:
            print(score, result_dict[key]['test'][score])

        print()
        print(result_dict[key]['confusion_matrix'])

        print()


def logistic_fn(x_train, y_train):
    model = LogisticRegression(solver='liblinear')
    model.fit(x_train, y_train)

    return model


def linear_discriminant_fn(x_train, y_train, solver='svd'):
    model = LinearDiscriminantAnalysis(solver=solver)
    model.fit(x_train, y_train)

    return model


def quadratic_discriminant_fn(x_train, y_train):
    model = QuadraticDiscriminantAnalysis()
    model.fit(x_train, y_train)

    return model


def sgd_fn(x_train, y_train, max_iter=10000, tol=1e-3):
    model = SGDClassifier(max_iter=max_iter, tol=tol)
    model.fit(x_train, y_train)

    return model


def linear_svc_fn(x_train, y_train, C=1.0, max_iter=1000, tol=1e-3):
    model = LinearSVC(C=C, max_iter=max_iter, tol=tol, dual=False)
    model.fit(x_train, y_train)

    return model


def radius_neighbor_fn(x_train, y_train, radius=40.0):
    model = RadiusNeighborsClassifier(radius=radius)
    model.fit(x_train, y_train)

    return model


def decision_tree_fn(x_train, y_train, max_depth=None, max_features=None):
    model = DecisionTreeClassifier(max_depth=max_depth, max_features=max_features)
    model.fit(x_train, y_train)

    return model


def naive_bayes_fn(x_train, y_train, priors=None):
    model = GaussianNB(priors=priors)
    model.fit(x_train, y_train)

    return model


result_dict['survived - logistic'] = build_model(logistic_fn,
                                                 'Survived',
                                                 FEATURES,
                                                 titanic_df)

result_dict['survived - linear_discriminant_analysis'] = build_model(linear_discriminant_fn,
                                                                     'Survived',
                                                                     FEATURES[0:-1],
                                                                     titanic_df)

result_dict['survived - quadratic_discriminant_analysis'] = build_model(quadratic_discriminant_fn,
                                                                        'Survived',
                                                                        FEATURES[0:-1],
                                                                        titanic_df)

result_dict['survived - sgd'] = build_model(sgd_fn,
                                            'Survived',
                                            FEATURES,
                                            titanic_df)

result_dict['survived - linear_svc'] = build_model(linear_svc_fn,
                                                   'Survived',
                                                   FEATURES,
                                                   titanic_df)

result_dict['survived - radius_neighbors'] = build_model(radius_neighbor_fn,
                                                         'Survived',
                                                         FEATURES,
                                                         titanic_df)

result_dict['survived - decision_tree'] = build_model(decision_tree_fn,
                                                      'Survived',
                                                      FEATURES,
                                                      titanic_df)

result_dict['survived - naive_bayes'] = build_model(naive_bayes_fn,
                                                    'Survived',
                                                    FEATURES,
                                                    titanic_df)

compare_results()

print()
print()
print('#### hyperparameter tuning for decision tree ###')
print()

parameters = {'max_depth': [2, 4, 5, 7, 9, 10]}

X = titanic_df[FEATURES]
Y = titanic_df['Survived']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

grid_search = GridSearchCV(DecisionTreeClassifier(), parameters, cv=3, return_train_score=True)
grid_search.fit(x_train, y_train)

print(grid_search.best_params_)

for i in range(6):
    print('Parameters: ', grid_search.cv_results_['params'][i])
    print('Mean Test Score: ', grid_search.cv_results_['mean_test_score'][i])
    print('Rank: ', grid_search.cv_results_['rank_test_score'][i])

decision_tree_model = DecisionTreeClassifier(max_depth=grid_search.best_params_['max_depth']).fit(x_train, y_train)
y_pred = decision_tree_model.predict(x_test)
print(summarize_classification(y_test, y_pred))

print()
print()
print('#### hyperparameter tuning for logistic regression ###')
print()

parameters = {'penalty': ['l1', 'l2'],
              'C': [0.1, 0.4, 0.8, 1, 2, 5]}

X = titanic_df[FEATURES]
Y = titanic_df['Survived']

x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.2)

grid_search = GridSearchCV(LogisticRegression(solver='liblinear'), parameters, cv=3, return_train_score=True)
grid_search.fit(x_train, y_train)

print(grid_search.best_params_)

for i in range(6):
    print('Parameters: ', grid_search.cv_results_['params'][i])
    print('Mean Test Score: ', grid_search.cv_results_['mean_test_score'][i])
    print('Rank: ', grid_search.cv_results_['rank_test_score'][i])

logistic_regression_model = LogisticRegression(solver='liblinear', penalty=grid_search.best_params_['penalty'], C=grid_search.best_params_['C']).fit(x_train, y_train)
y_pred = logistic_regression_model.predict(x_test)
print(summarize_classification(y_test, y_pred))
