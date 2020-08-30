# Make prediction for titanic data

import pandas as pd
import numpy as np

# Read in data
titanic = pd.read_csv("https://raw.githubusercontent.com/amueller/scipy-2017-sklearn/master/notebooks/datasets/titanic3.csv")

# Drop data that is not filled in (should think about doing Imputer later)
titanic2 = titanic.dropna(subset=['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked'])

# Setup target and feature columns
target = titanic2['survived']
features = titanic2[['pclass', 'sex', 'age', 'fare', 'embarked']]
feature_nam = ['pclass', 'sex', 'age', 'fare', 'embarked']

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import cross_val_score


# Preprocess data
num_attr = ['age', 'fare']
cat_attr = ['pclass', 'sex', 'embarked']

preprocess = ColumnTransformer([
    ("num", StandardScaler(), num_attr),
    ("cat", OneHotEncoder(), cat_attr),
])

features_prepared = preprocess.fit_transform(features)

# Set up train and test arrays
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features_prepared, target, random_state=0)

# Predict with SVC
from sklearn.svm import SVC

svm_clf = SVC(gamma="auto")
svm_clf.fit(X_train, y_train)
svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
print('SVM',svm_scores.mean())

# Random Forest Regression
from sklearn.ensemble import RandomForestRegressor
forest_reg = RandomForestRegressor()
forest_reg.fit(X_train, y_train)

# Random Forest Classification
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
print('Forest',forest_scores.mean())


# Also want to rank the important attributes
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2,3,4,5]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

grid_search = GridSearchCV(forest_clf, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(X_train, y_train)

grid_search.best_params_

feature_importances = grid_search.best_estimator_.feature_importances_
print(sorted(zip(feature_importances,feature_nam), reverse=True))