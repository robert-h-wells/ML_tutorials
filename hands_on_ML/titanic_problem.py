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

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer


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

from sklearn.model_selection import cross_val_score

svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
print(svm_scores.mean())

# Would like to look at other models and improve the models

# Also want to rank the important attributes