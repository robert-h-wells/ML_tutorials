# Make prediction for titanic data

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import cross_val_score
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Read in data
titanic = pd.read_csv("https://raw.githubusercontent.com/amueller/scipy-2017-sklearn/master/notebooks/datasets/titanic3.csv")

# 1 will have imputer fix values and 2 will drop values
# Also need to determine the best way to fix nan values for numerical and categorical data         TO DO !!!
titanic2 = titanic.dropna(subset=['pclass', 'sex', 'age', 'sibsp', 'parch', 'fare', 'embarked'])

# Setup target and feature columns
target_1 = titanic['survived']
target_2 = titanic2['survived']

features_1 = titanic[['pclass', 'sex', 'age', 'fare', 'embarked']]
features_2 = titanic2[['pclass', 'sex', 'age', 'fare', 'embarked']]
feature_nam = ['pclass', 'sex', 'age', 'fare', 'embarked']


# Preprocess titanic feature data
if 1==0:
    # Manually set
    num_attr = ['age', 'fare']
    cat_attr = ['pclass', 'sex', 'embarked']

    for i in cat_attr:
        features_1[i].fillna(features_1[i].value_counts().index[0], inplace=True)

if 1==1:
    # Auto Set
    num_attr = features.dtypes == 'float'
    cat_attr = ~num_attr

    for i in range(len(feature_nam)):
        if cat_attr[i] == True:
            print(feature_nam[i])
            val = feature_nam[i]
            features_1[val].fillna(features_1[val].value_counts().index[0], inplace=True)

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

preprocess = ColumnTransformer([
    ("num", num_pipeline, num_attr),
    ("cat", OneHotEncoder(), cat_attr),
])

features_prepared = preprocess.fit_transform(features_1)
features_prepared_2 = preprocess.fit_transform(features_2)

# Set up train and test arrays
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(features_prepared, target_1, random_state=0)
#X_train, X_test, y_train, y_test = train_test_split(features_prepared_2, target_2, random_state=0)


#=====================================================================================================#
# Prediction models

#========== SVC ==========#
from sklearn.svm import SVC

svm_clf = SVC(gamma="auto")
svm_clf.fit(X_train, y_train)
svm_predictions = svm_clf.predict(X_train)
svm_cv_predic = cross_val_predict(svm_clf, X_train, y_train, cv=3)

svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10, scoring='accuracy')
print(svm_scores.mean())

#========== Decision Tree ==========#
from sklearn.tree import DecisionTreeClassifier

tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)
tree_predictions = tree_clf.predict(X_train)
tree_cv_predic = cross_val_predict(tree_clf, X_train, y_train, cv=3)

tree_scores = cross_val_score(tree_clf, X_train, y_train, cv=10, scoring='accuracy')
print(tree_scores.mean())

#========== Random Forest ==========#
from sklearn.ensemble import RandomForestClassifier

forest_clf = RandomForestClassifier()
forest_clf.fit(X_train, y_train)
forest_predictions = forest_clf.predict(X_train)
forest_cv_predic = cross_val_predict(forest_clf, X_train, y_train, cv=3)

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10, scoring='accuracy')
print(forest_scores.mean())

#========== Logistic Regression ==========#
from sklearn.linear_model import LogisticRegression
log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_predictions = log_reg.predict(X_train)
log_cv_predic = cross_val_predict(log_reg, X_train, y_train, cv=3)

log_scores = cross_val_score(log_reg, X_train, y_train, cv=10, scoring='accuracy')
print(log_scores.mean())


#=========== Rank Models ===========#     TO DO !!!
from sklearn.metrics import confusion_matrix
pred_val = svm_cv_predic
val = confusion_matrix(y_train, pred_val)
print(val)

from sklearn.metrics import precision_score, recall_score
print(precision_score(y_train, pred_val))
print(recall_score(y_train, pred_val))

from sklearn.metrics import f1_score
print(f1_score(y_train, pred_val))

#=========== Fine-Tune Models ============#
# Need to look more into this         TO DO !!!
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2,3,4,5]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

# Also want to rank the important attributes
grid_search = GridSearchCV(forest_clf, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(X_train, y_train)

grid_search.best_params_

feature_importances = grid_search.best_estimator_.feature_importances_
print(sorted(zip(feature_importances,feature_nam), reverse=True))