# Make prediction for titanic data

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer, make_column_transformer
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error

# Read in data, 1 will have imputer fix values and 2 will drop values
titanic = pd.read_csv("https://raw.githubusercontent.com/amueller/scipy-2017-sklearn/master/notebooks/datasets/titanic3.csv")
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
    num_attr = features_1.dtypes == 'float'
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


#=====================================================================================================#
# Prediction models
print() ; print('=============== Predicition Models ===============')
nam_model = []
type_model = []

#========== SVC ==========#
from sklearn.svm import SVC

nam_model.append('SVC')

svm_clf = SVC(gamma="auto")
svm_clf.fit(X_train, y_train)
svm_predictions = svm_clf.predict(X_train)
svm_cv_predic = cross_val_predict(svm_clf, X_train, y_train, cv=3)

svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10, scoring='accuracy')
print('SVM',svm_scores.mean())
type_model.append(svm_scores.mean())

#========== Decision Tree ==========#
from sklearn.tree import DecisionTreeClassifier

nam_model.append('Decision Tree')

tree_clf = DecisionTreeClassifier()
tree_clf.fit(X_train, y_train)
tree_predictions = tree_clf.predict(X_train)
tree_cv_predic = cross_val_predict(tree_clf, X_train, y_train, cv=3)

tree_scores = cross_val_score(tree_clf, X_train, y_train, cv=10, scoring='accuracy')
print('Decision Tree',tree_scores.mean())
type_model.append(tree_scores.mean())

#========== Random Forest ==========#
from sklearn.ensemble import RandomForestClassifier

nam_model.append('Random Forest')

forest_clf = RandomForestClassifier()
forest_clf.fit(X_train, y_train)
forest_predictions = forest_clf.predict(X_train)
forest_cv_predic = cross_val_predict(forest_clf, X_train, y_train, cv=3)

forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10, scoring='accuracy')
print('Random Forest',forest_scores.mean())
type_model.append(forest_scores.mean())

print(sorted(zip(forest_clf.feature_importances_,feature_nam), reverse=True))

#========== Logistic Regression ==========#
from sklearn.linear_model import LogisticRegression

nam_model.append('Logistic Regression')

log_reg = LogisticRegression()
log_reg.fit(X_train, y_train)
log_predictions = log_reg.predict(X_train)
log_cv_predic = cross_val_predict(log_reg, X_train, y_train, cv=3)

log_scores = cross_val_score(log_reg, X_train, y_train, cv=10, scoring='accuracy')
print('Logistic',log_scores.mean())
type_model.append(log_scores.mean())


#=========== Rank Models ===========#
print() ; print('Model Ranking')
sorted_model = sorted(zip(type_model,nam_model),reverse=True)
print(sorted_model) ; print()

if 1==0:  # Can rank based on precision/recall/f1
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
# Look at each model in detail
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform

# Forest
param_grid_forest = [
    {'n_estimators': [3, 10, 30, 50], 'max_features': [2,3,4,5]},
    {'bootstrap': [False], 'n_estimators': [3, 10, 30, 50], 'max_features': [2, 3, 4]},
  ]

rnd_search_forest = GridSearchCV(forest_clf, param_grid_forest, cv=3, return_train_score=True)
rnd_search_forest.fit(X_train, y_train)

# Linear SVC
param_grid_svc = {"C": uniform(0.1, 10)}
rnd_search_cv = RandomizedSearchCV(svm_clf, param_grid_svc, n_iter=10, verbose=2, cv=3)
rnd_search_cv.fit(X_train, y_train)

# Logistic Regression 
param_grid_logistic = {"C": [0.01,0.1,1,10,50,100]}
rnd_search_logistic = GridSearchCV(log_reg, param_grid_logistic, cv=3, return_train_score=True)
rnd_search_logistic.fit(X_train, y_train)


print() ; print('============= Fine Tuning =============')

print('Forest')
print(rnd_search_forest.best_params_)
print(rnd_search_forest.best_score_) ; print()

print('Linear SVC')
print(rnd_search_cv.best_estimator_)
print(rnd_search_cv.best_score_) ; print()

print('Logistic Regression')
print(rnd_search_logistic.best_estimator_)
print(rnd_search_logistic.best_score_) ; print()
