import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score


# Problem 9

mnist = fetch_openml('mnist_784', version=1, cache=True)

X = mnist["data"]
y = mnist["target"].astype(np.uint8)

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# First model to check
from sklearn.svm import LinearSVC

svm_clf = LinearSVC(loss='hinge')
svm_clf.fit(X_train, y_train)

y_pred = svm_clf.predict(X_train)
print(accuracy_score(y_train, y_pred))

# See effect of scaling data
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
svm_clf = LinearSVC(loss='hinge')

svm_clf = LinearSVC(loss='hinge')
svm_clf.fit(X_train_scaled, y_train)

y_pred = svm_clf.predict(X_train_scaled)
print(accuracy_score(y_train, y_pred))

# Fine-Tune Linear SVC
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import reciprocal, uniform

if 1==0:   # Takes a long time
    param_distributions = {"C": uniform(0.1, 10)}
    rnd_search_cv = RandomizedSearchCV(svm_clf, param_distributions, n_iter=10, verbose=2, cv=3)
    rnd_search_cv.fit(X_train_scaled, y_train)

    print(rnd_search_cv.best_estimator_)
    print(rnd_search_cv.best_score_)

# SVM with RBF kernel
from sklearn.svm import SVC

svm_clf_rbf = SVC(gamma="scale")
svm_clf_rbf.fit(X_train_scaled, y_train)

y_pred = svm_clf_rbf.predict(X_train_scaled)
print(accuracy_score(y_train, y_pred))

if 1==0:
    param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
    rnd_search_cv = RandomizedSearchCV(svm_clf_rbf, param_distributions, n_iter=10, verbose=2, cv=3)
    rnd_search_cv.fit(X_train_scaled[:1000], y_train[:1000])

    print(rnd_search_cv.best_estimator_)
    print(rnd_search_cv.best_score_)


# Problem 10
from sklearn.datasets import fetch_california_housing

housing = fetch_california_housing()
X = housing["data"]
y = housing["target"]

# Make test/train set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Scale data
X_train_scaled = scaler.fit_transform(X_train.astype(np.float32))
X_test_scaled = scaler.fit_transform(X_test.astype(np.float32))

from sklearn.svm import LinearSVR

svm_reg = LinearSVR()
svm_reg.fit(X_train_scaled, y_train)

from sklearn.metrics import mean_squared_error

y_pred = lin_svr.predict(X_train_scaled)
mse = mean_squared_error(y_train, y_pred)
rmse = np.sqrt(mse)
print(rmse)

from sklearn.svm import SVR

if 1==0:
    param_distributions = {"gamma": reciprocal(0.001, 0.1), "C": uniform(1, 10)}
    rnd_search_cv = RandomizedSearchCV(SVR(), param_distributions, n_iter=10, verbose=2, cv=3, random_state=42)
    rnd_search_cv.fit(X_train_scaled, y_train)

    y_pred = rnd_search_cv.best_estimator_.predict(X_train_scaled)
    mse = mean_squared_error(y_train, y_pred)
    np.sqrt(mse)