import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score

mnist = fetch_openml('mnist_784', version=1, cache=True)

X = mnist["data"]
y = mnist["target"].astype(np.uint8)
print(np.shape(X),np.shape(y))

# Make train/validate/test set
from sklearn.model_selection import train_test_split

X_train_val, X_test, y_train_val, y_test = train_test_split(
    mnist.data, mnist.target, test_size=10000, random_state=42)

X_train, X_val, y_train, y_val = train_test_split(
    X_train_val, y_train_val, test_size=10000, random_state=42)

print(np.shape(X_train),np.shape(X_test),np.shape(X_val))
print(np.shape(y_train),np.shape(y_test),np.shape(y_val))

# Random Forest
from sklearn.ensemble import RandomForestClassifier

rnd_clf = RandomForestClassifier()
rnd_clf.fit(X_train, y_train)

y_pred_rf = rnd_clf.predict(X_val)
accuracy_score(y_val, y_pred_rf)

# Extra Trees
from sklearn.ensemble import ExtraTreesClassifier

xtra_clf = ExtraTreesClassifier()
xtra_clf.fit(X_train, y_train) 

y_pred_xtra = xtra_clf.predict(X_val)
accuracy_score(y_val, y_pred_xtra)

# SVC
from sklearn.svm import SVC

svm_clf = SVC()
svm_clf.fit(X_train, y_train) 

y_pred_svm = svm_clf.predict(X_val)
accuracy_score(y_val, y_pred_svm)

# Put them into a voting classifier
from sklearn.ensemble import VotingClassifier

voting_clf = VotingClassifier(
    estimators=[('rnd', rnd_clf), ('xtra', xtra_clf), ('svm', svm_clf)],
    voting='hard')
voting_clf.fit(X_train, y_train)

y_pred_vot = svm_clf.predict(X_val)
accuracy_score(y_val, y_pred_vot)