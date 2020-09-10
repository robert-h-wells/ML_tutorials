import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_openml
from sklearn.metrics import accuracy_score

mnist = fetch_openml('mnist_784', version=1, cache=True)

X_train, X_test, y_train, y_test = train_test_split(
    mnist.data, mnist.target, test_size=10000, random_state=42)

print(np.shape(X_train),np.shape(X_test))
print(np.shape(y_train),np.shape(y_test))

# Train full data with RandomForest
from sklearn.ensemble import RandomForestClassifier
import time

rnd_clf = RandomForestClassifier(n_estimators=100, random_state=42)

t0 = time.time()
rnd_clf.fit(X_train, y_train)
t1 = time.time()

print("Full data Training took {:.2f}s".format(t1 - t0))

y_pred_rf = rnd_clf.predict(X_test)
print(accuracy_score(y_test, y_pred_rf))


# Determine timing with Dimension Reduced Data
from sklearn.decomposition import PCA

pca = PCA(n_components=0.95)
X_reduced = pca.fit_transform(X_train)

rnd_clf2 = RandomForestClassifier(n_estimators=100, random_state=42)

t0 = time.time()
rnd_clf2.fit(X_reduced, y_train)
t1 = time.time()

print("Reduced data Training took {:.2f}s".format(t1 - t0))

X_test_reduced = pca.transform(X_test)

y_pred_rf = rnd_clf2.predict(X_test_reduced)
print(accuracy_score(y_test, y_pred_rf))