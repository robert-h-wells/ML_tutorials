import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Problem 7
from sklearn.datasets import make_moons

X, y = make_moons(n_samples=10000, noise=0.4, random_state=42)

# Split into test/train set
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

# Train a simple Decision Tree
from sklearn.tree import DecisionTreeClassifier
tree_clf = DecisionTreeClassifier(max_depth=2)
tree_clf.fit(X_train, y_train)

y_pred = tree_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))

# Fine-Tune Model
from sklearn.model_selection import GridSearchCV

params = {'max_leaf_nodes': list(range(2, 100)), 'min_samples_split': [2, 3, 4]}
grid_search_cv = GridSearchCV(DecisionTreeClassifier(random_state=42), params, verbose=1, cv=3)

grid_search_cv.fit(X_train, y_train)

print(grid_search_cv.best_estimator_)

y_pred = grid_search_cv.predict(X_test)
print(accuracy_score(y_test, y_pred))


# Problem 8
from sklearn.model_selection import ShuffleSplit

n_trees = 1000
n_instances = 100
mini_sets = []

print(len(X_train),len(X_train) - 100)

rs = ShuffleSplit(n_splits=n_trees, test_size=len(X_train)-n_instances, random_state=0)
print(rs)
for mini_train_index, mini_test_index in rs.split(X_train):
    X_mini_train = X_train[mini_train_index]
    y_mini_train = y_train[mini_train_index]
    mini_sets.append((X_mini_train, y_mini_train))

# Clone the best model and test on n_trees data sets
from sklearn.base import clone
from scipy.stats import mode

forest = [clone(grid_search_cv.best_estimator_) for _ in range(n_trees)]

accuracy_scores = []

for tree, (X_mini_train, y_mini_train) in zip(forest, mini_sets):
    tree.fit(X_mini_train,y_mini_train)
    
    y_pred = tree.predict(X_test)
    accuracy_scores.append(accuracy_score(y_test, y_pred))

print(np.mean(accuracy_scores))

# Get the predictions on each data set
Y_pred = np.empty([n_trees, len(X_test)], dtype=np.uint8)

for tree_index, tree in enumerate(forest):
    Y_pred[tree_index] = tree.predict(X_test)

# Choose the prediction from the mode of each data set
from scipy.stats import mode

y_pred_majority_votes, n_votes = mode(Y_pred, axis=0)

print(accuracy_score(y_test, y_pred_majority_votes.reshape([-1])))