# Will include all the useful code from the book in this file
# Properly label

#======================================================================================#
# Save your model and use it later 
from sklearn.externals import joblib

joblib.dump(my_model, "my_model.pkl")
# and later...
my_model_loaded = joblib.load("my_model.pkl")

#======================================================================================#
# split the data into a train and test set
from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

# split data based on Stratified Shuffling
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

#======================================================================================#
# replace missing values
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
imputer.fit(data)
imputer.statistics_
X = imputer.transform(data)


# transformation pipeline that contains Imputer and Scaler
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])

num_pipeline.fit_transform(data)

#======================================================================================#
# Linear Regression
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(data, data_labels)
lin_reg.predict()

# Evaluate MSE
from sklearn.metrics import mean_squared_error
data_predictions = lin_reg.predict(data_prepared)
lin_mse = mean_squared_error(data_labels, data_predictions)
lin_rmse = np.sqrt(lin_mse)

# Decision Tree Regression  
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor()
tree_reg.fit(data_prepared, data_labels)
data_predictions = tree_reg.predict(data_prepared)


# Cross Validation (K-Fold)
# Split data into K subsets (folds), train and evaluate model K times
# model can be lin_reg, tree_reg, etc.
from sklearn.model_selection import cross_val_score
scores = cross_val_score(model, data_prepared, data_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)

#======================================================================================#
# Grid search to find the best hyperparameters 
# This example uses the RandomForestRegressor
from sklearn.model_selection import GridSearchCV

param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor()

grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)

grid_search.fit(data_prepared, data_labels)
grid_search.best_params_
grid_search.best_estimator_

# Use Randomized when the number of parameters is very big
from sklearn.model_selection import RandomizedSearchCV