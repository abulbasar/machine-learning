#!/usr/bin/env python
# coding: utf-8


import pandas as pd
import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn import pipeline, preprocessing, compose, linear_model, impute, model_selection


# Load data
print("Loading the training observations")
df = pd.read_csv("https://raw.githubusercontent.com/abulbasar/data/master/insurance.csv")
print("Training data: \n", df.head())

target = "charges"
y = np.log10(df[target])
X = df.drop(columns=[target])


# Categorical features
cat_columns = ["gender", "smoker", "region"]

# Numeric features
num_columns = ["age", "bmi", "children"]


# Build pipelines for categorical data and numeric data
cat_pipe = pipeline.Pipeline([
    ('imputer', impute.SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', preprocessing.OneHotEncoder(handle_unknown='ignore'))
]) 

num_pipe = pipeline.Pipeline([
    ('imputer', impute.SimpleImputer(strategy='median')),
    ('poly', preprocessing.PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', preprocessing.StandardScaler()),
])

preprocessing_pipe = compose.ColumnTransformer([
    ("cat", cat_pipe, cat_columns),
    ("num", num_pipe, num_columns)
])


# Build estimator pipeline
estimator_pipe = pipeline.Pipeline([
    ("preprocessing", preprocessing_pipe),
    ("est", linear_model.ElasticNet(random_state=1))
])


# Parameter grid to tune hyper parameters
param_grid = {
    "est__alpha": 0.0 + np.random.random(10) * 0.02,
    "est__l1_ratio": np.linspace(0.0001, 1, 20),
}

# Grid Search estimator
gsearch = model_selection.GridSearchCV(estimator_pipe, param_grid, cv = 5, verbose=1, n_jobs=8)

# Fit grid search estimator
print("Fitting the model")
gsearch.fit(X, y)

print("Gridsearch best score: ", gsearch.best_score_, "best params: ", gsearch.best_params_)

# Sanity test the quality of model
y_pred = gsearch.predict(X)
plt.scatter(y, y_pred - y)

print("Sample predictions: ", pd.DataFrame({"actual": y, "predict": y_pred}).sample(10))


# Save the tuned model
path = "/tmp/model.pickle"
with open(path, "wb") as f:
    pickle.dump(gsearch, f)

print("Saved the model: " + path, "Now start or restart the flask web application.")