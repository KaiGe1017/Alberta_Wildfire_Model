# %%
# import labraries in the process if need 

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

# %%
# read data 
data = pd.read_excel("fp-historical-wildfire-data-2006-2023.xlsx")
data.head()


# %%
# data preparation
# Create a mapping table for forest_area
mapping = {
    'C': 'Calgary',
    'E': 'Edson',
    'G': 'Grande Prairie',
    'H': 'High Level',
    'L': 'Lac La Biche',
    'M': 'Fort McMurray',
    'P': 'Peace River',
    'R': 'Rocky Mountain House',
    'S': 'Slave Lake',
    'W': 'Whitecourt'
}
# Convert the dictionary to a DataFrame
mapping_df = pd.DataFrame(list(mapping.items()), columns=['initial', 'forest_area_full'])
# Extract the first letter of fire_number to create forest_area
data['forest_area'] = data['fire_number'].str[0]
# Merge the original data with the mapping DataFrame
data = data.merge(mapping_df, left_on='forest_area', right_on='initial', how='left')
# Replace the forest_area with the full name
data['forest_area'] = data['forest_area_full']
# Drop the temporary columns used for merging
data.drop(columns=['initial', 'forest_area_full'], inplace=True)

data['reported_date'] = pd.to_datetime(data['reported_date'], errors='coerce')
data['reported_month'] = data['reported_date'].dt.month

print(data['forest_area'].head())
print(data[['reported_date', 'reported_month']].head())

# %%
# select features
selected_columns = [
    'reported_month', 'forest_area', 'fire_location_latitude', 'fire_location_longitude',
    'fire_origin','general_cause_desc', 'weather_conditions_over_fire', 
    'temperature', 'relative_humidity', 'wind_direction',
    'wind_speed', 'size_class'
]

data_relevant = data[selected_columns]

# %%
# Data cleansing
numerical_columns = data_relevant.select_dtypes(include=[np.number]).columns.tolist()
# numerical_columns .remove('fire_location_latitude')
categorical_columns = data_relevant.select_dtypes(include=[object]).columns.tolist()
# categorical_columns.remove('size_class')

# Fill null value by median for numerical_columns
imputer = SimpleImputer(strategy='median')
data_relevant[numerical_columns] = imputer.fit_transform(data_relevant[numerical_columns])
# fill null value by using "missing" for categorical_columns
data_relevant[categorical_columns] = data_relevant[categorical_columns].fillna('missing')

# Create a feature combination
data_relevant['forest_cause_combined'] = data_relevant['forest_area'] + '_' + data_relevant['general_cause_desc']
# Append to categorical_columns
categorical_columns.append('forest_cause_combined')

# check the result
print(data_relevant.isnull().sum())

#%% Check latitude_range in the dataset
# latitude_range = data['fire_location_latitude'].describe()
# print(latitude_range)
# Divide the latitude into groups
# def categorize_latitude(fire_location_latitude):
#     if fire_location_latitude > 56.78:
#         return 'high'
#     elif fire_location_latitude >= 53.19:
#         return 'mid'
#     else:
#         return 'low'

# data_relevant['latitude_category'] = data_relevant['fire_location_latitude'].apply(categorize_latitude)
# categorical_columns.append('latitude_category')

# %%
# Encoding categorical_columns
# data_encoded = pd.get_dummies(data_relevant, columns=categorical_columns)
# check
# print(data_encoded.head())
# Encode categorical features with LabelEncoder

from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    data_relevant[column] = le.fit_transform(data_relevant[column])
    label_encoders[column] = le

# %%
from sklearn.preprocessing import StandardScaler
# numerical_columns standration 
scaler = StandardScaler()
data_relevant[numerical_columns] = scaler.fit_transform(data_relevant[numerical_columns])
# check
print(data_relevant.head())

# %%
# Separation of features and target variables
X = data_relevant.drop(columns=['size_class'])
y = data_relevant['size_class']

# Split to training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


# using random forest classifier 
model = RandomForestClassifier(random_state=42)
model.fit(X_train, y_train)

# Get the importance of features
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

print("Rank of Features:")
for i in range(X.shape[1]):
    print(f"{i + 1}. Feature {X.columns[indices[i]]} ({importances[indices[i]]})")


# %%
# based on the rank from random forest, drop some features.
# keep top 11 features
# data_new=data_relevant.drop(columns=['fire_origin'])
# categorical_columns.remove('fire_origin')
# categorical_columns.remove('size_class')

# keep top 9 features
data_new=data_relevant.drop(columns=['fire_origin','general_cause_desc','forest_area'])
categorical_columns.remove('fire_origin')
categorical_columns.remove('general_cause_desc')
categorical_columns.remove('forest_area')
categorical_columns.remove('size_class')

print(data_new)
print(categorical_columns)
# %%
# pip install pycaret
from pycaret.classification import *
# setting
exp_clf = setup(data=data_new, target='size_class', 
                numeric_features=numerical_columns, 
                categorical_features=categorical_columns,
                session_id=123, 
                verbose=False)
# compre all the models
best_model = compare_models()
top3_models = compare_models(n_select=3)
second_best_model = top3_models[1]
third_best_model = top3_models[2]

# get the best model
print("Best Mode:",best_model)
print("Second Best Model:", second_best_model)
print("Third Best Model:", third_best_model)


# %%
# Tune the hyperparameters of the best model
tuned_model = tune_model(best_model)
# Evaluate the tuned model
evaluate_model(tuned_model)
# Evaluate the model with cross-validation
cv_results = pull()
print(cv_results)

# %%
# Tune the hyperparameters of the second_best model
tuned_model = tune_model(second_best_model)
# Evaluate the tuned model
evaluate_model(tuned_model)
# Evaluate the model with cross-validation
cv_results = pull()
print(cv_results)

# %%
# Tune the hyperparameters of the third_best model
tuned_model = tune_model(third_best_model)
# Evaluate the tuned model
evaluate_model(tuned_model)
# Evaluate the model with cross-validation
cv_results = pull()
print(cv_results)

# %%
# Best parameters for lightgbm

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from lightgbm import LGBMClassifier
# Define the model
model = LGBMClassifier(random_state=2024)
# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}
# Setup GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
# Fit the model
grid_search.fit(X_train, y_train)
# Get the best estimator
best_model = grid_search.best_estimator_
# Predict on the test set
y_pred = best_model.predict(X_test)
# Evaluate the model
print(classification_report(y_test, y_pred))
# Print the best parameters
print("Best parameters for lighgbm found: ", grid_search.best_params_)


# %%
# Best parameters for xgboost

from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from xgboost import XGBClassifier
# Define the model
model = XGBClassifier(random_state=2024)
# Define the parameter grid for hyperparameter tuning
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.1, 0.2],
    'subsample': [0.8, 0.9, 1.0],
    'colsample_bytree': [0.8, 0.9, 1.0]
}
# Setup GridSearchCV
grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=5, scoring='accuracy', verbose=2, n_jobs=-1)
# Fit the model
grid_search.fit(X_train, y_train)
# Get the best estimator
best_model = grid_search.best_estimator_
# Predict on the test set
y_pred = best_model.predict(X_test)
# Evaluate the model
print(classification_report(y_test, y_pred))
# Print the best parameters
print("Best parameters for xgboost found: ", grid_search.best_params_)

# %%
