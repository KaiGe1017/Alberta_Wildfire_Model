# %%
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from sklearn.metrics import classification_report
import joblib

# %%
# Load data
data = pd.read_excel("fp-historical-wildfire-data-2006-2023.xlsx")

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

# Data preprocessing
data['reported_date'] = pd.to_datetime(data['reported_date'], errors='coerce')
data['reported_month'] = data['reported_date'].dt.month

# Select relevant columns
selected_columns = [
    'reported_month', 
    'forest_area', 
    'fire_location_latitude', 'fire_location_longitude',
    'general_cause_desc', 
    'weather_conditions_over_fire', 
    'temperature', 'relative_humidity', 'wind_direction',
    'wind_speed', 'size_class'
]

data_relevant = data[selected_columns].copy()

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

# %%

# Encoding categorical_columns
from sklearn.preprocessing import LabelEncoder
label_encoders = {}
for column in categorical_columns:
    le = LabelEncoder()
    data_relevant[column] = data_relevant[column].astype(str)
    le.fit(list(data_relevant[column]) + ['unknown'])  
    data_relevant[column] = data_relevant[column].apply(lambda x: x if x in le.classes_ else 'unknown')
    data_relevant[column] = le.transform(data_relevant[column])
    label_encoders[column] = le

# %%
# numerical_columns standration 
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
data_relevant[numerical_columns] = scaler.fit_transform(data_relevant[numerical_columns])
# check
print(data_relevant.head())

# %%
# Separation of features and target variables
# X = data_relevant.drop(columns=['size_class'])
X = data_relevant.drop(columns=['size_class','general_cause_desc','forest_area'])
y = data_relevant['size_class']

# Split to training and test data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# %%
# Apply SMOTE to handle class imbalance
smote = SMOTE(random_state=2024)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)

# %%
from lightgbm import LGBMClassifier
# Define the LGBM model with best parameters found
# 'colsample_bytree': 1.0, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 50, 'subsample': 0.8
lgbm_model = LGBMClassifier(
    colsample_bytree=1.0,
    learning_rate=0.1,
    max_depth=7,
    n_estimators=50,
    subsample=0.8,
    force_col_wise='true',
    random_state=2024
)
# Train the model
lgbm_model.fit(X_train_resampled, y_train_resampled)

# Predict on the test set
y_pred = lgbm_model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Print the feature importances
importances = lgbm_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature importances:")
for i in range(len(importances)):
    print(f"{i + 1}. Feature {X.columns[indices[i]]} ({importances[indices[i]]})")

# %%
from xgboost import XGBClassifier
# Define the xgboost model with best parameters found
# 'colsample_bytree': 0.9, 'learning_rate': 0.1, 'max_depth': 7, 'n_estimators': 100, 'subsample': 0.9
xgboost_model = XGBClassifier(
    colsample_bytree=0.9,
    learning_rate=0.1,
    max_depth=7,
    n_estimators=100,
    subsample=0.9,
    random_state=2024
)
# Train the model
xgboost_model.fit(X_train_resampled, y_train_resampled)

# Predict on the test set
y_pred = xgboost_model.predict(X_test)

# Evaluate the model
print(classification_report(y_test, y_pred))

# Print the feature importances
importances = xgboost_model.feature_importances_
indices = np.argsort(importances)[::-1]

print("Feature importances:")
for i in range(len(importances)):
    print(f"{i + 1}. Feature {X.columns[indices[i]]} ({importances[indices[i]]})")

# %%
# Save the model
joblib.dump(lgbm_model, 'lgbm_model.pkl')
joblib.dump(xgboost_model, 'xgboost_model.pkl')

# Save the scalers and encoders
joblib.dump(scaler, 'scaler.pkl')
joblib.dump(imputer, 'imputer.pkl')
joblib.dump(label_encoders, 'label_encoders.pkl')

# %%
