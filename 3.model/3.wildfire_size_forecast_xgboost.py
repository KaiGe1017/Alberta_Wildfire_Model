# %%
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# Load the model and preprocessors
xgboost_model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Read the input excel file
input_df = pd.read_excel('user_input.xlsx')

# Map month names to numbers
month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}
inverse_month_mapping = {v: k for k, v in month_mapping.items()}

if 'reported_month' in input_df.columns:
    input_df['reported_month'] = input_df['reported_month'].map(month_mapping)

# Ensure all expected columns are present
expected_columns = ['reported_month', 
                    'forest_area', 
                    'fire_location_latitude', 'fire_location_longitude',
                    'general_cause_desc', 
                    'weather_conditions_over_fire', 'temperature', 'relative_humidity',
                    'wind_direction', 'wind_speed']
for column in expected_columns:
    if column not in input_df.columns:
        input_df[column] = None

# Fill NA values
numerical_columns = input_df.select_dtypes(include=[float, int]).columns.tolist()
input_df[numerical_columns] = imputer.transform(input_df[numerical_columns])

# Fill categorical columns with 'missing'
for column in input_df.select_dtypes(include=[object]).columns:
    input_df[column].fillna('missing', inplace=True)

# Create combined feature
input_df['forest_cause_combined'] = input_df['forest_area'] + '_' + input_df['general_cause_desc']
input_df['forest_cause_combined'].fillna('missing', inplace=True)  # Ensure no NA in combined feature

# Add the combined feature to the categorical columns list
# categorical_columns = ['forest_area', 'general_cause_desc', 'weather_conditions_over_fire', 'wind_direction', 'forest_cause_combined']
categorical_columns = ['weather_conditions_over_fire', 'wind_direction', 'forest_cause_combined']

# Encode categorical features for prediction
# encoded_df = input_df.copy()
encoded_df = input_df.drop(columns=['general_cause_desc','forest_area'])
# for column in categorical_columns:
#     if column in encoded_df.columns:
#         encoded_df[column] = label_encoders[column].transform(encoded_df[column])
for column in categorical_columns:
    if column in encoded_df.columns:
        known_categories = set(label_encoders[column].classes_)
        encoded_df[column] = encoded_df[column].apply(lambda x: x if x in known_categories else 'unknown')
        encoded_df[column] = label_encoders[column].transform(encoded_df[column])

# Standardize numerical features
encoded_df[numerical_columns] = scaler.transform(encoded_df[numerical_columns])

# Predict size_class
predicted_class = xgboost_model.predict(encoded_df)

# Decode the predicted size_class
size_class_encoder = label_encoders['size_class']
input_df['predicted_size_class'] = size_class_encoder.inverse_transform(predicted_class)

# Convert reported_month back to month names
if 'reported_month' in input_df.columns:
    input_df['reported_month'] = input_df['reported_month'].map(inverse_month_mapping)

# Print results
print(input_df[['reported_month', 
                'forest_area', 
                'fire_location_latitude', 'fire_location_longitude',
                'general_cause_desc', 
                'weather_conditions_over_fire', 'temperature', 'relative_humidity',
                'wind_direction', 'wind_speed', 'predicted_size_class']])

# Output the results to Excel
output_file_path = 'prediction_results_xgboost.xlsx'
input_df.to_excel(output_file_path, index=False)
print(f"Predicted results have been saved to {output_file_path}")
