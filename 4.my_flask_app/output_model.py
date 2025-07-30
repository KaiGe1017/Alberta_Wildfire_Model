from flask import Flask, render_template, request, jsonify
import joblib
import pandas as pd

app = Flask(__name__)

# Load the model and preprocessors
xgboost_model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')
imputer = joblib.load('imputer.pkl')
label_encoders = joblib.load('label_encoders.pkl')

# Month mapping
month_mapping = {
    'January': 1, 'February': 2, 'March': 3, 'April': 4, 'May': 5, 'June': 6,
    'July': 7, 'August': 8, 'September': 9, 'October': 10, 'November': 11, 'December': 12
}
inverse_month_mapping = {v: k for k, v in month_mapping.items()}

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    data = request.json
    input_df = pd.DataFrame([data])

    # Map month names to numbers
    if 'reported_month' in input_df.columns:
        input_df['reported_month'] = input_df['reported_month'].map(month_mapping)

    # Ensure all expected columns are present
    expected_columns = ['reported_month', 'forest_area', 'fire_location_latitude', 'fire_location_longitude',
                        'general_cause_desc', 'weather_conditions_over_fire', 'temperature', 'relative_humidity',
                        'wind_direction', 'wind_speed']
    for column in expected_columns:
        if column not in input_df.columns:
            input_df[column] = None

    # Fill NA values
    numerical_columns = ['reported_month', 'fire_location_latitude', 'fire_location_longitude', 'temperature', 'relative_humidity', 'wind_speed']
    input_df[numerical_columns] = imputer.transform(input_df[numerical_columns])

    # Fill categorical columns with 'missing'
    for column in input_df.select_dtypes(include=[object]).columns:
        input_df[column].fillna('missing', inplace=True)

    # Create combined feature
    input_df['forest_cause_combined'] = input_df['forest_area'] + '_' + input_df['general_cause_desc']
    input_df['forest_cause_combined'].fillna('missing', inplace=True)

    # Encode categorical features for prediction
    categorical_columns = ['weather_conditions_over_fire', 'wind_direction', 'forest_cause_combined']
    encoded_df = input_df.drop(columns=['forest_area', 'general_cause_desc'])
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

    result = input_df[['reported_month', 'forest_area', 'fire_location_latitude', 'fire_location_longitude',
                       'general_cause_desc', 'weather_conditions_over_fire', 'temperature', 'relative_humidity',
                       'wind_direction', 'wind_speed', 'predicted_size_class']].iloc[0].to_dict()
    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True)
