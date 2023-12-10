from flask import Blueprint, jsonify, request
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score
import pandas as pd

xgboost_class = Blueprint("xgboost_class", __name__, url_prefix="/api/v1/algo")

@xgboost_class.route('/xgboost/', methods=['GET'])
def apply_xgboost():
    csv_file_path = 'D:\T&T\Kidney Disease Project\kidneydisease.csv'
    data_from_csv = pd.read_csv(csv_file_path)
    cleaned_data = data_from_csv.dropna()

    if cleaned_data.empty:
        return jsonify({'message': 'No data found in the CSV file or all data is NaN'}), 404

    if len(cleaned_data) < 2:
        return jsonify({'message': 'Insufficient data to split'}), 400

    features = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo',
                'pcv', 'rc']
    target = 'classification'
    x = cleaned_data[features].values
    y = cleaned_data[target].values

    # Convert 'classification' values to integers
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    model = XGBClassifier()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Retrieving the ID from the request parameters
    id = request.args.get('id')

    if id is None:
        return jsonify({'message': 'No ID provided in the request'}), 400

    # Process the new data using the ID
    #new_data = cleaned_data[cleaned_data['age'] == int(id)]  # Replace 'ID_COLUMN' with your actual ID column name

    #if new_data.empty:
        #    return jsonify({'message': f'No data found for ID: {id}'}), 404

    #new_data_predictions = model.predict(label_encoder.transform(new_data[features].values))

    response = {
        #'predictions': label_encoder.inverse_transform(new_data_predictions).tolist(),
        'accuracy': accuracy
    }
    return jsonify(response), 200
