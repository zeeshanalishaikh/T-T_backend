from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
import numpy as np
from flask import Blueprint, jsonify, request
from models import ImportDataDetail

logistic_regression = Blueprint("logistic_regression", __name__, url_prefix="/api/v1/algo")

@logistic_regression.route('/logistic/', methods=['GET'])
def apply_logistic_regression():
    id = request.args.get('id')
    if not id:
        return jsonify({'message': 'Please provide an ID parameter'}), 400

    data = ImportDataDetail.query.filter_by(importId=id).all()
    if not data:
        return jsonify({'message': 'No data found for the provided ID'}), 404

    features = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo',
                'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    target = 'classification'

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform([getattr(item, target) for item in data])

    categorical_features = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    for feature in categorical_features:
        encoded_values = label_encoder.fit_transform([getattr(item, feature) for item in data])
        for index, item in enumerate(data):
            setattr(item, feature, encoded_values[index])

    # Convert all values to float except non-numeric values
    X = []
    for item in data:
        row = []
        for feature in features:
            value = getattr(item, feature)
            try:
                row.append(float(value))
            except ValueError:
                row.append(np.nan)  # Replace non-numeric with np.nan
        X.append(row)

    X = np.array(X, dtype=np.float64)  # Convert the whole array to float64

    imp_mean = SimpleImputer(strategy='mean')
    X = imp_mean.fit_transform(X)

    # Remove rows with NaN values
    nan_indices = np.isnan(X).any(axis=1)
    X = X[~nan_indices]
    y = y[~nan_indices]
    model = LogisticRegression()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    model.fit(X_train, y_train)
    # ... (Train-test split, model fitting, prediction, and accuracy calculation)
    predictions = model.predict(X_test)
    accuracy = model.score(X_test, y_test)
    return jsonify({
        'predictions': predictions.tolist(),
        'accuracy': accuracy
    }), 200
