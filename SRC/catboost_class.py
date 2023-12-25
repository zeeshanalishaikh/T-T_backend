from flask import Blueprint, jsonify, request
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import  OneHotEncoder
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error, r2_score
import numpy as np
import pandas as pd

from SRC.logics_regix import get_numeric_value
from SRC.models import ImportDataDetail

catboost_class = Blueprint("catboost_class", __name__, url_prefix="/api/v1/algo")

@catboost_class.route('/catboost/', methods=['GET'])
def apply_catboost():
    id = request.args.get('id')
    key_value = request.json

    if not id or not key_value:
        return jsonify({'message': 'Please provide both id and key-value pairs'}), 400

    new_data = pd.DataFrame([key_value])

    # Fetch data from the database based on the provided criteria
    data = ImportDataDetail.query.filter_by(importId=id).all()

    if not data:
        return jsonify({'message': 'No data found for the provided criteria'}), 404

    cleaned_data = [item for item in data if
                    all(getattr(item, attr) is not None for attr in dir(item) if not attr.startswith('_'))]
    df = pd.DataFrame([
        (
            int(float(get_numeric_value(item.age))) if not pd.isna(get_numeric_value(item.age)) else None,
            float(get_numeric_value(item.bp)) if not pd.isna(get_numeric_value(item.bp)) else None,
            float(get_numeric_value(item.sg)) if not pd.isna(get_numeric_value(item.sg)) else None,
            float(get_numeric_value(item.al)) if not pd.isna(get_numeric_value(item.al)) else None,
            float(get_numeric_value(item.su)) if not pd.isna(get_numeric_value(item.su)) else None,
            float(get_numeric_value(item.bgr)) if not pd.isna(get_numeric_value(item.bgr)) else None,
            float(get_numeric_value(item.bu)) if not pd.isna(get_numeric_value(item.bu)) else None,
            float(get_numeric_value(item.sc)) if not pd.isna(get_numeric_value(item.sc)) else None,
            float(get_numeric_value(item.sod)) if not pd.isna(get_numeric_value(item.sod)) else None,
            float(get_numeric_value(item.pot)) if not pd.isna(get_numeric_value(item.pot)) else None,
            float(get_numeric_value(item.hemo)) if not pd.isna(get_numeric_value(item.hemo)) else None,
            float(get_numeric_value(item.pcv)) if not pd.isna(get_numeric_value(item.pcv)) else None,
            float(get_numeric_value(item.wc)) if not pd.isna(get_numeric_value(item.wc)) else None,
            float(get_numeric_value(item.rc)) if not pd.isna(get_numeric_value(item.rc)) else None,
            str(item.rbc),
            str(item.pc),
            str(item.pcc),
            str(item.ba),
            str(item.htn),
            str(item.dm),
            str(item.cad),
            str(item.appet),
            str(item.pe),
            str(item.ane),
            1 if str(item.classification) == 'ckd' else 0
        ) for item in cleaned_data
    ], columns=['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc', 'rbc', 'pc',
                'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification'])

    target_column = 'classification'
    X = df.drop(columns=[target_column])  # Features (excluding the target)
    y = df[target_column]
    categorical_columns = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']
    # Define a ColumnTransformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns),
            ('num', SimpleImputer(strategy='mean'), X.columns.difference(categorical_columns))
        ]
    )

    # Define the CatBoost classifier
    catboost = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1)  # You can adjust parameters as needed

    # Create a pipeline with preprocessing and CatBoost classifier
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', catboost)
    ])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=5)

    # Fit the pipeline on training data
    pipe.fit(X_train, y_train)

    # Predict using the pipeline
    predictions = pipe.predict(X_test)

    # Calculate accuracy and other metrics
    accuracy = accuracy_score(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    mae = mean_absolute_error(y_test, predictions)
    rmse = np.sqrt(mean_squared_error(y_test, predictions))
    new_predictions = pipe.predict(new_data)
    return jsonify({
        'Prediction': new_predictions.tolist(),
        'Accuracy': float(accuracy),
        'R2 Score': float(r2),
        'MAE': float(mae),
        'RMSE': float(rmse)
    })
