from catboost import CatBoostClassifier
from flask import Blueprint, jsonify, request
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score, mean_absolute_error, roc_auc_score, \
    confusion_matrix, roc_curve, f1_score, precision_score, recall_score
import pandas as pd
import numpy as np
from SRC.logics_regix import get_numeric_value
from models import ImportDataDetail, db

# Define the blueprint
xgboost_class = Blueprint("xgboost_class", __name__, url_prefix="/api/v1/algo")

# XGBoost Route
@xgboost_class.route('/xgboost', methods=['POST'])
def apply_xgboost():
    id = request.args.get('_id')
    key_value = request.json

    if not id or not key_value:
        return jsonify({'message': 'Please provide both id and key-value pairs'}), 400

    new_data = pd.DataFrame([key_value])

    # Fetch data from the database based on the provided criteria
    data = ImportDataDetail.query.filter_by(importId=id).all()

    if not data:
        return jsonify({'message': 'No data found for the provided criteria'}), 404

    cleaned_data = [item for item in data if all(getattr(item, attr) is not None for attr in dir(item) if not attr.startswith('_'))]

    # Prepare the DataFrame
    df = prepare_dataframe(cleaned_data)

    # Extract features and target variable
    target_column = 'classification'
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Define categorical columns
    categorical_columns = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

    # Define a ColumnTransformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns),
            ('num', SimpleImputer(strategy='mean'), X.columns.difference(categorical_columns))
        ]
    )

    # Define the XGBoost classifier
    xgb = XGBClassifier()

    # Create a pipeline with preprocessing and XGBoost classifier
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', xgb)
    ])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=5)

    # Fit the pipeline on training data
    pipe.fit(X_train, y_train)

    # Predict using the pipeline
    predictions = pipe.predict(X_test)

    # Calculate metrics
    metrics = calculate_metrics(pipe, X_test, y_test, new_data)

    return jsonify(metrics)

# CatBoost Route
@xgboost_class.route('/catboost', methods=['POST'])
def apply_catboost():
    id = request.args.get('_id')
    key_value = request.json

    if not id or not key_value:
        return jsonify({'message': 'Please provide both id and key-value pairs'}), 400

    new_data = pd.DataFrame([key_value])

    # Fetch data from the database based on the provided criteria
    data = ImportDataDetail.query.filter_by(importId=id).all()

    if not data:
        return jsonify({'message': 'No data found for the provided criteria'}), 404

    cleaned_data = [item for item in data if all(getattr(item, attr) is not None for attr in dir(item) if not attr.startswith('_'))]

    # Prepare the DataFrame
    df = prepare_dataframe(cleaned_data)

    # Extract features and target variable
    target_column = 'classification'
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Define categorical columns
    categorical_columns = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane']

    # Define a ColumnTransformer for preprocessing
    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), categorical_columns),
            ('num', SimpleImputer(strategy='mean'), X.columns.difference(categorical_columns))
        ]
    )

    # Define the CatBoost classifier
    catboost = CatBoostClassifier(iterations=100, depth=6, learning_rate=0.1)

    # Create a pipeline with preprocessing and CatBoost classifier
    pipe = Pipeline([
        ('preprocessor', preprocessor),
        ('classifier', catboost)
    ])

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=5)

    # Fit the pipeline on training data
    pipe.fit(X_train, y_train)

    # Predict using the pipeline
    predictions = pipe.predict(X_test)

    # Calculate metrics
    metrics = calculate_metrics(pipe, X_test, y_test, new_data)

    return jsonify(metrics)

# Utility function to prepare DataFrame
def prepare_dataframe(cleaned_data):
    return pd.DataFrame([
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

# Utility function to calculate metrics
def calculate_metrics(pipe, X_test, y_test, new_data):
    # Predict using the pipeline
    predictions = pipe.predict(X_test)

    # Calculate metrics
    accuracy = round(accuracy_score(y_test, predictions), 3)
    r2 = round(r2_score(y_test, predictions), 3)
    mae = round(mean_absolute_error(y_test, predictions), 3)
    rmse = round(np.sqrt(mean_squared_error(y_test, predictions)), 3)
    new_predictions = pipe.predict(new_data)
    f1 = round(f1_score(y_test, predictions), 3)
    micro = round(precision_score(y_test, predictions, average='micro'), 3)
    macro = round(recall_score(y_test, predictions, average='macro'), 3)
    #auc_score = round(roc_auc_score(y_test, np.zeros(len(y_test))), 3)

    return {
        'Prediction': new_predictions.tolist(),
        'Accuracy': float(accuracy),
        'R2 Score': float(r2),
        'MAE': float(mae),
        'RMSE': float(rmse),
        'F1 Score': float(f1),
        'Micro': float(micro),
        'Macro': float(macro),
        #'AUC': auc_score
    }

# Additional route for metrics
@xgboost_class.route('/matrix/', methods=['GET'])
def apply_metrics():
    id = request.args.get('_id')

    if not id:
        return jsonify({'message': 'Please provide id'}), 400

    data = ImportDataDetail.query.filter_by(importId=id).all()

    if not data:
        return jsonify({'message': 'No data found for the provided criteria'}), 404

    cleaned_data = [item for item in data if all(getattr(item, attr) is not None for attr in dir(item) if not attr.startswith('_'))]

    # Prepare the DataFrame
    df = prepare_dataframe(cleaned_data)

    target_column = 'classification'
    X = df.drop(columns=[target_column])
    y = df[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.9, random_state=5)

    # Calculate metrics without fitting any model
    tn, fp, fn, tp = confusion_matrix(y_test, np.zeros(len(y_test))).ravel()
    confusion_matrix_result = {
        'True Negative': int(tn),
        'False Positive': int(fp),
        'False Negative': int(fn),
        'True Positive': int(tp)
    }

    # Create ROC Curve without a specific model
    fpr, tpr, _ = roc_curve(y_test, np.zeros(len(y_test)))  # ROC for a model predicting all zeros

    # Prepare the response
    return jsonify({
        'Confusion Matrix': confusion_matrix_result,
        'ROC Curve': {
            'False Positive Rate': fpr.tolist(),
            'True Positive Rate': tpr.tolist()
        }
    })
