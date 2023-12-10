from flask import jsonify, request
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from flask import Blueprint
from models import ImportDataDetail

knn = Blueprint("knn", __name__, url_prefix="/api/v1/algo")


@knn.route('/knn/', methods=['GET'])
def apply_knn():
    from app import db  # Import 'db' here to avoid circular import

    id = request.args.get('id')
    if not id:
        return jsonify({'message': 'Please provide an ID parameter'}), 400

    data = ImportDataDetail.query.filter_by(importId=id).all()
    if not data:
        return jsonify({'message': 'No data found for the provided ID'}), 404

    features = ['age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv',
                'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']
    X = []
    y = []

    for item in data:
        X.append([getattr(item, feature) for feature in features])
        y.append(item.classification)

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y) # Encode the entire target array
    for i in range(len(features)):
        if isinstance(X[0][i], str):
            label_encoder = LabelEncoder()
            column = [row[i] for row in X]
            encoded_column = label_encoder.fit_transform(column)
            for j, row in enumerate(X):
                row[i] = encoded_column[j]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    k = 3
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)
    accuracy = knn.score(X_test, y_test)

    return jsonify({'predictions': predictions.tolist(), 'accuracy': accuracy}), 200
