from flask import Flask
from flask_cors import CORS
from flask_sqlalchemy import SQLAlchemy
from SRC.file_import import file_import
from sqlalchemy.sql import func
from SRC.knn import knn
from SRC.logistic_regression import logistic_regression
from SRC.xgboost_class import xgboost_class
from SRC.catboost_class import catboost_class
from models import db

app = Flask(__name__)
CORS(app)
def create_app(test_config=None):
    app = Flask(__name__, instance_relative_config=True)
app.config['SQLALCHEMY_DATABASE_URI'] = "mssql+pymssql://sa:123qwe@127.0.0.1/bookmark"

app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

db.init_app(app)
with app.app_context():
    db.create_all()

@app.route('/')
def hello_world():  # put application's code here
    return 'Hello From Flask'

app.register_blueprint(file_import)
app.register_blueprint(knn)
app.register_blueprint(logistic_regression)
app.register_blueprint(xgboost_class)
app.register_blueprint(catboost_class   )

if __name__ == '__main__':
    app.run(debug=True)