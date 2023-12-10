
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.sql import func

db = SQLAlchemy()

class ImportData(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    Name = db.Column(db.String(100), nullable=False)
    TotalRecords = db.Column(db.Integer, nullable=False)
    AfterNanRecords = db.Column(db.Integer, nullable=False)
    created_at = db.Column(db.DateTime(timezone=True),
                           server_default=func.now())
    import_data_details = db.relationship('ImportDataDetail', backref='import_data')

    def __repr__(self):
        return f'<ImportData {self.Name}>'

class ImportDataDetail(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    age = db.Column(db.String(100), nullable=True)
    bp = db.Column(db.String(100), nullable=True)
    sg = db.Column(db.String(100), nullable=True)
    al = db.Column(db.String(100), nullable=True)
    su = db.Column(db.String(100), nullable=True)
    rbc = db.Column(db.String(100), nullable=True)
    pc = db.Column(db.String(100), nullable=True)
    pcc = db.Column(db.String(100), nullable=True)
    ba = db.Column(db.String(100), nullable=True)
    bgr = db.Column(db.String(100), nullable=True)
    bu = db.Column(db.String(100), nullable=True)
    sc = db.Column(db.String(100), nullable=True)
    sod = db.Column(db.String(100), nullable=True)
    pot = db.Column(db.String(100), nullable=True)
    hemo = db.Column(db.String(100), nullable=True)
    pcv = db.Column(db.String(100), nullable=True)
    wc = db.Column(db.String(100), nullable=True)
    rc = db.Column(db.String(100), nullable=True)
    htn = db.Column(db.String(100), nullable=True)
    dm = db.Column(db.String(100), nullable=True)
    cad = db.Column(db.String(100), nullable=True)
    appet = db.Column(db.String(100), nullable=True)
    pe = db.Column(db.String(100), nullable=True)
    ane = db.Column(db.String(100), nullable=True)
    classification = db.Column(db.String(100), nullable=True)
    importId = db.Column(db.Integer, db.ForeignKey('import_data.id'))
    created_at = db.Column(db.DateTime(timezone=True),
                           server_default=func.now())

    def __repr__(self):
        return f'<ImportDataDetail {self.age}>'

