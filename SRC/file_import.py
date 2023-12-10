import datetime

from flask import request
from flask import Blueprint
import pandas as pd
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker
from models import ImportData, ImportDataDetail, db
from flask import jsonify

file_import = Blueprint("file_import", __name__, url_prefix="/api/v1/file")

@file_import.route('/import', methods=['POST'])
def file_import_post():
    data = request.json
    fileName = None
    if data and 'fileName' in data:
        fileName = data['fileName']
    csv_file_path = 'D:\\Unidata\T&T\dataframe\\'+fileName
    data_from_csv = pd.read_csv(csv_file_path)
    cleaned_data = data_from_csv.dropna()
    engine = create_engine('mssql+pymssql://sa:123qwe@127.0.0.1/bookmark')
    Session = sessionmaker(bind=engine)
    session = Session()
    import_data_row = ImportData(
        Name='file.csv',
        TotalRecords=len(data_from_csv),
        AfterNanRecords=len(cleaned_data),
    )
    session.add(import_data_row)
    session.commit()
    for index, row in data_from_csv.iterrows():
        import_data_detail_row = ImportDataDetail(
            age=row['age'] if not pd.isna(row['age']) else None,
            bp=row['bp'] if not pd.isna(row['bp']) else None,
            sg=row['sg'] if not pd.isna(row['sg']) else None,
            al=row['al'] if not pd.isna(row['al']) else None,
            su=row['su'] if not pd.isna(row['su']) else None,
            rbc=row['rbc'] if not pd.isna(row['rbc']) else None,
            pc=row['pc'] if not pd.isna(row['pc']) else None,
            pcc=row['pcc'] if not pd.isna(row['pcc']) else None,
            ba=row['ba'] if not pd.isna(row['ba']) else None,
            bgr=row['bgr'] if not pd.isna(row['bgr']) else None,
            bu=row['bu'] if not pd.isna(row['bu']) else None,
            sc=row['sc'] if not pd.isna(row['sc']) else None,
            sod=row['sod'] if not pd.isna(row['sod']) else None,
            pot=row['pot'] if not pd.isna(row['pot']) else None,
            hemo=row['hemo'] if not pd.isna(row['hemo']) else None,
            pcv=row['pcv'] if not pd.isna(row['pcv']) else None,
            wc=row['wc'] if not pd.isna(row['wc']) else None,
            rc=row['rc'] if not pd.isna(row['rc']) else None,
            htn=row['htn'] if not pd.isna(row['htn']) else None,
            dm=row['dm'] if not pd.isna(row['dm']) else None,
            cad=row['cad'] if not pd.isna(row['cad']) else None,
            appet=row['appet'] if not pd.isna(row['appet']) else None,
            pe=row['pe'] if not pd.isna(row['pe']) else None,
            ane=row['ane'] if not pd.isna(row['ane']) else None,
            classification=row['classification'] if not pd.isna(row['classification']) else None,
            importId=import_data_row.id,
        )
        session.add(import_data_detail_row)
    session.commit()
    session.close()
    return "Successfully File Imported"

@file_import.route('/result/', methods=['GET'])
def file_import_get():
    id = request.args.get('id')
    if not id:
        return jsonify({'message': 'Please provide an ID parameter'}), 400
    # Top 10 records
    top_10 = ImportDataDetail.query.filter_by(importId=id).order_by(ImportDataDetail.id.asc()).limit(10).all()

    # Bottom 10 records
    bottom_10 = ImportDataDetail.query.filter_by(importId=id).order_by(ImportDataDetail.id.desc()).limit(10).all()

    # Merging both lists
    merged_list = top_10 + bottom_10

    if not merged_list:
        return jsonify({'message': 'No data found for the provided ID'}), 404
    data_list = []
    for item in merged_list:
        data_list.append({
            'age': item.age.decode('utf-8') if isinstance(item.age, bytes) else item.age,
            'bp': item.bp.decode('utf-8') if isinstance(item.bp, bytes) else item.bp,
            'sg': item.sg.decode('utf-8') if isinstance(item.sg, bytes) else item.sg,
            'al': item.al.decode('utf-8') if isinstance(item.al, bytes) else item.al,
            'su': item.su.decode('utf-8') if isinstance(item.su, bytes) else item.su,
            'rbc': item.rbc.decode('utf-8') if isinstance(item.rbc, bytes) else item.rbc,
            'pc': item.pc.decode('utf-8') if isinstance(item.pc, bytes) else item.pc,
            'pcc': item.pcc.decode('utf-8') if isinstance(item.pcc, bytes) else item.pcc,
            'ba': item.ba.decode('utf-8') if isinstance(item.ba, bytes) else item.ba,
            'bgr': item.bgr.decode('utf-8') if isinstance(item.bgr, bytes) else item.bgr,
            'bu': item.bu.decode('utf-8') if isinstance(item.bu, bytes) else item.bu,
            'sc': item.sc.decode('utf-8') if isinstance(item.sc, bytes) else item.sc,
            'sod': item.sod.decode('utf-8') if isinstance(item.sod, bytes) else item.sod,
            'pot': item.pot.decode('utf-8') if isinstance(item.pot, bytes) else item.pot,
            'hemo': item.hemo.decode('utf-8') if isinstance(item.hemo, bytes) else item.hemo,
            'pcv': item.pcv.decode('utf-8') if isinstance(item.pcv, bytes) else item.pcv,
            'wc': item.wc.decode('utf-8') if isinstance(item.wc, bytes) else item.wc,
            'rc': item.rc.decode('utf-8') if isinstance(item.rc, bytes) else item.rc,
            'htn': item.htn.decode('utf-8') if isinstance(item.htn, bytes) else item.htn,
            'dm': item.dm.decode('utf-8') if isinstance(item.dm, bytes) else item.dm,
            'cad': item.cad.decode('utf-8') if isinstance(item.cad, bytes) else item.cad,
            'appet': item.appet.decode('utf-8') if isinstance(item.appet, bytes) else item.appet,
            'pe': item.pe.decode('utf-8') if isinstance(item.pe, bytes) else item.pe,
            'ane': item.ane.decode('utf-8') if isinstance(item.ane, bytes) else item.ane,
            'classification': item.classification.decode('utf-8') if isinstance(item.classification, bytes) else item.classification,
             'created_at': item.created_at.strftime("%Y-%m-%d %H:%M:%S") if isinstance(item.created_at, datetime.datetime) else None,
            'importId': item.importId.decode('utf-8') if isinstance(item.importId, bytes) else item.importId,
        })
    return jsonify(data_list), 200



