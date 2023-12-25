import pandas as pd
import numpy as np
import sys
def get_numeric_value(value):
    if pd.isna(value) or str(value).strip() in ['?', '\t', '']:
        return None
    try:
        return float(value)
    except ValueError:
        return None


def convert_to_serializable(item):
    if isinstance(item, np.integer):
        return int(item)
    elif isinstance(item, np.floating):
        return float(item)
    elif isinstance(item, np.bool_):
        return bool(item)
    else:
        return item

def get_size(obj):
    size = sys.getsizeof(obj)
    units = ['B', 'KiB', 'MiB', 'GiB', 'TiB']
    index = 0
    while size >= 1024 and index < len(units) - 1:
        size /= 1024
        index += 1
    return f"{size:.1f} {units[index]}"

def calculate_statistics(df):
    num_variables = len(df.columns)
    num_observations = len(df)
    missing_cells = df.isnull().sum().sum()
    duplicate_rows = df.duplicated().sum()
    missing_cells_percentage = (missing_cells / (df.shape[0] * df.shape[1])) * 100
    duplicate_rows_percentage = (duplicate_rows / df.shape[0]) * 100
    total_size_memory = get_size(df)
    total_size_memory_for_memory = sys.getsizeof(df)
    average_record_size_mem = total_size_memory_for_memory / df.shape[0]
    average_record_size_memory = get_size(average_record_size_mem)
    numeric = df.select_dtypes(include=['float64', 'int64']).columns.tolist()
    numeric_count = len(numeric)
    categorical = df.select_dtypes(include=['object']).columns.tolist()
    categorical_count = len(categorical)
    boolean = df.select_dtypes(include=['bool']).columns.tolist()
    boolean_count = len(boolean)
    # Construct results
    int_columns = df.select_dtypes(include=['int64']).columns
    for col in int_columns:
        df[col] = df[col].astype(str)

    # Convert DataFrame to dictionary with serialized values
    df_dict = df.apply(lambda x: x.apply(convert_to_serializable)).to_dict(orient='list')

    # Construct results
    results = {
        "Number of variables": num_variables,
        "Number of observations": num_observations,
        "Missing cells": int(missing_cells),
        "Missing cells (%)": missing_cells_percentage,
        "Duplicate rows": int(duplicate_rows),
        "Duplicate rows( %)": duplicate_rows_percentage,
        "Total size in memory": total_size_memory,
        "Average record size in memory": average_record_size_memory,
        "Numeric": int(numeric_count),
        "categorical": int(categorical_count),
        "boolean": int(boolean_count)
    }
    return results


def create_dataframe_heat(data):
    columns = [
        'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
        'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification'
    ]

    # Creating the DataFrame
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
            str(item.rbc) if not pd.isna(item.rbc) else None,
            str(item.pc) if not pd.isna(item.pc) else None,
            str(item.pcc) if not pd.isna(item.pcc) else None,
            str(item.ba) if not pd.isna(item.ba) else None,
            str(item.htn) if not pd.isna(item.htn) else None,
            str(item.dm) if not pd.isna(item.dm) else None,
            str(item.cad) if not pd.isna(item.cad) else None,
            str(item.appet) if not pd.isna(item.appet) else None,
            str(item.pe) if not pd.isna(item.pe) else None,
            str(item.ane) if not pd.isna(item.ane) else None,
            str(item.classification) if not pd.isna(item.classification) else None
        ) for item in data
    ], columns=columns)

    return df


def transposeNum(data):
    processed_data = [
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
        ) for item in data
    ]

    columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    processed_df = pd.DataFrame(processed_data, columns=columns)

    return processed_df

def transposeCat(data):
    processed_data = [
        (
            str(item.rbc) if not pd.isna(item.rbc) else None,
            str(item.pc) if not pd.isna(item.pc) else None,
            str(item.pcc) if not pd.isna(item.pcc) else None,
            str(item.ba) if not pd.isna(item.ba) else None,
            str(item.htn) if not pd.isna(item.htn) else None,
            str(item.dm) if not pd.isna(item.dm) else None,
            str(item.cad) if not pd.isna(item.cad) else None,
            str(item.appet) if not pd.isna(item.appet) else None,
            str(item.pe) if not pd.isna(item.pe) else None,
            str(item.ane) if not pd.isna(item.ane) else None,
            str(item.classification) if not pd.isna(item.classification) else None
        ) for item in data
    ]
    columns = ['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']
    processed_df = pd.DataFrame(processed_data, columns=columns)
    return processed_df

def findNull(data):
    transformed_data = [
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
            str(item.rbc) if not pd.isna(item.rbc) else None,
            str(item.pc) if not pd.isna(item.pc) else None,
            str(item.pcc) if not pd.isna(item.pcc) else None,
            str(item.ba) if not pd.isna(item.ba) else None,
            str(item.htn) if not pd.isna(item.htn) else None,
            str(item.dm) if not pd.isna(item.dm) else None,
            str(item.cad) if not pd.isna(item.cad) else None,
            str(item.appet) if not pd.isna(item.appet) else None,
            str(item.pe) if not pd.isna(item.pe) else None,
            str(item.ane) if not pd.isna(item.ane) else None,
            str(item.classification) if not pd.isna(item.classification) else None
        ) for item in data
    ]

    columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
               'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']
    processed_df = pd.DataFrame(transformed_data, columns=columns)
    return processed_df

def correlation(data):
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
        ) for item in data
    ], columns=['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc'])

    return df


def cat_count(data):
    processed_data = [
        (
            str(item.rbc) if not pd.isna(item.rbc) else None,
            str(item.pc) if not pd.isna(item.pc) else None,
            str(item.pcc) if not pd.isna(item.pcc) else None,
            str(item.ba) if not pd.isna(item.ba) else None,
            str(item.htn) if not pd.isna(item.htn) else None,
            str(item.dm) if not pd.isna(item.dm) else None,
            str(item.cad) if not pd.isna(item.cad) else None,
            str(item.appet) if not pd.isna(item.appet) else None,
            str(item.pe) if not pd.isna(item.pe) else None,
            str(item.ane) if not pd.isna(item.ane) else None,
            str(item.classification) if not pd.isna(item.classification) else None
        ) for item in data
    ]

    processed_df = pd.DataFrame(processed_data,
                                columns=['rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane',
                                         'classification'])

    return processed_df

def anylyze_missing(data):
    processed_data = [
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
            str(item.rbc) if not pd.isna(item.rbc) else None,
            str(item.pc) if not pd.isna(item.pc) else None,
            str(item.pcc) if not pd.isna(item.pcc) else None,
            str(item.ba) if not pd.isna(item.ba) else None,
            str(item.htn) if not pd.isna(item.htn) else None,
            str(item.dm) if not pd.isna(item.dm) else None,
            str(item.cad) if not pd.isna(item.cad) else None,
            str(item.appet) if not pd.isna(item.appet) else None,
            str(item.pe) if not pd.isna(item.pe) else None,
            str(item.ane) if not pd.isna(item.ane) else None,
            str(item.classification) if not pd.isna(item.classification) else None
        ) for item in data
    ]

    columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
               'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']
    processed_df = pd.DataFrame(processed_data, columns=columns)
    return processed_df


def data_stat(data):
    processed_data = [
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
            str(item.rbc) if not pd.isna(item.rbc) else None,
            str(item.pc) if not pd.isna(item.pc) else None,
            str(item.pcc) if not pd.isna(item.pcc) else None,
            str(item.ba) if not pd.isna(item.ba) else None,
            str(item.htn) if not pd.isna(item.htn) else None,
            str(item.dm) if not pd.isna(item.dm) else None,
            str(item.cad) if not pd.isna(item.cad) else None,
            str(item.appet) if not pd.isna(item.appet) else None,
            str(item.pe) if not pd.isna(item.pe) else None,
            str(item.ane) if not pd.isna(item.ane) else None,
            str(item.classification) if not pd.isna(item.classification) else None
        ) for item in data
    ]

    columns = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
               'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification']

    processed_df = pd.DataFrame(processed_data, columns=columns)
    return processed_df