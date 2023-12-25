from flask import jsonify
from scipy import stats
from SRC.logics_regix import *
from models import  ImportDataDetail, db
from flask import Blueprint
from flask import request
import pandas as pd
from collections import Counter
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler

eda = Blueprint("eda", __name__, url_prefix="/api/v1/eda")
@eda.route('/transposeNum', methods=['GET'])
def transpose_num():
    id = request.args.get('id')
    if not id:
        return jsonify({'message': 'Please provide an ID parameter'}), 400
    data = ImportDataDetail.query.filter_by(importId=id).all()
    if not list:
        return jsonify({'message': 'No data found for the provided ID'}), 404
    cdk_df = pd.DataFrame(transposeNum(data))


    description = cdk_df.describe(include='all').to_dict()
    custom_description = pd.DataFrame({
        'Count': round(cdk_df.count(), 3),
        'Missing': round(len(cdk_df) - cdk_df.count(), 3),
        'Mean': round(cdk_df.mean(), 3),
        'Std': round(cdk_df.std(), 3),
        'Variance': round(cdk_df.var(), 3),
        'Min': round(cdk_df.min(), 3),
        '25%': round(cdk_df.quantile(0.25), 3),
        '50%': round(cdk_df.quantile(0.5), 3),
        '75%': round(cdk_df.quantile(0.75), 3),
        'Max': round(cdk_df.max(), 3)
    })
    custom_description_dict = custom_description.T.to_dict()
    description.update(custom_description_dict)
    return jsonify(description), 200


@eda.route('/transposeCat', methods=['GET'])
def transpose_cat():
    id = request.args.get('id')
    if not id:
        return jsonify({'message': 'Please provide an ID parameter'}), 400
    data = ImportDataDetail.query.filter_by(importId=id).all()
    if not list:
        return jsonify({'message': 'No data found for the provided ID'}), 404

    cdk_cat = pd.DataFrame(transposeCat(data))
    description = cdk_cat.describe(include='all').T.to_dict()
    return jsonify(description), 200

@eda.route('/findNull', methods=['GET'])
def find_null():
    id = request.args.get('id')
    if not id:
        return jsonify({'message': 'Please provide an ID parameter'}), 400
    data = ImportDataDetail.query.filter_by(importId=id).all()

    if not data:  # Checking if data is empty
        return jsonify({'message': 'No data found for the provided ID'}), 404
    df = pd.DataFrame(findNull(data))
    missing_values = df.isnull().sum()  # Calculate missing values
    result = missing_values.to_dict()  # Convert Series to dictionary

    # Convert the dictionary keys to strings if they are integers
    result = {str(key): value for key, value in result.items()}

    return jsonify(result), 200


@eda.route('/heatmapVal', methods=['GET'])
def find_heatmapVal():
    id = request.args.get('id')
    if not id:
        return jsonify({'message': 'Please provide an ID parameter'}), 400
    data = ImportDataDetail.query.filter_by(importId=id).all()
    if not data:  # Checking if data is empty
        return jsonify({'message': 'No data found for the provided ID'}), 404
    df = pd.DataFrame(create_dataframe_heat(data))
    numeric_features = df.select_dtypes(include=['float64', 'int64'])
    corr = numeric_features.corr()
    correlation_values = []
    xcolumn = []
    ycolumn = []

    for column in corr.columns:
        for index, value in corr[column].items():
            correlation_values.append({'x': column, 'y': index, 'value': round(value, 1)})
            if column not in xcolumn:
                xcolumn.append(column)
            if index not in ycolumn:
                ycolumn.append(index)

    return jsonify({'xcolumn': xcolumn, 'ycolumn': ycolumn, 'correlation_values': correlation_values}), 200


@eda.route('/outlierval', methods=['GET'])
def outlier_values():
    id = request.args.get('id')
    if not id:
        return jsonify({'message': 'Please provide an ID parameter'}), 400

    # Fetch data from the database
    data = ImportDataDetail.query.filter_by(importId=id).all()
    if not data:
        return jsonify({'message': 'No data found for the provided ID'}), 404

    # Extract numeric values from the dataset
    data_values = []
    for item in data:
        row = (
            get_numeric_value(item.age),
            get_numeric_value(item.bp),
            get_numeric_value(item.sg),
            get_numeric_value(item.al),
            get_numeric_value(item.su),
            get_numeric_value(item.bgr),
            get_numeric_value(item.bu),
            get_numeric_value(item.sc),
            get_numeric_value(item.sod),
            get_numeric_value(item.pot),
            get_numeric_value(item.hemo),
            get_numeric_value(item.pcv),
            get_numeric_value(item.wc),
            get_numeric_value(item.rc),
        )
        data_values.append(row)

    # Create DataFrame using the extracted numeric values
    cdk_df = pd.DataFrame(data_values, columns=[
        'age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc'
    ])

    # Select numeric features for outlier detection
    numeric_features = cdk_df.select_dtypes(include=['float64', 'int64'])

    # Calculate outliers for each numeric feature
    outlier_values = {}
    for column in numeric_features.columns:
        # Calculate bounds based on median instead of mean (more robust to outliers)
        median = cdk_df[column].median()
        mad = cdk_df[column].sub(median).abs().median() * 1.4826
        lower_bound = median - 3 * mad
        upper_bound = median + 3 * mad
        outliers = cdk_df[(cdk_df[column] < lower_bound) | (cdk_df[column] > upper_bound)][column].tolist()
        outlier_values[column] = outliers

    return jsonify(outlier_values), 200

@eda.route('/pairPlot', methods=['GET'])
def compute_correlation():
    id = request.args.get('id')
    if not id:
        return jsonify({'message': 'Please provide an ID parameter'}), 400

    # Fetch data from the database based on the provided ID
    data = ImportDataDetail.query.filter_by(importId=id).all()
    if not data:
        return jsonify({'message': 'No data found for the provided ID'}), 404

    # Convert the data to a DataFrame
    cdk_df = pd.DataFrame(correlation(data))

    # Calculate the correlation matrix for the numeric features
    numeric_features = cdk_df.select_dtypes(include=['float64', 'int64'])
    correlation_matrix = round(numeric_features.corr(),3)

    # Convert the correlation matrix to a dictionary for JSON serialization
    correlation_dict = correlation_matrix.to_dict()

    return jsonify(correlation_dict), 200

@eda.route('/categorical', methods=['GET'])
def generate_categorical_count_plots():
    id = request.args.get('id')
    if not id:
        return jsonify({'message': 'Please provide an ID parameter'}), 400

    # Fetch data from the database based on the provided ID
    data = ImportDataDetail.query.filter_by(importId=id).all()
    if not data:
        return jsonify({'message': 'No data found for the provided ID'}), 404

    # Convert the data to a DataFrame
    cdk_cat = pd.DataFrame(cat_count(data))

    # Select categorical features for count plots
    #categorical_features = cdk_cat.select_dtypes(include=['object']).drop('classification', axis=1)

    # Generate count plots for all categorical features
    count_plots = {}
    for column in cdk_cat.columns:
        counts = cdk_cat[column].value_counts().to_dict()
        count_plots[column] = counts

    return jsonify(count_plots), 200


@eda.route('/missing_data', methods=['GET'])
def analyze_missing_data():
    id = request.args.get('id')
    if not id:
        return jsonify({'message': 'Please provide an ID parameter'}), 400

    # Fetch data from the database based on the provided ID
    data = ImportDataDetail.query.filter_by(importId=id).all()
    if not data:
        return jsonify({'message': 'No data found for the provided ID'}), 404

    # Convert the data to a DataFrame
    df = pd.DataFrame(anylyze_missing(data))

    # Calculate the percentage of missing values for each feature
    missing_percentage = round((df.isnull().mean() * 100), 3)

    # Convert the missing percentage to a dictionary
    headers_array = missing_percentage.index.tolist()
    percentages_array = missing_percentage.values.tolist()

    # Create a dictionary from the separate arrays
    missing_percentage_dict = dict(zip(headers_array, percentages_array))
    #missing_percentage_dict = [headers_array, percentages_array]
    return jsonify(missing_percentage_dict), 200

@eda.route('/dataset_statistics', methods=['GET'])
def get_dataset_statistics():
    id = request.args.get('id')
    if not id:
        return jsonify({'message': 'Please provide an ID parameter'}), 400

    # Fetch data from the database based on the provided ID
    data = ImportDataDetail.query.filter_by(importId=id).all()
    if not data:
        return jsonify({'message': 'No data found for the provided ID'}), 404

    # Convert the data to a DataFrame
    df = pd.DataFrame(data_stat(data))

    # Calculate dataset statistics
    stats = calculate_statistics(df)

    return jsonify(stats), 200



@eda.route('/result', methods=['GET'])
def column_wise_eda():
    id = request.args.get('id')
    column_name = request.args.get('column_name')
    if not id:
        return jsonify({'message': 'Please provide an ID parameter'}), 400
    if not column_name:
        return jsonify({'message': 'Please provide a column_name parameter'}), 400
    data = ImportDataDetail.query.filter_by(importId=id).all()
    if not data:
        return jsonify({'message': 'No data found for the provided ID'}), 404
    df = pd.DataFrame([
        (
            int(float(item.age)) if not pd.isna(item.age) else None,
            float(item.bp) if not pd.isna(item.bp) else None,
            float(item.sg) if not pd.isna(item.sg) else None,
            float(item.al) if not pd.isna(item.al) else None,
            float(item.su) if not pd.isna(item.su) else None,
            float(item.bgr) if not pd.isna(item.bgr) else None,
            float(item.bu) if not pd.isna(item.bu) else None,
            float(item.sc) if not pd.isna(item.sc) else None,
            float(item.sod) if not pd.isna(item.sod) else None,
            float(item.pot) if not pd.isna(item.pot) else None,
            float(item.hemo) if not pd.isna(item.hemo) else None,
            float(item.pcv) if not pd.isna(item.pcv) else None,
            float(item.wc) if not pd.isna(item.wc) else None,
            float(item.rc) if not pd.isna(item.rc) else None,
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
        ) for item in data],
        columns=['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
                 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification'])
    if column_name not in df.columns:
        return jsonify({'message': 'Invalid column name'}), 400
    column_data = df[column_name].dropna()
    total_values = len(column_data)
    stats_list = {}
    distinct_values = column_data.unique()
    missing_values = total_values - column_data.count()
    if np.issubdtype(column_data.dtype, np.number):
        stats_list['Distinct'] = len(distinct_values)
        stats_list['Distinct (%)'] = len(distinct_values) / total_values * 100 if total_values > 0 else 0
        # Missing values
        stats_list['Missing'] = int(missing_values)
        stats_list['Missing (%)'] = missing_values / int(total_values) * 100 if total_values > 0 else 0
        # Other statistics
        stats_list['Mean'] = round(np.mean(column_data),3)
        stats_list['Median'] = round(np.median(column_data),3)
        stats_list['Mode'] = round(float(stats.mode(column_data)[0]),3)
        stats_list['SD'] = round(np.std(column_data),3)
        stats_list['Variance'] = round(np.var(column_data),3)
        stats_list['Minimum'] = round(int(np.min(column_data)),3)
        stats_list['Maximum'] = round(int(np.max(column_data)),3)
        stats_list['Zeros'] = round(int((column_data == 0).sum()),3)
        stats_list['Memory size'] = column_data.memory_usage(deep=True)
        stats_list['type'] = "num"
    else:
        stats_list['Distinct'] = round(len(distinct_values),3)
        stats_list['Distinct (%)'] = round(len(distinct_values) / total_values * 100, 3) if total_values > 0 else 0
        stats_list['Missing'] = round(int(missing_values), 3)
        stats_list['Missing (%)'] = round(missing_values / int(total_values) * 100, 3) if total_values > 0 else 0
        stats_list['Memory size'] = column_data.memory_usage(deep=True)
        stats_list['type'] = "cat"

    histogram_data = Counter(column_data.dropna().values)
    histogram = [{'value': str(key), 'count': value} for key, value in histogram_data.items()]

    return jsonify({'statistics': stats_list, 'histogram': histogram}), 200


@eda.route('/error', methods=['GET'])
def column_error():
    id = request.args.get('id')
    column_name_1 = request.args.get('column_1')  # Replace 'column_1' with the actual name parameter
    column_name_2 = request.args.get('column_2')
    if not id or not column_name_1 or not column_name_2:
        return jsonify({'message': 'Please provide ID, column_name_1, and column_name_2 parameters'}), 400
    data = ImportDataDetail.query.filter_by(importId=id).all()
    if not data:
        return jsonify({'message': 'No data found for the provided ID'}), 404
    df = pd.DataFrame([
        (
            int(float(item.age)) if not pd.isna(item.age) else None,
            float(item.bp) if not pd.isna(item.bp) else None,
            float(item.sg) if not pd.isna(item.sg) else None,
            float(item.al) if not pd.isna(item.al) else None,
            float(item.su) if not pd.isna(item.su) else None,
            float(item.bgr) if not pd.isna(item.bgr) else None,
            float(item.bu) if not pd.isna(item.bu) else None,
            float(item.sc) if not pd.isna(item.sc) else None,
            float(item.sod) if not pd.isna(item.sod) else None,
            float(item.pot) if not pd.isna(item.pot) else None,
            float(item.hemo) if not pd.isna(item.hemo) else None,
            float(item.pcv) if not pd.isna(item.pcv) else None,
            float(item.wc) if not pd.isna(item.wc) else None,
            float(item.rc) if not pd.isna(item.rc) else None,
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
        ) for item in data],
        columns=['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
                 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification'])
    if column_name_1 not in df.columns or column_name_2 not in df.columns:
        return jsonify({'message': 'Invalid column names provided'}), 400

    if np.issubdtype(df[column_name_1].dtype, np.number) and np.issubdtype(df[column_name_2].dtype, np.number):
        # Drop NaNs and convert columns to float
        column_1_data = df[column_name_1].dropna().astype(float)
        column_2_data = df[column_name_2].dropna().astype(float)

        # Ensure both columns have the same number of samples
        common_index = column_1_data.index.intersection(column_2_data.index)
        column_1_data = column_1_data.loc[common_index]
        column_2_data = column_2_data.loc[common_index]
        correlation_value = column_1_data.corr(column_2_data)
        r2 = r2_score(column_1_data, column_2_data)
        mae = mean_absolute_error(column_1_data, column_2_data)
        rmse = np.sqrt(mean_squared_error(column_1_data, column_2_data))
        heatmap_data = pd.DataFrame({column_name_1: column_1_data, column_name_2: column_2_data})
        stats_list = {
            'Correlation': correlation_value,
            'R2': r2,
            'MAE': mae,
            'RMSE': rmse,
            'Heatmap Data': heatmap_data.to_dict(orient='list')
        }
        return jsonify(stats_list), 200

    else:
        return jsonify({'message': 'Required Numeric Columns'}), 500

@eda.route('/ratio', methods=['GET'])
def get_dataset_ratio():
    id = request.args.get('id')
    column_name_1 = request.args.get('column_1')  # Replace 'column_1' with the actual name parameter
    column_name_2 = request.args.get('column_2')
    if not id or not column_name_1 or not column_name_2:
        return jsonify({'message': 'Please provide ID, column_name_1, and column_name_2 parameters'}), 400
    data = ImportDataDetail.query.filter_by(importId=id).all()
    if not data:
        return jsonify({'message': 'No data found for the provided ID'}), 404
    df = pd.DataFrame([
        (
            int(float(item.age)) if not pd.isna(item.age) else None,
            float(item.bp) if not pd.isna(item.bp) else None,
            float(item.sg) if not pd.isna(item.sg) else None,
            float(item.al) if not pd.isna(item.al) else None,
            float(item.su) if not pd.isna(item.su) else None,
            float(item.bgr) if not pd.isna(item.bgr) else None,
            float(item.bu) if not pd.isna(item.bu) else None,
            float(item.sc) if not pd.isna(item.sc) else None,
            float(item.sod) if not pd.isna(item.sod) else None,
            float(item.pot) if not pd.isna(item.pot) else None,
            float(item.hemo) if not pd.isna(item.hemo) else None,
            float(item.pcv) if not pd.isna(item.pcv) else None,
            float(item.wc) if not pd.isna(item.wc) else None,
            float(item.rc) if not pd.isna(item.rc) else None,
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
        ) for item in data],
        columns=['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
                 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification'])
    if column_name_1 not in df.columns or column_name_2 not in df.columns:
        return jsonify({'message': 'Invalid column names provided'}), 400

    if np.issubdtype(df[column_name_1].dtype, np.number) and np.issubdtype(df[column_name_2].dtype, np.number):
        column_1_data = df[column_name_1].dropna().astype(float)
        column_2_data = df[column_name_2].dropna().astype(float)

        common_index = column_1_data.index.intersection(column_2_data.index)
        column_1_data = column_1_data.loc[common_index]
        column_2_data = column_2_data.loc[common_index]
        small_constant = 1e-5
        df[column_name_1 + '_to_' + column_name_2 + '_ratio'] = column_1_data / (column_2_data + small_constant)

        # Prepare JSON-serializable data
        #statistics = df.describe().to_dict()
        selected_columns = df[
            [column_name_1, column_name_2, column_name_1 + '_to_' + column_name_2 + '_ratio']].head().to_dict()
        selected_columns = selected_columns
        return jsonify({'selected_columns': selected_columns}), 200
    else:
        return jsonify({'message': 'Required Numeric Columns'}), 500



@eda.route('/allRatio', methods=['GET'])
def get_dataset_allratio():
    id = request.args.get('id')
    if not id:
        return jsonify({'message': 'Please provide ID parameter'}), 400
    data = ImportDataDetail.query.filter_by(importId=id).all()
    if not data:
        return jsonify({'message': 'No data found for the provided ID'}), 404
    df = pd.DataFrame([
        (
            int(float(item.age)) if not pd.isna(item.age) else None,
            float(item.bp) if not pd.isna(item.bp) else None,
            float(item.sg) if not pd.isna(item.sg) else None,
            float(item.al) if not pd.isna(item.al) else None,
            float(item.su) if not pd.isna(item.su) else None,
            float(item.bgr) if not pd.isna(item.bgr) else None,
            float(item.bu) if not pd.isna(item.bu) else None,
            float(item.sc) if not pd.isna(item.sc) else None,
            float(item.sod) if not pd.isna(item.sod) else None,
            float(item.pot) if not pd.isna(item.pot) else None,
            float(item.hemo) if not pd.isna(item.hemo) else None,
            float(item.pcv) if not pd.isna(item.pcv) else None,
            float(item.wc) if not pd.isna(item.wc) else None,
            float(item.rc) if not pd.isna(item.rc) else None,
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
        ) for item in data],
        columns=['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc',
                 'rbc', 'pc', 'pcc', 'ba', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane', 'classification'])

    numeric_features = df.select_dtypes(include=['float64', 'int64'])

    scaler = MinMaxScaler()
    scaled_numeric_features = scaler.fit_transform(round(numeric_features, 3))
    scaled_numeric_df = pd.DataFrame(scaled_numeric_features, columns=numeric_features.columns)
    scaled_numeric_df = scaled_numeric_df.round(3)
    columns_to_fillna = ['age', 'bp', 'sg', 'al', 'su', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 'pcv', 'wc', 'rc']
    scaled_numeric_df[columns_to_fillna] = scaled_numeric_df[columns_to_fillna].fillna('NaN')
    return jsonify(scaled_numeric_df.head().to_dict()), 200