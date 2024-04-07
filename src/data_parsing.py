import pandas as pd
from sklearn.impute import SimpleImputer

def calculate_missing_data_info(data):
    # Count missing values in each column
    missing_values = data.isnull().sum()

    # Calculate the percentage of existing and missing data for each feature
    total_rows = data.shape[0]

    # Create an empty DataFrame with desired columns and data types
    missing_data_info = pd.DataFrame(columns=['Feature', 'Type', 'Existing (%)', 'Missing (%)', 'Existing Count', 'Missing Count'], dtype=object)

    for column in data.columns:
        total_data = total_rows - missing_values[column]
        percent_existing = (total_data / total_rows) * 100 // 1
        percent_missing = (missing_values[column] / total_rows) * 100 // 1
        data_type = data[column].dtype
        missing_data_info = pd.concat([missing_data_info, 
                                       pd.DataFrame({'Feature': [column],
                                                     'Type': [data_type],
                                                     'Existing (%)': [percent_existing],
                                                     'Missing (%)': [percent_missing],
                                                     'Existing Count': [total_data],
                                                     'Missing Count': [missing_values[column]]})], 
                                       ignore_index=True)
    return missing_data_info

def replace_missing_categorical(data, missing_data_info):
    """
    Replace NaN with "unrecorded" for categorical features with missing values.

    Args:
        data (DataFrame): The DataFrame containing the data.
        missing_data_info (DataFrame): DataFrame containing information about missing data.

    Returns:
        DataFrame: The DataFrame with missing values replaced for categorical features.
    """
    # Extract categorical features with missing values
    categorical_features_missing = missing_data_info[
        (missing_data_info['Missing Count'] > 0) &
        (missing_data_info['Type'] == 'object')
    ]['Feature'].tolist()
    
    # Replace missing values with "unrecorded"
    data[categorical_features_missing] = data[categorical_features_missing].fillna("unrecorded")
    
    return data

def drop_missing_numeric_features(data, missing_data_info, columns_to_drop):
    """
    Drop specified columns if they have missing values and exist in the DataFrame.

    Args:
        data (DataFrame): The DataFrame containing the data.
        missing_data_info (DataFrame): DataFrame containing information about missing data.
        columns_to_drop (list): List of column names to be dropped if they have missing values.

    Returns:
        DataFrame: The DataFrame with specified columns dropped if they have missing values.
    """
    # Extract numeric features with missing values
    numeric_features_missing = missing_data_info[
        (missing_data_info['Missing Count'] > 0) &
        (missing_data_info['Type'] != 'object')
    ]['Feature'].tolist()
    
    # Drop specified columns if they exist and have missing values
    for column in columns_to_drop:
        if column in numeric_features_missing and column in data.columns:
            data.drop(columns=[column], inplace=True)
        elif column in numeric_features_missing and column not in data.columns:
            print(f"Warning: '{column}' not found in DataFrame.")
    
    return data

def impute_missing_values(data, numeric_features_to_impute):
    """
    Impute missing values in specified numeric features using mean imputation.

    Args:
        data (DataFrame): The DataFrame containing the data.
        numeric_features_to_impute (list): List of column names with missing values to be imputed.

    Returns:
        DataFrame: The DataFrame with missing values imputed using mean imputation.
    """
    # Initialize the imputer
    imputer = SimpleImputer(strategy='mean')
    
    # Impute missing values in specified numeric features
    data[numeric_features_to_impute] = imputer.fit_transform(data[numeric_features_to_impute])
    
    return data