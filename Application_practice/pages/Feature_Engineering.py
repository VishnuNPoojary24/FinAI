import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler

# Function to handle missing values
def handle_missing_values(dataframe, strategy):
    if strategy == 'Remove':
        original_shape = dataframe.shape
        dataframe.dropna(inplace=True)
        return dataframe, f"Removed rows with missing values. Original shape: {original_shape}, New shape: {dataframe.shape}"
    elif strategy == 'Mean':
        original_shape = dataframe.shape
        for column in dataframe.select_dtypes(include=[float, int]).columns:
            dataframe[column].fillna(dataframe[column].mean(), inplace=True)
        return dataframe, f"Filled missing values with mean. Original shape: {original_shape}, New shape: {dataframe.shape}"
    elif strategy == 'Median':
        original_shape = dataframe.shape
        for column in dataframe.select_dtypes(include=[float, int]).columns:
            dataframe[column].fillna(dataframe[column].median(), inplace=True)
        return dataframe, f"Filled missing values with median. Original shape: {original_shape}, New shape: {dataframe.shape}"
    else:
        return dataframe, "No changes made."

# Function to encode categorical variables
def encode_categorical_variables(dataframe, method):
    if method == 'Label Encoding':
        for column in dataframe.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            dataframe[column] = le.fit_transform(dataframe[column])
        return dataframe, "Categorical variables encoded using Label Encoding."
    elif method == 'One-Hot Encoding':
        dataframe = pd.get_dummies(dataframe, drop_first=True)
        return dataframe, "Categorical variables encoded using One-Hot Encoding."
    return dataframe, "No changes made."

# Function to scale numerical features
def scale_numerical_features(dataframe, method):
    numeric_cols = dataframe.select_dtypes(include=[float, int]).columns.tolist()
    if method == 'Standard':
        scaler = StandardScaler()
        dataframe[numeric_cols] = scaler.fit_transform(dataframe[numeric_cols])
        return dataframe, "Numerical features scaled using StandardScaler."
    elif method == 'MinMax':
        scaler = MinMaxScaler()
        dataframe[numeric_cols] = scaler.fit_transform(dataframe[numeric_cols])
        return dataframe, "Numerical features scaled using MinMaxScaler."
    return dataframe, "No scaling applied."

# Function to detect and remove outliers
def remove_outliers(dataframe, method='IQR'):
    numeric_cols = dataframe.select_dtypes(include=[float, int]).columns.tolist()
    if method == 'IQR':
        for column in numeric_cols:
            Q1 = dataframe[column].quantile(0.25)
            Q3 = dataframe[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            dataframe = dataframe[(dataframe[column] >= lower_bound) & (dataframe[column] <= upper_bound)]
        return dataframe, "Outliers removed using IQR method."
    return dataframe, "No changes made."

# Function to apply transformations to numerical features
def transform_numerical_features(dataframe, transformation):
    numeric_cols = dataframe.select_dtypes(include=[float, int]).columns.tolist()
    if transformation == 'Log':
        for column in numeric_cols:
            dataframe[column] = np.log1p(dataframe[column])  # Using log1p to handle zeros
        return dataframe, "Log transformation applied."
    elif transformation == 'Square Root':
        for column in numeric_cols:
            dataframe[column] = np.sqrt(dataframe[column])
        return dataframe, "Square root transformation applied."
    return dataframe, "No changes made."

# Streamlit application layout for feature engineering
st.title("Dynamic Feature Engineering - Data Cleaning and Preprocessing")

# Initialize session state for the dataframe if it doesn't exist
if 'dataframe' not in st.session_state:
    st.session_state.dataframe = None

# File uploader
uploaded_file = st.file_uploader("Choose a file (CSV)", type="csv")

if uploaded_file is not None:
    try:
        # Load dataset
        st.session_state.dataframe = pd.read_csv(uploaded_file)
        st.write("Dataset loaded successfully!")
        st.write(st.session_state.dataframe.head())

    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")

# Check if dataframe exists in session state
if st.session_state.dataframe is not None:
    # Analyze dataset characteristics
    st.subheader("Dataset Analysis")
    missing_values = st.session_state.dataframe.isnull().sum()
    st.write("Missing Values in each column:")
    st.write(missing_values)

    categorical_columns = st.session_state.dataframe.select_dtypes(include=['object']).columns.tolist()
    numerical_columns = st.session_state.dataframe.select_dtypes(include=[float, int]).columns.tolist()

    st.write("Categorical Columns:", categorical_columns)
    st.write("Numerical Columns:", numerical_columns)

    # Data Cleaning and Preprocessing
    st.subheader("Data Cleaning and Preprocessing")

    # Handle missing values
    if missing_values.sum() > 0:
        missing_value_strategy = st.selectbox(
            "Select missing value handling strategy:",
            ("None", "Remove", "Mean", "Median")
        )
        if missing_value_strategy != "None":
            st.session_state.dataframe, message = handle_missing_values(st.session_state.dataframe, missing_value_strategy)
            st.success(message)

    # Encode categorical variables
    if categorical_columns:
        encoding_method = st.selectbox(
            "Select encoding method for categorical variables:",
            ("None", "Label Encoding", "One-Hot Encoding")
        )
        if encoding_method != "None":
            st.session_state.dataframe, message = encode_categorical_variables(st.session_state.dataframe, encoding_method)
            st.success(message)

    # Scale numerical features
    if numerical_columns:
        scaling_method = st.selectbox(
            "Select scaling method:",
            ("None", "Standard", "MinMax")
        )
        if scaling_method != "None":
            st.session_state.dataframe, message = scale_numerical_features(st.session_state.dataframe, scaling_method)
            st.success(message)

    # Outlier removal
    if numerical_columns:
        outlier_strategy = st.selectbox(
            "Select outlier handling strategy:",
            ("None", "Remove using IQR")
        )
        if outlier_strategy != "None":
            st.session_state.dataframe, message = remove_outliers(st.session_state.dataframe, outlier_strategy)
            st.success(message)

    # Data transformation
    if numerical_columns:
        transformation_method = st.selectbox(
            "Select transformation for numerical features:",
            ("None", "Log", "Square Root")
        )
        if transformation_method != "None":
            st.session_state.dataframe, message = transform_numerical_features(st.session_state.dataframe, transformation_method)
            st.success(message)

    # Show the cleaned and processed dataframe
    st.subheader("Processed Dataset")
    st.write(st.session_state.dataframe.head())

    # Option to download cleaned data
    st.download_button(
        label="Download Cleaned Data",
        data=st.session_state.dataframe.to_csv(index=False).encode('utf-8'),
        file_name='cleaned_data.csv',
        mime='text/csv'
    )
else:
    st.info("Please upload a dataset to perform feature engineering.")
