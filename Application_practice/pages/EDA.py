import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO

# Function to plot different types of graphs based on data characteristics
# Function to plot different types of graphs based on data characteristics
def plot_graphs(dataframe):
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = dataframe.select_dtypes(include=['object']).columns.tolist()
    date_cols = dataframe.select_dtypes(include=['datetime']).columns.tolist()

    # Histograms and KDE plots for numeric columns
    for col in numeric_cols:
        plt.figure(figsize=(10, 4))
        sns.histplot(dataframe[col].dropna(), bins=30, kde=True)
        plt.title(f'Distribution of {col}')
        plt.xlabel(col)
        plt.ylabel('Frequency')
        st.pyplot(plt)
        plt.clf()

    # Count plots for categorical columns
    for col in categorical_cols:
        plt.figure(figsize=(10, 4))
        sns.countplot(x=dataframe[col].dropna(), data=dataframe)
        plt.title(f'Count of {col}')
        plt.xticks(rotation=45)
        plt.ylabel('Count')
        st.pyplot(plt)
        plt.clf()

    # Box plots for numeric columns
    for col in numeric_cols:
        plt.figure(figsize=(10, 4))
        # Drop NaN values before plotting
        sns.boxplot(x=dataframe[col].dropna())
        plt.title(f'Box Plot of {col}')
        plt.xlabel(col)
        st.pyplot(plt)
        plt.clf()

    # Pairplot for numeric columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(10, 10))
        sns.pairplot(dataframe[numeric_cols].dropna())  # Drop NaNs for pairplot
        st.pyplot(plt)
        plt.clf()

    # Correlation heatmap if there are enough numeric columns
    if len(numeric_cols) > 1:
        plt.figure(figsize=(12, 8))
        correlation_matrix = dataframe[numeric_cols].corr()
        sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
        plt.title('Correlation Heatmap')
        st.pyplot(plt)
        plt.clf()

    # Time series plot for date columns
    for col in date_cols:
        if not dataframe[col].isnull().all():  # Check if the column has non-null values
            plt.figure(figsize=(10, 4))
            counts = dataframe[col].value_counts().sort_index()
            if counts.size > 0:  # Ensure there are counts to plot
                counts.plot()
                plt.title(f'Time Series of {col}')
                plt.xlabel(col)
                plt.ylabel('Count')
                st.pyplot(plt)
                plt.clf()
            else:
                st.warning(f"No data to plot for {col}.")
        else:
            st.warning(f"All values are null in {col}, skipping the time series plot.")


# Streamlit application layout for MLyzer
st.title("MLyzer - Exploratory Data Analysis")

# File uploader
uploaded_file = st.file_uploader("Choose a file (CSV)", type="csv")

# Check if the DataFrame already exists in session state
if "dataframe" not in st.session_state:
    st.session_state.dataframe = None

if uploaded_file is not None:
    try:
        # Load dataset
        dataframe = pd.read_csv(uploaded_file)

        # Handle potential date columns
        for col in dataframe.columns:
            if pd.api.types.is_string_dtype(dataframe[col]):
                # Attempt to convert string columns to datetime
                dataframe[col] = pd.to_datetime(dataframe[col], errors='coerce')

        st.session_state.dataframe = dataframe  # Store the DataFrame in session state

        st.write("Dataset loaded successfully!")
        st.write(dataframe.head())

        # Exploratory Data Analysis (EDA)
        st.subheader("Exploratory Data Analysis (EDA)")
        st.write("Dataset Info:")
        buffer = StringIO()
        dataframe.info(buf=buffer)
        st.text(buffer.getvalue())

        st.write("Missing Values:")
        st.write(dataframe.isnull().sum())

        st.write("Basic Statistics:")
        st.write(dataframe.describe(include='all'))

        # Plotting distributions and counts for numeric and categorical columns
        st.subheader("Distribution and Count Plots")
        plot_graphs(dataframe)

    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
else:
    if st.session_state.dataframe is not None:
        st.write("Previously loaded dataset:")
        st.write(st.session_state.dataframe.head())
    else:
        st.info("Please upload a dataset to perform exploratory data analysis.")
