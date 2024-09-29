import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from streamlit_extras.add_vertical_space import add_vertical_space
from io import StringIO

# Custom Styles
st.markdown("""
    <style>
    .main-title {
        font-size: 48px;
        font-weight: bold;
        color: #004aad;
        text-align: center;
    }
    .sub-title {
        font-size: 24px;
        color: #555;
        margin-bottom: 30px;
    }
    .about-section, .feature-section, .contact-section {
        background-color: #f9f9f9;
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.1);
        margin-top: 30px;
    }
    .contact-section {
        background-color: #eef7fa;
    }
    .feature-heading {
        color: #004aad;
        font-size: 20px;
        font-weight: bold;
        margin-top: 15px;
    }
    .feature-text {
        font-size: 16px;
        margin-bottom: 15px;
    }
    .header-line {
        height: 3px;
        background-color: #004aad;
        margin-bottom: 20px;
    }
    .contact-details {
        font-size: 16px;
        margin-top: 20px;
    }
    .contact-details strong {
        color: #004aad;
    }
    </style>
    """, unsafe_allow_html=True)

# Title Section
st.markdown('<h1 class="main-title">FinSight AI</h1>', unsafe_allow_html=True)

# Subheader to Upload the Dataset
st.subheader("Upload and Analyze Your Financial Data")
uploaded_file = st.file_uploader("Choose a CSV file", type=["csv"])

# Check if a file has been uploaded
if uploaded_file is not None:
    try:
        # Read the CSV file into a DataFrame
        dataframe = pd.read_csv(uploaded_file)

        # Convert date columns to datetime, if necessary
        for column in dataframe.columns:
            if dataframe[column].dtype == 'object':  # Check for object type (strings)
                try:
                    dataframe[column] = pd.to_datetime(column, errors='ignore')  # Attempt conversion
                except ValueError:
                    pass  # If conversion fails, leave it as is

        # Create a form for the input box and submit button
        with st.form("my_form"):
            head_val = st.text_input("Number of rows you want to display")
            submit_button = st.form_submit_button("Submit")

            if submit_button:
                if head_val:
                    try:
                        # Convert input to integer
                        num_rows = int(head_val)
                        st.write(dataframe.head(num_rows))
                    except ValueError:
                        st.error("Please enter a valid integer.")
                else:
                    st.warning("Please enter a number of rows to display.")

        option = st.selectbox(
            'Select basic analysis criteria',
            ('None', 'Describe', 'Info', 'Correlation')
        )
        st.write('You selected:', option)

        # Perform analysis based on selected option
        if option == 'Describe':
            st.write(dataframe.describe())
        elif option == 'None':
            pass
        elif option == 'Info':
            buffer = StringIO()  # Create a buffer to capture the output
            dataframe.info(buf=buffer)  # Pass the buffer to the info() method
            s = buffer.getvalue()  # Get the value from the buffer
            st.text(s)
        else:  # Correlation
            # Filter only numeric columns for correlation
            numeric_df = dataframe.select_dtypes(include=['float64', 'int64'])
            if not numeric_df.empty:
                st.write(numeric_df.corr())
            else:
                st.warning("No numeric columns available for correlation.")
    except Exception as e:
        st.error(f"An error occurred while reading the file: {e}")
else:
    st.info("Please upload a dataset to analyze.")

# Adding Vertical Space Correctly
add_vertical_space(3)

# About Section
st.markdown('<div class="about-section">', unsafe_allow_html=True)
st.header("About FinSight AI")
st.markdown("""
**FinSight AI** is a powerful, intuitive tool designed to help financial analysts, data scientists, and businesses make data-driven decisions. With a focus on efficiency and ease of use, our software allows users to seamlessly upload, explore, and derive insights from financial datasets.

We offer a comprehensive set of features, from basic data analysis to advanced machine learning, aimed at streamlining financial analytics workflows.
""")
st.markdown('</div>', unsafe_allow_html=True)

# Features Section
st.markdown('<div class="feature-section">', unsafe_allow_html=True)
st.header("Key Features")
st.markdown('<div class="header-line"></div>', unsafe_allow_html=True)

st.markdown("""
- <span class="feature-heading">Exploratory Data Analysis (EDA):</span>  
Perform basic data exploration like displaying the head of the dataset, descriptive statistics, and analyzing data correlations. Use visualizations to discover patterns in your data.

- <span class="feature-heading">Feature Engineering:</span>  
Generate or modify features to improve predictive model performance. Includes transformations like encoding categorical variables, normalization, and polynomial features.

- <span class="feature-heading">Fin-GPT (Financial GPT):</span>  
An AI-powered chatbot interface designed for querying financial datasets. Powered by advanced language models, **Fin-GPT** can respond to user queries and extract trends or insights from the data in a conversational format.

- <span class="feature-heading">Model Building and Evaluation:</span>  
Develop machine learning models on your data and evaluate their performance using accuracy, precision, recall, F1-score, and more. Optimize models through cross-validation and hyperparameter tuning.
""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

# Contact Section
st.markdown('<div class="contact-section">', unsafe_allow_html=True)
st.header("Contact Us")
st.markdown('<div class="header-line"></div>', unsafe_allow_html=True)

st.markdown("""
For any queries or support, feel free to contact us:

<div class="contact-details">
- <strong>Email:</strong> vishnunpoojary34@gmail.com  
- <strong>Phone:</strong> +91 8317422461  
- <strong>Website:</strong> [www.finsightai.com]
</div>

We are available to assist you Monday through Friday, 9 AM to 5 PM (EST).
""", unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
