import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import (RandomForestClassifier, GradientBoostingClassifier, 
                              AdaBoostClassifier, VotingClassifier, 
                              RandomForestRegressor, GradientBoostingRegressor, AdaBoostRegressor)
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                             mean_squared_error, r2_score)
import matplotlib.pyplot as plt
import seaborn as sns
from streamlit_extras.switch_page_button import switch_page

# Function to display confusion matrix
def plot_confusion_matrix(cm):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    st.pyplot(plt)

# Function to suggest models based on data characteristics
def suggest_models(dataframe, target_variable):
    numeric_cols = dataframe.select_dtypes(include=[np.number]).columns.tolist()
    
    if len(numeric_cols) < 2:
        return ["Consider having at least two numeric features for training models."]
    
    target_type = "classification" if (isinstance(dataframe[target_variable].dtype, pd.CategoricalDtype) or 
                        len(dataframe[target_variable].unique()) < 20) else "regression"
    
    if target_type == "classification":
        return ["Random Forest", "Logistic Regression", "Support Vector Machine",
                "Gradient Boosting", "AdaBoost", "K-Nearest Neighbors", "Voting Classifier"]
    else:
        return ["Random Forest Regressor", "Linear Regression", "Support Vector Regressor",
                "Gradient Boosting Regressor", "AdaBoost Regressor", "K-Nearest Neighbors Regressor"]

# Streamlit application layout for Model Training and Evaluation
st.title("MLyzer - Model Training and Evaluation")

# Load dataset (assuming it's already uploaded)
uploaded_file = st.file_uploader("Choose a file (CSV)", type="csv")

if uploaded_file is not None:
    try:
        dataframe = pd.read_csv(uploaded_file)

        # Check if the dataset is preprocessed
        preprocessed = st.radio("Is the dataset preprocessed?", ("Yes", "No"))
        
        if preprocessed == "No":
            st.warning("Please preprocess the dataset before training the model.")
            # Button to redirect to Feature Engineering page using switch_page
            if st.button("Go to Feature Engineering Page"):
                switch_page("Feature Engineering")  # Switch to feature engineering page
        else:
            # Proceed with model training and evaluation
            st.subheader("Select Features and Target Variable")
            target_variable = st.selectbox("Choose the target variable", dataframe.columns.tolist())
            feature_columns = st.multiselect("Select feature columns", dataframe.columns.tolist(), default=dataframe.columns.tolist())
            
            if target_variable in feature_columns:
                st.warning("Target variable should not be included in feature columns.")
            else:
                # Display model suggestions
                st.subheader("Recommended Machine Learning Models")
                model_recommendations = suggest_models(dataframe, target_variable)
                st.write(model_recommendations)

                # Dropdown to choose model for training
                model_choice = st.selectbox("Choose a model for training", model_recommendations)

                # Check if valid model recommendations were returned
                if model_choice != "Consider having at least two numeric features for training models.":

                    # Train and evaluate model
                    if st.button("Train Model"):
                        X = dataframe[feature_columns]
                        y = dataframe[target_variable]
                        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

                        # Determine if the target variable is classification or regression
                        target_type = "classification" if (isinstance(y.dtype, pd.CategoricalDtype) or 
                                        len(y.unique()) < 20) else "regression"

                        # Initialize the chosen model
                        model = None  # Initialize model variable

                        # Classifier models
                        if model_choice == "Random Forest":
                            model = RandomForestClassifier()
                        elif model_choice == "Logistic Regression":
                            model = LogisticRegression(max_iter=1000)
                        elif model_choice == "Support Vector Machine":
                            model = SVC()
                        elif model_choice == "Gradient Boosting":
                            model = GradientBoostingClassifier()
                        elif model_choice == "AdaBoost":
                            model = AdaBoostClassifier()
                        elif model_choice == "K-Nearest Neighbors":
                            model = KNeighborsClassifier()
                        elif model_choice == "Voting Classifier":
                            model1 = RandomForestClassifier()
                            model2 = LogisticRegression(max_iter=1000)
                            model3 = SVC()
                            model = VotingClassifier(estimators=[('rf', model1), ('lr', model2), ('svc', model3)], voting='hard')

                        # Regressor models
                        elif model_choice == "Random Forest Regressor":
                            model = RandomForestRegressor()
                        elif model_choice == "Linear Regression":
                            model = LinearRegression()
                        elif model_choice == "Support Vector Regressor":
                            model = SVR()
                        elif model_choice == "Gradient Boosting Regressor":
                            model = GradientBoostingRegressor()
                        elif model_choice == "AdaBoost Regressor":
                            model = AdaBoostRegressor()
                        elif model_choice == "K-Nearest Neighbors Regressor":
                            model = KNeighborsRegressor()

                        # Check if model is defined before training
                        if model is not None:
                            # Train the model
                            model.fit(X_train, y_train)
                            y_pred = model.predict(X_test)

                            # Evaluation Metrics
                            st.subheader("Model Evaluation")
                            if target_type == "classification":
                                accuracy = accuracy_score(y_test, y_pred)
                                st.write(f"Accuracy: {accuracy:.2f}")
                                st.write("Classification Report:")
                                st.text(classification_report(y_test, y_pred))

                                # Confusion Matrix
                                cm = confusion_matrix(y_test, y_pred)
                                plot_confusion_matrix(cm)

                                # Recommendations
                                if accuracy < 0.5:
                                    st.warning("The model performance is low. Consider trying different features, tuning hyperparameters, or using a more complex model.")
                                elif accuracy < 0.8:
                                    st.success("The model performance is acceptable but can be improved. Consider feature engineering or trying different algorithms.")
                                else:
                                    st.success("The model performs well. You can further optimize it or consider deploying it.")
                            else:
                                mse = mean_squared_error(y_test, y_pred)
                                r2 = r2_score(y_test, y_pred)
                                st.write(f"Mean Squared Error: {mse:.2f}")
                                st.write(f"R² Score: {r2:.2f}")

                                # Recommendations
                                if r2 < 0.5:
                                    st.warning("The model performance is low. Consider trying different features or tuning hyperparameters.")
                                elif r2 < 0.8:
                                    st.success("The model performance is acceptable but can be improved.")
                                else:
                                    st.success("The model performs well.")

    except Exception as e:
        st.error(f"An error occurred while processing the file: {e}")
else:
    st.info("Please upload a dataset to train and evaluate models.")


# Feature Engineering Page
if st.session_state.get("page") == "feature_engineering":
    st.title("Feature Engineering Page")
    # Add your feature engineering code here
