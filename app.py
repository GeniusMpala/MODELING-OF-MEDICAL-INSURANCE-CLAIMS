import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the saved model
model_path = 'best_gbm_model.pkl'
model = joblib.load(model_path)

# Load the feature names used in the model
feature_names = X.columns

# Title of the app
st.title("Medical Insurance Claim Prediction")

# Sidebar for user inputs
st.sidebar.header("Input Features")

# Function to get user input
def get_user_input():
    age = st.sidebar.slider("Age", 18, 100, 30)
    weight = st.sidebar.slider("Weight", 30, 150, 70)
    bmi = st.sidebar.slider("BMI", 15.0, 50.0, 25.0)
    no_of_dependents = st.sidebar.slider("Number of Dependents", 0, 10, 2)
    smoker = st.sidebar.selectbox("Smoker", [0, 1])
    bloodpressure = st.sidebar.slider("Blood Pressure", 60, 200, 120)
    diabetes = st.sidebar.selectbox("Diabetes", [0, 1])
    regular_ex = st.sidebar.selectbox("Regular Exercise", [0, 1])
    # Add more fields as necessary

    # Create a DataFrame for the input
    input_data = pd.DataFrame({
        'age': [age],
        'weight': [weight],
        'bmi': [bmi],
        'no_of_dependents': [no_of_dependents],
        'smoker': [smoker],
        'bloodpressure': [bloodpressure],
        'diabetes': [diabetes],
        'regular_ex': [regular_ex]
    })

    # Include default values for one-hot encoded features
    for feature in feature_names:
        if feature not in input_data.columns:
            input_data[feature] = 0

    return input_data[feature_names]

# Get user input
user_input = get_user_input()

# Display user input
st.subheader("User Input Features")
st.write(user_input)

# Predict using the model
prediction = model.predict(user_input)

# Display the prediction
st.subheader("Predicted Claim Amount")
st.write(f"${prediction[0]:.2f}")
