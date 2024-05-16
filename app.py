import streamlit as st
import joblib
import numpy as np

# Load the saved model
model_path = 'best_gbm_model.pkl'
model = joblib.load(model_path)

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

    user_input = np.array([age, weight, bmi, no_of_dependents, smoker, bloodpressure, diabetes, regular_ex])
    # Adjust based on the features your model expects
    return user_input.reshape(1, -1)

# Get user input
user_input = get_user_input()

# Display user input
st.subheader("User Input Features")
st.write(f"Age: {user_input[0,0]}")
st.write(f"Weight: {user_input[0,1]}")
st.write(f"BMI: {user_input[0,2]}")
st.write(f"Number of Dependents: {user_input[0,3]}")
st.write(f"Smoker: {user_input[0,4]}")
st.write(f"Blood Pressure: {user_input[0,5]}")
st.write(f"Diabetes: {user_input[0,6]}")
st.write(f"Regular Exercise: {user_input[0,7]}")
# Add more fields as necessary

# Predict using the model
prediction = model.predict(user_input)

# Display the prediction
st.subheader("Predicted Claim Amount")
st.write(f"${prediction[0]:.2f}")
