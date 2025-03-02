import streamlit as st
import pickle
import numpy as np

#Load the model and scaler
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the user interface
st.title("Iris Flower Classification 🌸")
st.write("Enter dimensions to predict the flower species.")

# User inputs
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# Prediction button
if st.button("Predict"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    species = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    
    st.success(f"The flower is of the species: {species[prediction[0]]}")
