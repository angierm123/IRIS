import streamlit as st
import pickle
import numpy as np

#Load the model and scaler
with open('iris_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

# Define the user interface
st.title("Clasificación de Flores Iris 🌸")
st.write("Ingrese las dimensiones para predecir la especie de la flor.")

# User inputs
sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, step=0.1)
sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, step=0.1)
petal_length = st.number_input("Petal Length (cm)", min_value=0.0, step=0.1)
petal_width = st.number_input("Petal Width (cm)", min_value=0.0, step=0.1)

# Prediction button
if st.button("Predecir"):
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
    features_scaled = scaler.transform(features)
    prediction = model.predict(features_scaled)
    species = {0: "Setosa", 1: "Versicolor", 2: "Virginica"}
    
    st.success(f"La flor es de la especie: {species[prediction[0]]}")
