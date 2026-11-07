
import streamlit as st
import joblib as jb
import numpy as np

# Load trained model
model = jb.load("iris_model.pkl")

# Page configuration
st.set_page_config(
    page_title="Iris Flower Classifier",
    page_icon="🌸",
    layout="centered"
)

# Application title
st.title("Iris Flower Classifier 🌸")

st.write(
    "Enter the measurements of an iris flower "
    "to predict its species."
)

# User inputs
st.subheader("Flower Measurements")

sepal_length = st.slider(
    "Sepal Length (cm)", 4.0, 8.0, 5.1, 0.1
)

sepal_width = st.slider(
    "Sepal Width (cm)", 2.0, 4.5, 3.5, 0.1
)

petal_length = st.slider(
    "Petal Length (cm)", 1.0, 7.0, 1.4, 0.1
)

petal_width = st.slider(
    "Petal Width (cm)", 0.1, 2.5, 0.2, 0.1
)

# Prediction button
if st.button("Predict Species", use_container_width=True):

    # Prepare input
    input_data = np.array([[
        sepal_length,
        sepal_width,
        petal_length,
        petal_width
    ]])

    # Predict species
    prediction = model.predict(input_data)

    # Display result
    st.success(f"Predicted Species: {prediction[0]}")
