import pickle
import tensorflow as tf
import pandas as pd
import streamlit as st
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(page_title="Breast Cancer Detection", layout="wide")

st.title("🧬 Breast Cancer Detection using ANN")
st.write("This AI system predicts whether a tumor is **Benign or Malignant**.")

# Load model
model = tf.keras.models.load_model('model1.keras')

# Load scaler
with open("scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

st.subheader("Select Tumor Feature Values")

# ------------- 3 COLUMN LAYOUT ------------- #

col1, col2, col3 = st.columns(3)

with col1:
    radius_mean = st.slider("Radius Mean", 5.0, 30.0, 14.0)
    texture_mean = st.slider("Texture Mean", 5.0, 40.0, 20.0)
    perimeter_mean = st.slider("Perimeter Mean", 40.0, 200.0, 90.0)
    area_mean = st.slider("Area Mean", 100.0, 2500.0, 600.0)
    smoothness_mean = st.slider("Smoothness Mean", 0.05, 0.20, 0.10)
    compactness_mean = st.slider("Compactness Mean", 0.02, 0.40, 0.15)
    concavity_mean = st.slider("Concavity Mean", 0.0, 0.50, 0.20)
    concave_points_mean = st.slider("Concave Points Mean", 0.0, 0.20, 0.10)
    symmetry_mean = st.slider("Symmetry Mean", 0.10, 0.40, 0.20)
    fractal_dimension_mean = st.slider("Fractal Dimension Mean", 0.04, 0.10, 0.06)

with col2:
    radius_se = st.slider("Radius SE", 0.0, 2.5, 0.2)
    texture_se = st.slider("Texture SE", 0.0, 5.0, 1.0)
    perimeter_se = st.slider("Perimeter SE", 0.0, 10.0, 1.5)
    area_se = st.slider("Area SE", 0.0, 200.0, 20.0)
    smoothness_se = st.slider("Smoothness SE", 0.0, 0.02, 0.005)
    compactness_se = st.slider("Compactness SE", 0.0, 0.10, 0.02)
    concavity_se = st.slider("Concavity SE", 0.0, 0.10, 0.03)
    concave_points_se = st.slider("Concave Points SE", 0.0, 0.05, 0.01)
    symmetry_se = st.slider("Symmetry SE", 0.0, 0.10, 0.03)
    fractal_dimension_se = st.slider("Fractal Dimension SE", 0.0, 0.02, 0.004)

with col3:
    radius_worst = st.slider("Radius Worst", 7.0, 40.0, 16.0)
    texture_worst = st.slider("Texture Worst", 10.0, 50.0, 25.0)
    perimeter_worst = st.slider("Perimeter Worst", 50.0, 300.0, 105.0)
    area_worst = st.slider("Area Worst", 200.0, 4000.0, 800.0)
    smoothness_worst = st.slider("Smoothness Worst", 0.05, 0.30, 0.12)
    compactness_worst = st.slider("Compactness Worst", 0.05, 1.10, 0.20)
    concavity_worst = st.slider("Concavity Worst", 0.0, 1.30, 0.30)
    concave_points_worst = st.slider("Concave Points Worst", 0.0, 0.30, 0.15)
    symmetry_worst = st.slider("Symmetry Worst", 0.10, 0.70, 0.25)
    fractal_dimension_worst = st.slider("Fractal Dimension Worst", 0.05, 0.20, 0.08)


# -------- INPUT DATA -------- #

input_data = {
    'radius_mean':[radius_mean],
    'texture_mean':[texture_mean],
    'perimeter_mean':[perimeter_mean],
    'area_mean':[area_mean],
    'smoothness_mean':[smoothness_mean],
    'compactness_mean':[compactness_mean],
    'concavity_mean':[concavity_mean],
    'concave points_mean':[concave_points_mean],
    'symmetry_mean':[symmetry_mean],
    'fractal_dimension_mean':[fractal_dimension_mean],
    'radius_se':[radius_se],
    'texture_se':[texture_se],
    'perimeter_se':[perimeter_se],
    'area_se':[area_se],
    'smoothness_se':[smoothness_se],
    'compactness_se':[compactness_se],
    'concavity_se':[concavity_se],
    'concave points_se':[concave_points_se],
    'symmetry_se':[symmetry_se],
    'fractal_dimension_se':[fractal_dimension_se],
    'radius_worst':[radius_worst],
    'texture_worst':[texture_worst],
    'perimeter_worst':[perimeter_worst],
    'area_worst':[area_worst],
    'smoothness_worst':[smoothness_worst],
    'compactness_worst':[compactness_worst],
    'concavity_worst':[concavity_worst],
    'concave points_worst':[concave_points_worst],
    'symmetry_worst':[symmetry_worst],
    'fractal_dimension_worst':[fractal_dimension_worst]
}

input_df = pd.DataFrame(input_data)

st.write("")
st.write("")

# -------- PREDICT BUTTON -------- #

if st.button("🔍 Predict Tumor Type"):

    input_array = scaler.transform(input_df)

    prediction = model.predict(input_array)

    probability = prediction[0][0]

    if probability > 0.5:

        st.error("🔴 Malignant Tumor Detected")

        st.metric("Malignant Probability", f"{probability*100:.2f}%")

    else:

        st.success("🟢 Benign Tumor")

        st.metric("Benign Probability", f"{(1-probability)*100:.2f}%")