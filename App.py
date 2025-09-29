import streamlit as st
import numpy as np
import joblib

@st.cache_resource
def load_artifacts():
    model = joblib.load('emotion_model.joblib')
    scaler = joblib.load('scaler.joblib')
    label_encoder = joblib.load('label_encoder.joblib')
    return model, scaler, label_encoder

model, scaler, label_encoder = load_artifacts()

st.title("Speech Emotion Recognition 🎤")
st.markdown("Enter feature values to predict the emotion (try real RAVDESS features).")

feature_count = model.n_features_in_
features = []

with st.form("features_form"):
    for i in range(feature_count):
        value = st.number_input(f"Feature {i+1}", value=0.0, key=f"f_{i+1}")
        features.append(value)
    submitted = st.form_submit_button("Predict")

if submitted:
    X = np.array(features).reshape(1, -1)
    X_scaled = scaler.transform(X)
    pred = model.predict(X_scaled)[0]
    label = label_encoder.inverse_transform([pred])[0]
    st.success(f"Predicted emotion: **{label}**")

st.caption("Upload feature vectors exported from your Kaggle workflow for a production deployment.")
