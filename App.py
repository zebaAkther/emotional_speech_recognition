import streamlit as st
import librosa
import numpy as np
import joblib

# Load model once
model = joblib.load('model.pkl')

emotion_map = {
    0: "neutral",
    1: "calm",
    2: "happy",
    3: "sad",
    4: "angry",
    5: "fearful",
    6: "disgust",
    7: "surprised"
}

def extract_features(file_path):
    audio, sample_rate = librosa.load(file_path, sr=None)
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sample_rate).T, axis=0)
    mel = np.mean(librosa.feature.melspectrogram(y=audio, sr=sample_rate).T, axis=0)
    features = np.concatenate((mfccs, chroma, mel))
    return features.reshape(1, -1)

st.title("Speech Emotion Recognition")

uploaded_file = st.file_uploader("Upload a speech audio file (.wav)", type=["wav"])

if uploaded_file is not None:
    with open("temp.wav", "wb") as f:
        f.write(uploaded_file.getbuffer())
    features = extract_features("temp.wav")
    prediction = model.predict(features)[0]
    emotion = emotion_map[prediction]
    st.write(f"Predicted Emotion: **{emotion}**")
