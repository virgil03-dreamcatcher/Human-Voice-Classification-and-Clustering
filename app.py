import streamlit as st
import joblib
import numpy as np
import pandas as pd
import librosa
import tempfile
from scipy.stats import skew, kurtosis
import json
# Load model and scaler
model = joblib.load("rf_voice_classifier.pkl")
scaler = joblib.load("feature_scaler.pkl")

# List of feature columns used in training (replace with your actual feature columns)
with open('feature_cols.json', 'r') as f:
    feature_cols = json.load(f)

st.title("Human Voice Gender Classification")

audio_file = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac"])

def extract_features(audio_path):
    y, sr = librosa.load(audio_path, sr=None)
    features = {}

    features['mean_spectral_centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr).mean()
    features['std_spectral_centroid'] = librosa.feature.spectral_centroid(y=y, sr=sr).std()

    features['mean_spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=sr).mean()
    features['std_spectral_bandwidth'] = librosa.feature.spectral_bandwidth(y=y, sr=sr).std()

    features['mean_spectral_contrast'] = librosa.feature.spectral_contrast(y=y, sr=sr).mean()
    features['mean_spectral_flatness'] = librosa.feature.spectral_flatness(y=y).mean()
    features['mean_spectral_rolloff'] = librosa.feature.spectral_rolloff(y=y, sr=sr).mean()

    features['zero_crossing_rate'] = librosa.feature.zero_crossing_rate(y).mean()
    features['rms_energy'] = librosa.feature.rms(y=y).mean()

    f0, voiced_flag, voiced_probs = librosa.pyin(y, fmin=librosa.note_to_hz('C2'),
                                                 fmax=librosa.note_to_hz('C7'))
    f0 = f0[~np.isnan(f0)] if f0 is not None else np.array([])
    features['mean_pitch'] = f0.mean() if len(f0) > 0 else 0
    features['min_pitch'] = f0.min() if len(f0) > 0 else 0
    features['max_pitch'] = f0.max() if len(f0) > 0 else 0
    features['std_pitch'] = f0.std() if len(f0) > 0 else 0

    spec = np.mean(np.abs(librosa.stft(y)), axis=1)
    features['spectral_skew'] = skew(spec)
    features['spectral_kurtosis'] = kurtosis(spec)

    energy = y**2
    energy_entropy = -np.sum(energy * np.log2(energy + 1e-10))
    features['energy_entropy'] = energy_entropy
    features['log_energy'] = np.log(np.sum(energy) + 1e-10)

    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
    for i in range(13):
        features[f'mfcc_{i+1}_mean'] = mfccs[i].mean()
        features[f'mfcc_{i+1}_std'] = mfccs[i].std()

    return features

if audio_file is not None:
    with tempfile.NamedTemporaryFile(delete=False) as tmp_file:
        tmp_file.write(audio_file.read())
        tmp_path = tmp_file.name

    features = extract_features(tmp_path)
    st.write("Extracted features keys:", list(features.keys()))

    missing = [col for col in feature_cols if col not in features.keys()]
    if missing:
        st.error(f"Missing features in extraction: {missing}")
    else:
        input_df = pd.DataFrame([features])

        # Reorder columns exactly as in feature_cols
        input_df = input_df[feature_cols]

        # Scale features
        input_df_scaled = scaler.transform(input_df)

        # Convert scaled array back to DataFrame for clarity (optional)
        input_df_scaled = pd.DataFrame(input_df_scaled, columns=feature_cols)

        prediction = model.predict(input_df_scaled)[0]
        gender = "Male" if prediction == 1 else "Female"
        st.success(f"Predicted Gender: {gender}")


    features = extract_features(tmp_path)
    st.write("Extracted features keys:", list(features.keys()))

    missing = [col for col in feature_cols if col not in features.keys()]
    if missing:
        st.error(f"Missing features in extraction: {missing}")
    else:
        input_df = pd.DataFrame([features])
        input_df = input_df[feature_cols]  # reorder columns properly
        input_df[feature_cols] = scaler.transform(input_df[feature_cols])
        prediction = model.predict(input_df)[0]
        gender = "Male" if prediction == 1 else "Female"
        st.success(f"Predicted Gender: {gender}")
