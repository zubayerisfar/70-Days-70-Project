"""
Rainfall Prediction — Streamlit App
Loads all 8 trained models (6 classical ML + ANN + LSTM) and returns a rain/no-rain
prediction (with probability) for user-entered weather conditions.

Run with:  streamlit run app.py
"""

import os
import joblib
import numpy as np
import pandas as pd
import streamlit as st
import torch
import torch.nn as nn

MODEL_DIR = "models"

# ----------------------------------------------------------------------------
# Model architectures (must match training notebook exactly to load state_dicts)
# ----------------------------------------------------------------------------
class RainfallDNN(nn.Module):
    def __init__(self, input_size):
        super(RainfallDNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.drop2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(64, 32)
        self.bn3 = nn.BatchNorm1d(32)
        self.drop3 = nn.Dropout(0.3)
        self.out = nn.Linear(32, 1)

    def forward(self, x):
        x = torch.relu(self.fc1(x)); x = self.bn1(x); x = self.drop1(x)
        x = torch.relu(self.fc2(x)); x = self.bn2(x); x = self.drop2(x)
        x = torch.relu(self.fc3(x)); x = self.bn3(x); x = self.drop3(x)
        return torch.sigmoid(self.out(x))


class RainfallLSTM(nn.Module):
    def __init__(self, input_size, hidden_size=64, num_layers=2, dropout=0.3):
        super(RainfallLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size=1, hidden_size=hidden_size, num_layers=num_layers,
                             batch_first=True, dropout=dropout if num_layers > 1 else 0)
        self.fc1 = nn.Linear(hidden_size, 32)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = x.unsqueeze(2)
        lstm_out, _ = self.lstm(x)
        x = lstm_out[:, -1, :]
        x = self.fc1(x); x = self.relu(x); x = self.dropout(x)
        x = self.fc2(x)
        return self.sigmoid(x)


# ----------------------------------------------------------------------------
# Cached loaders — everything loads once per session
# ----------------------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    scaler = joblib.load(os.path.join(MODEL_DIR, "scaler.joblib"))
    lencoders = joblib.load(os.path.join(MODEL_DIR, "label_encoders.joblib"))
    feature_columns = joblib.load(os.path.join(MODEL_DIR, "feature_columns.joblib"))
    return scaler, lencoders, feature_columns


@st.cache_resource
def load_ml_models():
    names = ["lr", "dt", "rf", "lgb", "cb", "xgb"]
    return {n: joblib.load(os.path.join(MODEL_DIR, f"model_{n}.joblib")) for n in names}


@st.cache_resource
def load_dl_models(input_size):
    dnn = RainfallDNN(input_size)
    dnn.load_state_dict(torch.load(os.path.join(MODEL_DIR, "dnn_model.pth"), map_location="cpu"))
    dnn.eval()

    lstm = RainfallLSTM(input_size)
    lstm.load_state_dict(torch.load(os.path.join(MODEL_DIR, "lstm_model.pth"), map_location="cpu"))
    lstm.eval()
    return dnn, lstm


MODEL_DISPLAY_NAMES = {
    "lr": "Logistic Regression", "dt": "Decision Tree", "rf": "Random Forest",
    "lgb": "LightGBM", "cb": "CatBoost", "xgb": "XGBoost",
    "ANN (DNN)": "ANN (Deep Neural Network)", "LSTM": "LSTM",
}


def build_input_dataframe(raw_inputs, feature_columns, lencoders):
    row = {}
    for col in feature_columns:
        val = raw_inputs[col]
        if col in lencoders:  # categorical -> encode with the saved LabelEncoder
            val = lencoders[col].transform([val])[0]
        row[col] = val
    return pd.DataFrame([row], columns=feature_columns)


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------
st.set_page_config(page_title="Rainfall Prediction", page_icon="🌧️", layout="wide")
st.title("🌧️ Rainfall Prediction — Will it rain tomorrow?")
st.write("Enter today's weather conditions and compare predictions from all 8 trained models.")

scaler, lencoders, feature_columns = load_artifacts()
ml_models = load_ml_models()
dnn_model, lstm_model = load_dl_models(input_size=len(feature_columns))

with st.form("weather_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        location = st.selectbox("Location", sorted(lencoders["Location"].classes_))
        min_temp = st.number_input("MinTemp (°C)", value=15.0)
        max_temp = st.number_input("MaxTemp (°C)", value=25.0)
        rainfall = st.number_input("Rainfall today (mm)", value=0.0, min_value=0.0)
        evaporation = st.number_input("Evaporation (mm)", value=5.0, min_value=0.0)
        sunshine = st.number_input("Sunshine (hours)", value=7.0, min_value=0.0, max_value=24.0)
        rain_today = st.selectbox("Rain today?", ["No", "Yes"])

    with col2:
        wind_gust_dir = st.selectbox("WindGustDir", sorted(lencoders["WindGustDir"].classes_))
        wind_gust_speed = st.number_input("WindGustSpeed (km/h)", value=40.0, min_value=0.0)
        wind_dir_9am = st.selectbox("WindDir9am", sorted(lencoders["WindDir9am"].classes_))
        wind_dir_3pm = st.selectbox("WindDir3pm", sorted(lencoders["WindDir3pm"].classes_))
        wind_speed_9am = st.number_input("WindSpeed9am (km/h)", value=15.0, min_value=0.0)
        wind_speed_3pm = st.number_input("WindSpeed3pm (km/h)", value=20.0, min_value=0.0)

    with col3:
        humidity_9am = st.slider("Humidity9am (%)", 0, 100, 60)
        humidity_3pm = st.slider("Humidity3pm (%)", 0, 100, 50)
        pressure_9am = st.number_input("Pressure9am (hPa)", value=1015.0)
        pressure_3pm = st.number_input("Pressure3pm (hPa)", value=1013.0)
        cloud_9am = st.slider("Cloud9am (oktas)", 0, 9, 4)
        cloud_3pm = st.slider("Cloud3pm (oktas)", 0, 9, 4)
        temp_9am = st.number_input("Temp9am (°C)", value=18.0)
        temp_3pm = st.number_input("Temp3pm (°C)", value=23.0)

    submitted = st.form_submit_button("Predict")

if submitted:
    raw_inputs = {
        "Location": location, "MinTemp": min_temp, "MaxTemp": max_temp, "Rainfall": rainfall,
        "Evaporation": evaporation, "Sunshine": sunshine, "WindGustDir": wind_gust_dir,
        "WindGustSpeed": wind_gust_speed, "WindDir9am": wind_dir_9am, "WindDir3pm": wind_dir_3pm,
        "WindSpeed9am": wind_speed_9am, "WindSpeed3pm": wind_speed_3pm, "Humidity9am": humidity_9am,
        "Humidity3pm": humidity_3pm, "Pressure9am": pressure_9am, "Pressure3pm": pressure_3pm,
        "Cloud9am": cloud_9am, "Cloud3pm": cloud_3pm, "Temp9am": temp_9am, "Temp3pm": temp_3pm,
        "RainToday": 1 if rain_today == "Yes" else 0,
    }

    input_df = build_input_dataframe(raw_inputs, feature_columns, lencoders)
    scaled_input = scaler.transform(input_df)

    rows = []
    # 6 classical ML models
    for key, model in ml_models.items():
        pred = model.predict(scaled_input)[0]
        proba = model.predict_proba(scaled_input)[0][1]
        rows.append({"Model": MODEL_DISPLAY_NAMES[key], "Prediction": "Rain" if pred == 1 else "No Rain",
                     "Rain Probability": f"{proba:.1%}"})

    # ANN + LSTM
    tensor_input = torch.tensor(scaled_input, dtype=torch.float32)
    with torch.no_grad():
        dnn_proba = dnn_model(tensor_input).item()
        lstm_proba = lstm_model(tensor_input).item()

    rows.append({"Model": MODEL_DISPLAY_NAMES["ANN (DNN)"],
                 "Prediction": "Rain" if dnn_proba >= 0.5 else "No Rain",
                 "Rain Probability": f"{dnn_proba:.1%}"})
    rows.append({"Model": MODEL_DISPLAY_NAMES["LSTM"],
                 "Prediction": "Rain" if lstm_proba >= 0.5 else "No Rain",
                 "Rain Probability": f"{lstm_proba:.1%}"})

    results_df = pd.DataFrame(rows)

    st.subheader("Predictions from all 8 models")
    st.dataframe(results_df, use_container_width=True)

    votes_rain = sum(1 for r in rows if r["Prediction"] == "Rain")
    st.metric("Models predicting rain tomorrow", f"{votes_rain} / 8")

    chart_df = results_df.copy()
    chart_df["Rain Probability (%)"] = chart_df["Rain Probability"].str.rstrip('%').astype(float)
    st.bar_chart(chart_df.set_index("Model")["Rain Probability (%)"])
