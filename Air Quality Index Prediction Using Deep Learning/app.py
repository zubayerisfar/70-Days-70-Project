import os
from pathlib import Path

import numpy as np
import streamlit as st
import torch
import torch.nn as nn


# -----------------------------
# Page configuration
# -----------------------------
st.set_page_config(
    page_title="AQI Forecast",
    page_icon="🌍",
    layout="centered",
)


# -----------------------------
# Model definition
# Must match the notebook exactly
# -----------------------------
class AQILSTM(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
        )
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = out[:, -1, :]
        return self.fc(out)


# -----------------------------
# AQI helpers
# -----------------------------
def aqi_status(aqi: float) -> str:
    if aqi <= 50:
        return "Good"
    elif aqi <= 100:
        return "Moderate"
    elif aqi <= 150:
        return "Unhealthy for Sensitive Groups"
    elif aqi <= 200:
        return "Unhealthy"
    elif aqi <= 300:
        return "Very Unhealthy"
    return "Hazardous"


def status_emoji(status: str) -> str:
    return {
        "Good": "🟢",
        "Moderate": "🟡",
        "Unhealthy for Sensitive Groups": "🟠",
        "Unhealthy": "🔴",
        "Very Unhealthy": "🟣",
        "Hazardous": "🟤",
    }.get(status, "🌫️")


# -----------------------------
# Load checkpoint
# -----------------------------
MODEL_PATH = Path(__file__).with_name("aqi_lstm_checkpoint.pth")


@st.cache_resource
def load_model(model_path: Path):
    # The notebook stores tensors and primitive Python values only.
    checkpoint = torch.load(model_path, map_location="cpu")

    input_size = int(checkpoint.get("input_size", 1))
    hidden_size = int(checkpoint.get("hidden_size", 64))
    num_layers = int(checkpoint.get("num_layers", 2))
    seq_len = int(checkpoint.get("seq_len", 7))

    model = AQILSTM(
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    scaler_scale = float(checkpoint["scaler_scale"])
    scaler_min = float(checkpoint["scaler_min"])

    return model, seq_len, scaler_scale, scaler_min


def predict_next_aqi(
    model: nn.Module,
    recent_aqi_values,
    seq_len: int,
    scaler_scale: float,
    scaler_min: float,
) -> float:
    if len(recent_aqi_values) != seq_len:
        raise ValueError(f"Please provide exactly {seq_len} AQI values.")

    values = np.asarray(recent_aqi_values, dtype=np.float32)

    if not np.all(np.isfinite(values)):
        raise ValueError("All AQI values must be valid numbers.")

    # Equivalent to sklearn MinMaxScaler.transform for one feature:
    # X_scaled = X * scaler.scale_[0] + scaler.min_[0]
    scaled = values * scaler_scale + scaler_min

    input_tensor = torch.tensor(
        scaled.reshape(1, seq_len, 1),
        dtype=torch.float32,
    )

    with torch.no_grad():
        pred_scaled = model(input_tensor).cpu().numpy()[0, 0]

    # Equivalent to MinMaxScaler.inverse_transform:
    # X = (X_scaled - scaler.min_[0]) / scaler.scale_[0]
    predicted_aqi = (float(pred_scaled) - scaler_min) / scaler_scale

    return predicted_aqi


# -----------------------------
# App UI
# -----------------------------
st.title("🌍 Next-Day AQI Forecast")
st.write(
    "Enter the AQI values for the previous 7 days. "
    "The trained LSTM model will estimate the AQI for the next day."
)

if not MODEL_PATH.exists():
    st.error(
        "Model checkpoint not found. Put `aqi_lstm_checkpoint.pth` "
        "in the same folder as `app.py`."
    )
    st.stop()

try:
    model, seq_len, scaler_scale, scaler_min = load_model(MODEL_PATH)
except Exception as exc:
    st.error("The model checkpoint could not be loaded.")
    st.exception(exc)
    st.stop()


with st.form("aqi_form"):
    st.subheader(f"Previous {seq_len} days")

    default_values = [55.0, 60.0, 58.0, 62.0, 65.0, 63.0, 66.0]
    if seq_len != len(default_values):
        default_values = [50.0] * seq_len

    recent_values = []
    cols = st.columns(min(seq_len, 4))

    for i in range(seq_len):
        with cols[i % len(cols)]:
            value = st.number_input(
                f"Day {i + 1}",
                min_value=0.0,
                max_value=1000.0,
                value=float(default_values[i]),
                step=1.0,
                key=f"aqi_{i}",
            )
            recent_values.append(value)

    submitted = st.form_submit_button(
        "Predict next-day AQI",
        use_container_width=True,
    )


if submitted:
    try:
        prediction = predict_next_aqi(
            model=model,
            recent_aqi_values=recent_values,
            seq_len=seq_len,
            scaler_scale=scaler_scale,
            scaler_min=scaler_min,
        )

        # AQI is not meaningful below zero. Keep the raw model estimate available
        # internally while presenting a physically sensible display value.
        display_prediction = max(0.0, prediction)
        status = aqi_status(display_prediction)

        st.divider()
        st.subheader("Prediction")

        left, right = st.columns(2)
        left.metric("Predicted AQI", f"{display_prediction:.2f}")
        right.metric("AQI status", f"{status_emoji(status)} {status}")

        st.subheader("Recent AQI trend")
        chart_values = [float(v) for v in recent_values] + [display_prediction]
        st.line_chart({"AQI": chart_values})

        if prediction < 0:
            st.caption(
                f"The model's raw estimate was {prediction:.2f}; "
                "the displayed AQI was clipped to 0 because AQI cannot be negative."
            )

        st.caption(
            "This prediction is based on the model trained in the supplied notebook "
            "and should not replace official air-quality measurements or health guidance."
        )

    except Exception as exc:
        st.error(f"Prediction failed: {exc}")


with st.expander("About the model"):
    st.write(
        f"""
        - Sequence length: **{seq_len} days**
        - Model: **2-layer LSTM**
        - Hidden size: **64**
        - Input features: **AQI Value only**
        - Output: **next-day AQI estimate**
        """
    )
