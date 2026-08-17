from pathlib import Path

import numpy as np
import streamlit as st
import keras

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "heart_model.keras"
PREPROCESS_PATH = BASE_DIR / "heart_preprocessing.npz"

FEATURES = [
    "age",
    "sex",
    "smoker",
    "years_of_smoking",
    "LDL_cholesterol",
    "chest_pain_type",
    "height",
    "weight",
    "familyhist",
    "activity",
    "lifestyle",
    "cardiac intervention",
    "heart_rate",
    "diabets",
    "blood_pressure_sys",
    "blood_pressure_dias",
    "hypertention",
    "Interventricular_septal_end_diastole",
    "ecg_pattern",
    "Q_wave",
]

st.set_page_config(
    page_title="Heart Disease Prediction",
    page_icon="❤️",
    layout="wide",
)


@st.cache_resource
def load_artifacts():
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Missing {MODEL_PATH.name}. Put it in the same folder as app.py."
        )
    if not PREPROCESS_PATH.exists():
        raise FileNotFoundError(
            f"Missing {PREPROCESS_PATH.name}. Put it in the same folder as app.py."
        )

    model = keras.models.load_model(MODEL_PATH)

    preprocessing = np.load(PREPROCESS_PATH, allow_pickle=False)
    mean = preprocessing["mean"].astype(np.float32)
    std = preprocessing["std"].astype(np.float32)
    saved_features = preprocessing["feature_names"].astype(str).tolist()

    if saved_features != FEATURES:
        raise ValueError(
            "Feature order in heart_preprocessing.npz does not match app.py."
        )

    if len(mean) != len(FEATURES) or len(std) != len(FEATURES):
        raise ValueError("Preprocessing file does not contain 20 feature values.")

    # Avoid division by zero if a column happened to have zero variance.
    std = np.where(std == 0, 1.0, std)

    return model, mean, std


def binary_input(label, key):
    return st.selectbox(
        label,
        options=[0, 1],
        index=0,
        key=key,
        help="Use the same 0/1 coding used in the training dataset.",
    )


st.title("❤️ Heart Disease Prediction")
st.caption(
    "Educational demonstration of your trained neural-network model. "
    "This is not a medical diagnosis."
)

try:
    model, mean, std = load_artifacts()
except Exception as exc:
    st.error(str(exc))
    st.stop()

st.subheader("Enter patient data")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=20, max_value=90, value=55, step=1)
    sex = binary_input("Sex (dataset code)", "sex")
    smoker = binary_input("Smoker", "smoker")
    years_of_smoking = st.number_input(
        "Years of smoking", min_value=0, max_value=50, value=0, step=1
    )
    LDL_cholesterol = st.number_input(
        "LDL cholesterol", min_value=26.0, max_value=260.0, value=110.0, step=1.0
    )
    chest_pain_type = st.selectbox(
        "Chest pain type", options=[1, 2, 3, 4], index=2
    )
    height = st.number_input(
        "Height", min_value=128.0, max_value=192.0, value=162.0, step=1.0
    )
    weight = st.number_input(
        "Weight", min_value=41.0, max_value=134.0, value=81.0, step=0.5
    )
    familyhist = binary_input("Family history", "familyhist")
    activity = binary_input("Activity", "activity")

with col2:
    # The uploaded notebook does not document the semantic coding/range
    # of "lifestyle", so keep it as the raw dataset code.
    lifestyle = st.number_input(
        "Lifestyle (dataset code)", value=0.0, step=1.0
    )
    cardiac_intervention = binary_input(
        "Cardiac intervention", "cardiac_intervention"
    )
    heart_rate = st.number_input(
        "Heart rate", min_value=40, max_value=140, value=84, step=1
    )
    diabets = binary_input("Diabetes (dataset field: diabets)", "diabets")
    blood_pressure_sys = st.number_input(
        "Systolic blood pressure",
        min_value=80,
        max_value=220,
        value=120,
        step=1,
    )
    blood_pressure_dias = st.number_input(
        "Diastolic blood pressure",
        min_value=40,
        max_value=140,
        value=70,
        step=1,
    )
    hypertention = binary_input(
        "Hypertension (dataset field: hypertention)", "hypertention"
    )
    ivs = binary_input(
        "Interventricular septal end diastole", "ivs"
    )
    ecg_pattern = st.selectbox(
        "ECG pattern", options=[1, 2, 3, 4], index=3
    )
    Q_wave = binary_input("Q wave", "Q_wave")


values = {
    "age": age,
    "sex": sex,
    "smoker": smoker,
    "years_of_smoking": years_of_smoking,
    "LDL_cholesterol": LDL_cholesterol,
    "chest_pain_type": chest_pain_type,
    "height": height,
    "weight": weight,
    "familyhist": familyhist,
    "activity": activity,
    "lifestyle": lifestyle,
    "cardiac intervention": cardiac_intervention,
    "heart_rate": heart_rate,
    "diabets": diabets,
    "blood_pressure_sys": blood_pressure_sys,
    "blood_pressure_dias": blood_pressure_dias,
    "hypertention": hypertention,
    "Interventricular_septal_end_diastole": ivs,
    "ecg_pattern": ecg_pattern,
    "Q_wave": Q_wave,
}

st.divider()

if st.button("Predict", type="primary", use_container_width=True):
    raw = np.array(
        [[values[name] for name in FEATURES]],
        dtype=np.float32,
    )

    # Apply EXACTLY the same normalization used during training.
    x = (raw - mean) / std

    output = np.asarray(model.predict(x, verbose=0))

    if output.ndim != 2 or output.shape[0] != 1:
        st.error(f"Unexpected model output shape: {output.shape}")
        st.stop()

    if output.shape[1] == 2:
        scores = output[0]
        predicted_class = int(np.argmax(scores))

        if predicted_class == 1:
            st.error("Prediction: Heart disease class (1)")
        else:
            st.success("Prediction: No-heart-disease class (0)")

        st.write(
            f"Class 0 model score: **{float(scores[0]):.4f}**  \n"
            f"Class 1 model score: **{float(scores[1]):.4f}**"
        )

        # If the final layer is softmax, these are probabilities.
        if np.isclose(float(scores.sum()), 1.0, atol=1e-3):
            st.metric(
                "Predicted-class probability",
                f"{float(scores[predicted_class]) * 100:.2f}%",
            )
        else:
            st.caption(
                "Your current 2-output sigmoid model produces class scores that "
                "do not necessarily sum to 1, so they should not be interpreted "
                "as calibrated probabilities."
            )

    elif output.shape[1] == 1:
        probability = float(output[0, 0])
        predicted_class = int(probability >= 0.5)

        if predicted_class == 1:
            st.error("Prediction: Heart disease class (1)")
        else:
            st.success("Prediction: No-heart-disease class (0)")

        st.metric("Class 1 score", f"{probability * 100:.2f}%")

    else:
        st.error(
            "The app supports either a 1-output binary model or a 2-output "
            "categorical model."
        )
