from pathlib import Path
import os
import datetime as dt

import joblib
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
from tensorflow.keras import layers


# -----------------------------------------------------------------------------
# Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="AEP Energy Prediction",
    page_icon="⚡",
    layout="centered",
)

DATA_PATH = Path(os.getenv("AEP_DATA_PATH", "data/AEP_hourly.csv"))
MODEL_PATH = Path(os.getenv("AEP_MODEL_PATH", "energy_model.keras"))
TARGET_SCALER_PATH = Path(
    os.getenv("AEP_TARGET_SCALER_PATH", "target_scaler.joblib"))
FEATURE_SCALER_PATH = Path(
    os.getenv("AEP_FEATURE_SCALER_PATH", "feature_scaler.joblib"))

FEATURE_COLUMNS = [
    "Month_sin",
    "Month_cos",
    "Hour_sin",
    "Hour_cos",
    "DayOfWeek_sin",
    "DayOfWeek_cos",
    "AEP_MW_lag_1",
    "AEP_MW_lag_2",
    "AEP_MW_lag_24",
]

TEST_SIZE = 0.20
RANDOM_SEED = 42
EPOCHS = 200
BATCH_SIZE = 256


# -----------------------------------------------------------------------------
# Data preparation - intentionally mirrors the supplied TensorFlow notebook.
# IMPORTANT: We DO NOT sort the CSV because the notebook did not sort it either.
# Sorting here would change lag values and the first-80% training split.
# -----------------------------------------------------------------------------
def prepare_dataset(csv_path: Path):
    dataset = pd.read_csv(csv_path)

    required = {"Datetime", "AEP_MW"}
    missing = required - set(dataset.columns)
    if missing:
        raise ValueError(
            f"CSV is missing required column(s): {', '.join(sorted(missing))}")

    dataset = dataset.copy()
    dataset["Datetime"] = pd.to_datetime(dataset["Datetime"], errors="coerce")
    dataset["AEP_MW"] = pd.to_numeric(dataset["AEP_MW"], errors="coerce")
    dataset = dataset.dropna(
        subset=["Datetime", "AEP_MW"]).reset_index(drop=True)

    # Same target scaling as the notebook: fit on the full AEP_MW column.
    target_scaler = MinMaxScaler()
    dataset["AEP_MW_scaled"] = target_scaler.fit_transform(
        dataset[["AEP_MW"]]
    ).reshape(-1)

    # Same lag features as the notebook.
    dataset["AEP_MW_lag_1"] = dataset["AEP_MW_scaled"].shift(1)
    dataset["AEP_MW_lag_2"] = dataset["AEP_MW_scaled"].shift(2)
    dataset["AEP_MW_lag_24"] = dataset["AEP_MW_scaled"].shift(24)

    # Same datetime components / cyclic encodings as the notebook.
    dataset["Year"] = dataset["Datetime"].dt.year
    dataset["Month"] = dataset["Datetime"].dt.month
    dataset["Day"] = dataset["Datetime"].dt.day
    dataset["Hour"] = dataset["Datetime"].dt.hour

    dataset["Month_sin"] = np.sin(2 * np.pi * dataset["Month"] / 12)
    dataset["Month_cos"] = np.cos(2 * np.pi * dataset["Month"] / 12)
    dataset["Hour_sin"] = np.sin(2 * np.pi * dataset["Hour"] / 24)
    dataset["Hour_cos"] = np.cos(2 * np.pi * dataset["Hour"] / 24)

    dataset["DayOfWeek"] = dataset["Datetime"].dt.dayofweek
    dataset["DayOfWeek_sin"] = np.sin(2 * np.pi * dataset["DayOfWeek"] / 7)
    dataset["DayOfWeek_cos"] = np.cos(2 * np.pi * dataset["DayOfWeek"] / 7)

    clean = dataset.dropna(subset=FEATURE_COLUMNS + ["AEP_MW_scaled"]).copy()
    clean = clean.reset_index(drop=True)

    if len(clean) < 10:
        raise ValueError("Not enough usable rows after creating lag features.")

    # Mirrors train_test_split(..., test_size=0.2, shuffle=False).
    split_index = int(len(clean) * (1 - TEST_SIZE))
    train_df = clean.iloc[:split_index].copy()
    test_df = clean.iloc[split_index:].copy()

    X_train = train_df[FEATURE_COLUMNS].to_numpy()
    y_train = train_df["AEP_MW_scaled"].to_numpy().reshape(-1, 1)
    X_test = test_df[FEATURE_COLUMNS].to_numpy()
    y_test = test_df["AEP_MW_scaled"].to_numpy().reshape(-1, 1)

    feature_scaler = MinMaxScaler()
    X_train_scaled = feature_scaler.fit_transform(X_train).astype("float32")
    X_test_scaled = feature_scaler.transform(X_test).astype("float32")

    y_train = y_train.astype("float32")
    y_test = y_test.astype("float32")

    return {
        "clean": clean,
        "train_df": train_df,
        "test_df": test_df,
        "X_train": X_train_scaled,
        "X_test": X_test_scaled,
        "y_train": y_train,
        "y_test": y_test,
        "target_scaler": target_scaler,
        "feature_scaler": feature_scaler,
    }


def build_energy_model(input_size: int):
    model = keras.Sequential(
        [
            layers.Input(shape=(input_size,)),
            layers.Dense(128),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.30),
            layers.Dense(64),
            layers.BatchNormalization(),
            layers.ReLU(),
            layers.Dropout(0.20),
            layers.Dense(32, activation="relu"),
            layers.Dense(1),
        ],
        name="EnergyConsumptionModel",
    )

    optimizer = keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=1e-5,
    )
    model.compile(
        optimizer=optimizer,
        loss="mse",
        metrics=[
            keras.metrics.MeanAbsoluteError(name="mae"),
            keras.metrics.RootMeanSquaredError(name="rmse"),
        ],
    )
    return model


@st.cache_resource(show_spinner="Loading data and model…")
def load_assets(data_path_str: str):
    data_path = Path(data_path_str)
    prepared = prepare_dataset(data_path)

    # Prefer previously exported artifacts. If they do not exist, train once and
    # save them. Streamlit cache prevents retraining on every widget interaction.
    artifacts_exist = (
        MODEL_PATH.exists()
        and TARGET_SCALER_PATH.exists()
        and FEATURE_SCALER_PATH.exists()
    )

    if artifacts_exist:
        model = keras.models.load_model(MODEL_PATH)
        target_scaler = joblib.load(TARGET_SCALER_PATH)
        feature_scaler = joblib.load(FEATURE_SCALER_PATH)

        # Transform test data with the loaded scaler so evaluation matches model.
        X_test = feature_scaler.transform(
            prepared["test_df"][FEATURE_COLUMNS].to_numpy()
        ).astype("float32")
    else:
        tf.random.set_seed(RANDOM_SEED)
        np.random.seed(RANDOM_SEED)

        model = build_energy_model(len(FEATURE_COLUMNS))
        lr_scheduler = keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            mode="min",
            factor=0.5,
            patience=10,
            verbose=0,
        )

        model.fit(
            prepared["X_train"],
            prepared["y_train"],
            validation_data=(prepared["X_test"], prepared["y_test"]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            callbacks=[lr_scheduler],
            verbose=0,
        )

        target_scaler = prepared["target_scaler"]
        feature_scaler = prepared["feature_scaler"]
        X_test = prepared["X_test"]

        # Save for future app launches.
        model.save(MODEL_PATH)
        joblib.dump(target_scaler, TARGET_SCALER_PATH)
        joblib.dump(feature_scaler, FEATURE_SCALER_PATH)

    # Test-set metrics for context.
    pred_scaled = model.predict(X_test, verbose=0)
    pred_mw = target_scaler.inverse_transform(pred_scaled).reshape(-1)
    actual_mw = target_scaler.inverse_transform(prepared["y_test"]).reshape(-1)

    mse = mean_squared_error(actual_mw, pred_mw)
    rmse = float(np.sqrt(mse))
    mae = float(mean_absolute_error(actual_mw, pred_mw))
    r2 = float(r2_score(actual_mw, pred_mw))

    nonzero = actual_mw != 0
    ape = np.abs((actual_mw[nonzero] - pred_mw[nonzero]
                  ) / actual_mw[nonzero]) * 100
    acc_5 = float(np.mean(ape <= 5) * 100) if len(ape) else float("nan")
    acc_10 = float(np.mean(ape <= 10) * 100) if len(ape) else float("nan")

    metrics = {
        "RMSE": rmse,
        "MAE": mae,
        "R2": r2,
        "ACC5": acc_5,
        "ACC10": acc_10,
    }

    return prepared, model, target_scaler, feature_scaler, metrics


def predict_for_row(row: pd.Series, model, target_scaler, feature_scaler) -> float:
    features = row[FEATURE_COLUMNS].to_numpy(dtype=float).reshape(1, -1)
    features_scaled = feature_scaler.transform(features).astype("float32")
    pred_scaled = model.predict(features_scaled, verbose=0)
    return float(target_scaler.inverse_transform(pred_scaled)[0, 0])


# -----------------------------------------------------------------------------
# App UI
# -----------------------------------------------------------------------------
st.title("⚡ AEP Hourly Energy Prediction")
st.caption(
    "Choose a datetime from the model's training period. Predictions outside the "
    "training split are intentionally blocked."
)

if not DATA_PATH.exists():
    st.error(
        f"Dataset not found at `{DATA_PATH}`.\n\n"
        "Place `AEP_hourly.csv` inside a `data/` folder next to `app.py`, "
        "or set the `AEP_DATA_PATH` environment variable."
    )
    st.stop()

try:
    prepared, model, target_scaler, feature_scaler, metrics = load_assets(
        str(DATA_PATH))
except Exception as exc:
    st.exception(exc)
    st.stop()

train_df = prepared["train_df"].copy()

# Only datetimes that were actually part of the first-80% training split are valid.
valid_datetimes = pd.DatetimeIndex(train_df["Datetime"])
train_min = valid_datetimes.min().to_pydatetime()
train_max = valid_datetimes.max().to_pydatetime()

st.info(
    f"**Allowed training datetime range:** "
    f"{train_min:%Y-%m-%d %H:%M} to {train_max:%Y-%m-%d %H:%M}"
)

with st.sidebar:
    st.header("Model quality")
    st.metric("Test RMSE", f"{metrics['RMSE']:,.2f} MW")
    st.metric("Test MAE", f"{metrics['MAE']:,.2f} MW")
    st.metric("Test R²", f"{metrics['R2']:.4f}")
    st.metric("Within ±5%", f"{metrics['ACC5']:.2f}%")
    st.metric("Within ±10%", f"{metrics['ACC10']:.2f}%")
    st.caption("Tolerance accuracy is used because this is a regression model.")

# Keep defaults within the valid range.
default_dt = train_max
selected_date = st.date_input(
    "Date",
    value=default_dt.date(),
    min_value=train_min.date(),
    max_value=train_max.date(),
)
selected_time = st.time_input(
    "Time",
    value=default_dt.time().replace(minute=0, second=0, microsecond=0),
    step=dt.timedelta(hours=1),
)
selected_dt = pd.Timestamp.combine(selected_date, selected_time)

if st.button("Predict energy consumption", type="primary", use_container_width=True):
    # Hard guard #1: never exceed the training datetime bounds.
    if selected_dt < pd.Timestamp(train_min) or selected_dt > pd.Timestamp(train_max):
        st.error(
            "That datetime is outside the model's training period. "
            f"Please choose a datetime between {train_min:%Y-%m-%d %H:%M} "
            f"and {train_max:%Y-%m-%d %H:%M}."
        )
        st.stop()

    # Hard guard #2: the datetime must actually be a row in the training split.
    matches = train_df.loc[train_df["Datetime"] == selected_dt]
    if matches.empty:
        st.error(
            "That exact hourly timestamp is not present in the training data. "
            "Choose another hour within the displayed training range."
        )
        st.stop()

    # DST can create duplicate local timestamps. Use the first occurrence exactly
    # as represented by the original CSV order, which preserves notebook behavior.
    row = matches.iloc[0]
    prediction_mw = predict_for_row(
        row,
        model=model,
        target_scaler=target_scaler,
        feature_scaler=feature_scaler,
    )

    st.success(f"Predicted AEP demand: **{prediction_mw:,.2f} MW**")
    st.caption(f"Prediction datetime: {selected_dt:%Y-%m-%d %H:%M}")

    with st.expander("Prediction inputs used by the model"):
        st.write(
            {
                "1-hour lag (scaled)": float(row["AEP_MW_lag_1"]),
                "2-hour lag (scaled)": float(row["AEP_MW_lag_2"]),
                "24-hour lag (scaled)": float(row["AEP_MW_lag_24"]),
                "month": int(row["Month"]),
                "hour": int(row["Hour"]),
                "day_of_week": int(row["DayOfWeek"]),
            }
        )

st.divider()
st.caption(
    "Safety rule: the app only predicts timestamps contained in the notebook's "
    "first-80% training split. It will not predict into the held-out test period "
    "or beyond the training data."
)
