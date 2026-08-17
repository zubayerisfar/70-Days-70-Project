import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt


MODEL_PATH = "crowd_counting_unet_clean.keras"


# -----------------------------
# Load model once
# -----------------------------
@st.cache_resource
def load_model():

    model = tf.keras.models.load_model(
        MODEL_PATH,
        compile=False
    )

    return model


model = load_model()


# -----------------------------
# Page
# -----------------------------
st.set_page_config(
    page_title="Crowd Density Estimation",
    layout="wide"
)

st.title("Crowd Density Estimation using U-Net CNN")

st.write(
    "Upload a crowd image and the model will generate a density map and estimate the number of people."
)


# -----------------------------
# Upload image
# -----------------------------
uploaded_file = st.file_uploader(
    "Upload crowd image",
    type=[
        "jpg",
        "jpeg",
        "png"
    ]
)


if uploaded_file:

    image = Image.open(
        uploaded_file
    ).convert("RGB")

    col1, col2 = st.columns(2)

    with col1:

        st.subheader(
            "Original Image"
        )

        st.image(
            image,
            use_container_width=True
        )

    # -----------------------------
    # Preprocessing
    # -----------------------------

    resized = image.resize(
        (320, 240)
    )

    img = np.array(
        resized
    ).astype(
        "float32"
    )

    img = img / 255.0

    img = np.expand_dims(
        img,
        axis=0
    )

    # -----------------------------
    # Prediction
    # -----------------------------

    prediction = model.predict(
        img
    )

    density_map = prediction[0, :, :, 0]

    # Count formula from your metadata
    count = (
        np.sum(density_map)
        /
        1000.0
    )

    with col2:

        st.subheader(
            "Prediction"
        )

        st.metric(
            "Estimated People",
            f"{count:.2f}"
        )

    # -----------------------------
    # Density map
    # -----------------------------

    st.subheader(
        "Density Heatmap"
    )

    fig, ax = plt.subplots(
        figsize=(8, 5)
    )

    ax.imshow(
        density_map,
        cmap="jet"
    )

    ax.axis(
        "off"
    )

    st.pyplot(
        fig
    )

    # -----------------------------
    # Overlay
    # -----------------------------

    st.subheader(
        "Density Overlay"
    )

    fig2, ax2 = plt.subplots(
        figsize=(8, 5)
    )

    ax2.imshow(
        image
    )

    ax2.imshow(
        density_map,
        cmap="jet",
        alpha=0.45
    )

    ax2.axis(
        "off"
    )

    st.pyplot(
        fig2
    )
