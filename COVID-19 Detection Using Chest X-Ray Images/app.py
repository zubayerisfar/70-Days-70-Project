import os

import streamlit as st
import torch
import torch.nn as nn
from PIL import Image
from torchvision import models, transforms


# ---------------------------------------------------------
# Page configuration
# ---------------------------------------------------------
st.set_page_config(
    page_title="COVID-19 Chest X-ray Classifier",
    page_icon="🫁",
    layout="centered",
)

MODEL_PATH = "covid_xray_resnet18.pth"

DEFAULT_IMAGE_SIZE = 224
DEFAULT_THRESHOLD = 0.5
DEFAULT_CLASS_NAMES = ["covid", "normal"]

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ---------------------------------------------------------
# Model loading
# ---------------------------------------------------------
@st.cache_resource
def load_model(model_path: str):
    if not os.path.exists(model_path):
        raise FileNotFoundError(
            f"Model file '{model_path}' was not found. "
            "Place covid_xray_resnet18.pth in the same folder as app.py."
        )

    checkpoint = torch.load(
        model_path,
        map_location=DEVICE,
        weights_only=False,
    )

    class_names = checkpoint.get("class_names", DEFAULT_CLASS_NAMES)
    image_size = checkpoint.get("image_size", DEFAULT_IMAGE_SIZE)
    threshold = checkpoint.get("threshold", DEFAULT_THRESHOLD)

    # The notebook trained ResNet18 and replaced the final layer with
    # one output neuron for BCEWithLogitsLoss.
    model = models.resnet18(weights=None)
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, 1)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    return model, class_names, image_size, threshold


def build_transform(image_size: int):
    """Use the same preprocessing as the notebook's validation transform."""
    return transforms.Compose(
        [
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ]
    )


def predict_image(image: Image.Image, model, class_names, image_size, threshold):
    image = image.convert("RGB")
    transform = build_transform(image_size)

    tensor = transform(image).unsqueeze(0).to(DEVICE)

    with torch.no_grad():
        logit = model(tensor)
        probability_class_1 = torch.sigmoid(logit).item()

    # In the notebook, ImageFolder produced:
    # class_names = ['covid', 'normal']
    #
    # BCE labels are therefore:
    # covid  -> 0
    # normal -> 1
    #
    # So sigmoid(logit) is the probability of class index 1 ("normal").
    probability_normal = probability_class_1
    probability_covid = 1.0 - probability_normal

    predicted_index = 1 if probability_class_1 >= threshold else 0

    if len(class_names) >= 2:
        predicted_class = class_names[predicted_index]
    else:
        predicted_class = "normal" if predicted_index == 1 else "covid"

    confidence = (
        probability_normal
        if predicted_index == 1
        else probability_covid
    )

    return {
        "predicted_class": predicted_class,
        "confidence": confidence,
        "probability_covid": probability_covid,
        "probability_normal": probability_normal,
    }


# ---------------------------------------------------------
# UI
# ---------------------------------------------------------
st.title("🫁 COVID-19 Chest X-ray Classifier")
st.write(
    "Upload a chest X-ray image and the trained ResNet18 model will "
    "classify it as **COVID** or **Normal**."
)

st.warning(
    "Educational/demo use only. This model is not a medical device and "
    "must not be used to diagnose, rule out, or treat COVID-19. "
    "Clinical decisions should be made by qualified healthcare professionals."
)

with st.sidebar:
    st.header("Model")
    st.write(f"Device: **{DEVICE.type.upper()}**")
    st.write("Architecture: **ResNet18**")
    st.write("Input: **Chest X-ray image**")

try:
    model, class_names, image_size, threshold = load_model(MODEL_PATH)

    with st.sidebar:
        st.success("Model loaded")
        st.write(f"Classes: **{', '.join(class_names)}**")
        st.write(f"Image size: **{image_size} × {image_size}**")
        st.write(f"Threshold: **{threshold:.2f}**")

except Exception as exc:
    st.error("The trained model could not be loaded.")
    st.code(str(exc))
    st.info(
        "Put `covid_xray_resnet18.pth` in the same directory as `app.py`, "
        "then restart the Streamlit app."
    )
    st.stop()


uploaded_file = st.file_uploader(
    "Choose a chest X-ray",
    type=["jpg", "jpeg", "png"],
)

if uploaded_file is None:
    st.info("Upload a JPG, JPEG, or PNG chest X-ray to begin.")

else:
    try:
        image = Image.open(uploaded_file).convert("RGB")
    except Exception:
        st.error("That file could not be opened as an image.")
        st.stop()

    st.image(
        image,
        caption="Uploaded chest X-ray",
        use_container_width=True,
    )

    if st.button("Analyze X-ray", type="primary", use_container_width=True):
        try:
            with st.spinner("Analyzing image..."):
                result = predict_image(
                    image=image,
                    model=model,
                    class_names=class_names,
                    image_size=image_size,
                    threshold=threshold,
                )

            predicted = result["predicted_class"].lower()
            confidence_pct = result["confidence"] * 100

            st.subheader("Prediction")

            if predicted == "covid":
                st.error(f"COVID — confidence: {confidence_pct:.2f}%")
            elif predicted == "normal":
                st.success(f"Normal — confidence: {confidence_pct:.2f}%")
            else:
                st.info(
                    f"{result['predicted_class']} — "
                    f"confidence: {confidence_pct:.2f}%"
                )

            col1, col2 = st.columns(2)
            col1.metric(
                "COVID probability",
                f"{result['probability_covid'] * 100:.2f}%",
            )
            col2.metric(
                "Normal probability",
                f"{result['probability_normal'] * 100:.2f}%",
            )

            st.progress(
                min(max(float(result["confidence"]), 0.0), 1.0),
                text=f"Prediction confidence: {confidence_pct:.2f}%",
            )

            with st.expander("How this prediction is calculated"):
                st.write(
                    "The app uses the same 224×224 resizing and ImageNet "
                    "normalization used by the notebook's validation pipeline. "
                    "The model returns one logit, which is converted to a "
                    "probability with a sigmoid function."
                )
                st.write(
                    "Because the notebook's class order is `covid = 0` and "
                    "`normal = 1`, sigmoid values at or above the stored "
                    "threshold are classified as Normal; lower values are "
                    "classified as COVID."
                )

        except Exception as exc:
            st.error("Prediction failed.")
            st.code(str(exc))


st.divider()
st.caption(
    "ResNet18 chest X-ray classification demo • "
    "Not intended for clinical diagnosis"
)
