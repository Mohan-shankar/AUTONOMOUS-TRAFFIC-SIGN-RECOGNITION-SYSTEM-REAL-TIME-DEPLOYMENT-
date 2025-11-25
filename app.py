import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import os

# -----------------------------
# Page Setup
# -----------------------------
st.set_page_config(page_title="Traffic Sign Recognition", layout="centered")

st.title("🚦 Autonomous Traffic Sign Recognition System")
st.subheader("Upload an image to predict the traffic sign")

# -----------------------------
# Load Class Names
# -----------------------------
# Load Class Names
try:
    CLASS_PATH = os.path.join(os.path.dirname(__file__), "class_names.json")
    with open(CLASS_PATH, "r") as f:
        CLASS_NAMES = json.load(f)
except Exception as e:
    CLASS_NAMES = None
    st.warning("⚠️ class_names.json not found — showing class index only.")

# -----------------------------
# Load Model
# -----------------------------
MODEL_PATH = "models/Auto_traffic_sign_model.h5"

@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# -----------------------------
# Preprocess Function
# -----------------------------
IMG_SIZE = (224, 224)

def preprocess_image(uploaded_file):
    img = Image.open(uploaded_file).convert("RGB")
    img = img.resize(IMG_SIZE)
    img = np.array(img) / 255.0
    img = np.expand_dims(img, axis=0)  # (1,224,224,3)
    return img


# -----------------------------
# UI — Upload File
# -----------------------------
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:

    # Show uploaded image
    display_img = Image.open(uploaded_file)
    st.image(display_img, caption="Uploaded Image", width=300)

    # Preprocess
    img = preprocess_image(uploaded_file)

    # Predict
    pred = model.predict(img)[0]  # shape: (43,)
    class_index = int(np.argmax(pred))
    confidence = float(pred[class_index] * 100)

    # -----------------------------
    # Output Section
    # -----------------------------
    st.subheader("🔍 Prediction Result")

    if CLASS_NAMES:
        st.write(f"### 🛑 Predicted Sign: **{CLASS_NAMES[str(class_index)]}**")
    else:
        st.write(f"### 🛑 Predicted Class ID: **{class_index}**")

    st.write(f"### 🎯 Confidence: **{confidence:.2f}%**")

    # Show probability bar chart
    st.subheader("📉 All Class Probabilities")
    st.bar_chart(pred)

