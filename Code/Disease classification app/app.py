import os
import json
import numpy as np
import streamlit as st
from PIL import Image
import tensorflow as tf
from tensorflow.keras.models import load_model

# === Configuration ===
MODEL_DIR = "trained_model"  # adjust this if needed

# Load all models
@st.cache_resource
def load_models():
    xception = load_model(os.path.join(MODEL_DIR, "Xception_best.keras"))
    densenet = load_model(os.path.join(MODEL_DIR, "DenseNet_best.keras"))
    deepplantnet = load_model(os.path.join(MODEL_DIR, "DeepPlantNet_best.keras"))
    return [xception, densenet, deepplantnet]

# Load class label map
@st.cache_data
def load_class_indices():
    with open(os.path.join(MODEL_DIR, "class_indices.json")) as f:
        class_indices = json.load(f)
    return {int(k): v for k, v in class_indices.items()}

# Preprocess image
def load_and_preprocess_image(image_file, target_size=(256, 256)):
    img = Image.open(image_file).convert("RGB")
    img = img.resize(target_size)
    img_array = np.array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = img_array.astype('float32') / 255.
    return img_array

# Ensemble prediction function
def ensemble_predict(models, image_tensor):
    predictions = [model.predict(image_tensor) for model in models]
    avg_prediction = np.mean(predictions, axis=0)
    predicted_class_index = np.argmax(avg_prediction, axis=1)[0]
    # confidence = np.max(avg_prediction)  # <-- Removed use
    return predicted_class_index

# === Streamlit UI ===
st.title("🌿 Plant Leaf Disease Classifier (Ensemble Model)")

uploaded_image = st.file_uploader("📤 Upload a leaf image (JPG/PNG)...", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    # Show uploaded image
    st.image(uploaded_image, caption="Uploaded Image", width=200)

    if st.button("🔍 Classify"):
        with st.spinner("Predicting..."):
            models = load_models()
            class_labels = load_class_indices()
            img_tensor = load_and_preprocess_image(uploaded_image)

            class_idx = ensemble_predict(models, img_tensor)
            predicted_label = class_labels[class_idx]

        st.success(f"✅ **Prediction**: `{predicted_label}`")
