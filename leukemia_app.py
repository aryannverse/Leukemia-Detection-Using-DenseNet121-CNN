import os
import json
import shutil
import tempfile
import zipfile

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns
import streamlit as st
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.densenet import preprocess_input as densenet_preprocess

st.set_page_config(
    page_title="Leukemia Classification Dashboard",
    page_icon="🩸",
    layout="wide"
)

IMG_SIZE = (224, 224)
CLASS_NAMES = ["all", "hem"]
LOCAL_MODEL_PATH = "Models/densenet121-trained.keras"


def _strip_key_recursive(obj, key_to_remove):
    if isinstance(obj, dict):
        return {
            k: _strip_key_recursive(v, key_to_remove)
            for k, v in obj.items()
            if k != key_to_remove
        }
    if isinstance(obj, list):
        return [_strip_key_recursive(item, key_to_remove) for item in obj]
    return obj


def _build_compat_keras_archive(model_path: str) -> str:
    temp_dir = tempfile.mkdtemp(prefix="keras_compat_")
    with zipfile.ZipFile(model_path, "r") as zin:
        zin.extractall(temp_dir)

    config_path = os.path.join(temp_dir, "config.json")
    if os.path.exists(config_path):
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)
        config = _strip_key_recursive(config, "quantization_config")
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config, f)

    compat_model_path = os.path.join(temp_dir, "compat_model.keras")
    with zipfile.ZipFile(compat_model_path, "w", zipfile.ZIP_DEFLATED) as zout:
        for root, _, files in os.walk(temp_dir):
            for filename in files:
                full_path = os.path.join(root, filename)
                if full_path == compat_model_path:
                    continue
                arcname = os.path.relpath(full_path, temp_dir)
                zout.write(full_path, arcname)

    return compat_model_path


@st.cache_resource
def load_model_local(local_path: str):
    if not os.path.exists(local_path):
        st.error(f"Model file not found locally: {local_path}")
        return None
    try:
        return tf.keras.models.load_model(local_path, compile=False)
    except Exception as e:
        if "quantization_config" in str(e):
            try:
                compat_path = _build_compat_keras_archive(local_path)
                model = tf.keras.models.load_model(compat_path, compile=False)
                shutil.rmtree(os.path.dirname(compat_path), ignore_errors=True)
                st.warning("Loaded model in compatibility mode (removed unsupported quantization_config fields).")
                return model
            except Exception as compat_e:
                st.error(f"Error loading model from file: {e}\nCompatibility load failed: {compat_e}")
                return None
        st.error(f"Error loading model from file: {e}")
        return None


def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize(IMG_SIZE)
    img_array = np.array(image).astype("float32")

    if img_array.ndim == 2:
        img_array = np.stack([img_array] * 3, axis=-1)
    if img_array.shape[-1] == 4:
        img_array = img_array[..., :3]

    img_array = np.expand_dims(img_array, axis=0)
    img_array = densenet_preprocess(img_array)
    return img_array


st.sidebar.title("🩸 C-NMC Classification")
st.sidebar.info("Acute Lymphoblastic Leukemia Detection System")

app_mode = st.sidebar.radio(
    "Navigation",
    ["Project Overview", "Evaluation Metrics", "Live Prediction"]
)

if app_mode == "Project Overview":
    st.title("Project Overview")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ### About the Model
        This application utilizes a Deep Learning model based on **DenseNet121** to classify Leukemia from microscopic blood cell images.

        **Model Architecture:**
        * **Base:** DenseNet121 (ImageNet pre-trained)
        * **Custom Head:**
            * Batch Normalization
            * Dense (256 units, ReLU, L2 Regularization)
            * Dropout (0.40)
            * Output Dense (2 units, Softmax)

        **Optimizer:** Adam (LR=0.0001)
        """)

    with col2:
        st.markdown("### Dataset Distribution (Test Set)")
        data = {
            "Class": ["ALL (Leukemia)", "HEM (Normal)"],
            "Count": [1091, 509],
        }
        df_dist = pd.DataFrame(data)

        fig = px.pie(
            df_dist,
            values="Count",
            names="Class",
            title="Class Distribution in Test Data",
            color_discrete_sequence=["#ff6b6b", "#4ecdc4"],
        )
        st.plotly_chart(fig, use_container_width=True)

elif app_mode == "Evaluation Metrics":
    st.title("Model Evaluation Metrics")
    st.caption("Update these curves after retraining DenseNet121 in the notebook.")

    st.markdown("### Training History")

    epochs = list(range(1, 20))
    train_acc = [82.485, 88.756, 91.088, 92.428, 93.648, 94.184, 94.666, 95.417, 97.293,
                 97.856, 98.258, 98.459, 99.049, 98.847, 99.317, 99.558, 99.223, 99.571, 99.611]
    val_acc = [70.169, 73.671, 88.993, 86.116, 88.430, 87.805, 93.183, 91.307, 93.684,
               95.122, 94.371, 94.809, 95.810, 95.872, 95.935, 96.185, 96.248, 95.872, 96.060]

    train_loss = [5.135, 2.546, 1.479, 0.888, 0.560, 0.392, 0.296, 0.237, 0.185,
                  0.164, 0.145, 0.132, 0.118, 0.118, 0.107, 0.099, 0.104, 0.092, 0.090]
    val_loss = [3.637, 2.381, 1.149, 0.864, 0.546, 0.437, 0.296, 0.303, 0.253,
                0.223, 0.229, 0.221, 0.201, 0.194, 0.193, 0.180, 0.181, 0.183, 0.180]

    tab1, tab2, tab3 = st.tabs(["Accuracy Plot", "Loss Plot", "Confusion Matrix"])

    with tab1:
        fig_acc = go.Figure()
        fig_acc.add_trace(go.Scatter(x=epochs, y=train_acc, mode="lines+markers", name="Train Accuracy"))
        fig_acc.add_trace(go.Scatter(x=epochs, y=val_acc, mode="lines+markers", name="Validation Accuracy"))
        fig_acc.update_layout(title="Accuracy over Epochs", xaxis_title="Epoch", yaxis_title="Accuracy (%)")
        st.plotly_chart(fig_acc, use_container_width=True)

    with tab2:
        fig_loss = go.Figure()
        fig_loss.add_trace(go.Scatter(x=epochs, y=train_loss, mode="lines+markers", name="Train Loss"))
        fig_loss.add_trace(go.Scatter(x=epochs, y=val_loss, mode="lines+markers", name="Validation Loss"))
        fig_loss.update_layout(title="Loss over Epochs", xaxis_title="Epoch", yaxis_title="Loss")
        st.plotly_chart(fig_loss, use_container_width=True)

    with tab3:
        cm_data = [[1068, 23], [49, 460]]

        fig_cm = plt.figure(figsize=(8, 6))
        sns.heatmap(cm_data, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
        plt.xlabel("Predicted")
        plt.ylabel("Actual")
        plt.title("Confusion Matrix (Test Data)")
        st.pyplot(fig_cm)

        st.metric(label="Overall Test Accuracy", value="95.5%")

elif app_mode == "Live Prediction":
    st.title("Live Model Prediction")

    model = load_model_local(LOCAL_MODEL_PATH)

    if model is not None:
        st.markdown("### Upload Cell Image")
        uploaded_file = st.file_uploader("Choose a microscopic image...", type=["jpg", "png", "jpeg", "bmp"])

        if uploaded_file is not None:
            col1, col2 = st.columns(2)

            with col1:
                image = Image.open(uploaded_file)
                st.image(image, caption="Uploaded Image", use_column_width=True)

            with col2:
                st.markdown("### Analysis Results")
                if st.button("Predict Classification"):
                    with st.spinner("Analyzing cell structure..."):
                        processed_img = preprocess_image(image)
                        prediction = model.predict(processed_img, verbose=0)

                        if prediction.ndim == 1 or prediction.shape[-1] == 1:
                            pred_probs = np.array([1 - prediction.flatten()[0], prediction.flatten()[0]])
                        else:
                            pred_probs = prediction.flatten()

                        predicted_class_idx = int(np.argmax(pred_probs))
                        predicted_class = CLASS_NAMES[predicted_class_idx]
                        confidence = float(np.max(pred_probs)) * 100

                        if predicted_class == "all":
                            st.error("**Prediction: ALL (Leukemia)**")
                            st.markdown("Result: **Acute Lymphoblastic Leukemia** Detected")
                        else:
                            st.success("**Prediction: HEM (Normal)**")
                            st.markdown("Result: **Normal / Hemoglobin** Detected")

                        st.progress(min(max(int(confidence), 0), 100))
                        st.write(f"Model Confidence: **{confidence:.2f}%**")

                        with st.expander("See probability details"):
                            prob_df = pd.DataFrame([pred_probs], columns=[c.upper() for c in CLASS_NAMES])
                            st.dataframe(prob_df.style.format("{:.2%}"))
    else:
        st.error("Model could not be loaded. Please check that 'models/densenet121-trained.keras' is present in the app repository.")

st.markdown("---")
st.markdown("Created based on Leukemia Classification Jupyter Notebook | Source: Repo")
