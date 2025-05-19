import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
from PIL import Image

# -------------------- Configuration --------------------
MODEL_PATH = "proj.h5"  # 🔁 Replace with your actual model path
IMAGE_SIZE = (128, 128)       # 🔁 Match your model input size

# Load trained model
@st.cache_resource
def load_trained_model():
    model = load_model(MODEL_PATH)
    return model

model = load_trained_model()

# Preprocess single image
def preprocess_image(image):
    image = cv2.resize(image, IMAGE_SIZE)
    image = image.astype("float32") / 255.0
    image = np.expand_dims(image, axis=0)
    return image

# Extract first frame from video
def get_first_frame(video_path):
    cap = cv2.VideoCapture(video_path)
    ret, frame = cap.read()
    cap.release()
    if ret:
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    else:
        return None

# -------------------- Streamlit UI --------------------
st.set_page_config(page_title="PPE Safety Classifier", layout="centered")
st.title("🏗️ Construction Site Safety Classifier using CNN")
st.markdown("Upload a video — we'll classify the video if **Safe** or **Unsafe**.")

video_file = st.file_uploader("📤 Upload a Video", type=["mp4", "avi", "mov"])

if video_file is not None:
    with open("temp_video.mp4", "wb") as f:
        f.write(video_file.read())
    
    st.video("temp_video.mp4")
    st.info("Extracting the first frame...")

    first_frame = get_first_frame("temp_video.mp4")
    if first_frame is not None:
        st.image(first_frame, caption="🖼️ First Frame", use_column_width=True)

        # Predict
        preprocessed = preprocess_image(first_frame)
        prediction = model.predict(preprocessed)[0][0]
        label = "Safe ✅" if prediction > 0.5 else "Unsafe ❌"

        st.subheader(f"Prediction: **{label}**")
    else:
        st.error("⚠️ Couldn't read the video. Please try another file.")
