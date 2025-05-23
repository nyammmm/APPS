import streamlit as st
import numpy as np
import cv2
from tensorflow.keras.models import load_model
from PIL import Image
import tempfile
import os

# Streamlit app title
st.set_page_config(page_title="PPE Video Classifier", layout="centered")
st.title("🛠️ Construction Site Safety Classifier")

# Upload video
uploaded_file = st.file_uploader("Upload a construction site video", type=["mp4", "avi", "mov"])

if uploaded_file:
    st.video(uploaded_file)

    # Save uploaded video to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_path = tmp_file.name

    # Extract the first frame using OpenCV
    cap = cv2.VideoCapture(tmp_path)
    success, frame = cap.read()
    cap.release()
    os.remove(tmp_path)

    if success:
        # Resize and normalize
        image_size = (128, 128)  # match the model input
        sequence_length = 10     # your model expects a sequence
        frame = cv2.resize(frame, image_size)
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = frame.astype(np.float32) / 255.0

        # Create pseudo-sequence from single frame
        sequence = np.repeat(frame[np.newaxis, ...], sequence_length, axis=0)  # (10, 128, 128, 3)
        input_data = np.expand_dims(sequence, axis=0)  # (1, 10, 128, 128, 3)

        # Load model
        model = load_model("proj.h5")  # make sure model.h5 is in the same folder

        # Predict
        prediction = model.predict(input_data)[0][0]
        label = "✅ Safe" if prediction >= 0.5 else "✅ Safe"

        # Display result
        st.markdown("### Classification Result")
        st.success(f"This Construction Site is **{label}**")
    else:
        st.error("Failed to read the video. Please try a different file.")
