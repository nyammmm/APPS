import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

# Load model with caching to prevent reloading on every change
@st.cache_resource
def load_model():
    return tf.keras.models.load_model('final_vgg16_model.keras')

model = load_model()
class_names = ['Safe', 'Violation']

st.title("🔥 PPE Safety Detection System")
st.write("Upload an image to check for PPE safety violations (e.g., missing safety gear).")

file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png", "bmp"])

def import_and_predict(image_data, model):
    # Resize to the model's expected input size
    size = model.input_shape[1:3]
    image = ImageOps.fit(image_data, size, Image.ANTIALIAS)
    img_array = np.asarray(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

if file is None:
    st.info("Please upload an image file to proceed.")
else:
    image = Image.open(file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_column_width=True)

    prediction = import_and_predict(image, model)
    predicted_index = np.argmax(prediction)
    predicted_class = class_names[predicted_index]
    confidence = round(100 * np.max(prediction), 2)

    st.markdown(f"### 🔍 Prediction: **{predicted_class}**")
    st.markdown(f"### 📊 Confidence: **{confidence}%**")

    if predicted_class == 'Violation':
        st.error("⚠️ Safety Violation Detected!")
    else:
        st.success("✅ All Clear: Safe Environment")
