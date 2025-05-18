import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.set_page_config(page_title="🌦️ Weather Classification 🌤️", layout="centered")

@st.cache_resource
def load_model():
    return tf.keras.models.load_model('weather.keras')

model = load_model()
class_names = ['Rainy', 'Cloudy', 'Sunshine', 'Sunrise']

st.title("🌦️ Weather Image Classifier")
st.write("Upload an image to classify the current weather condition.")

file = st.file_uploader("📸 Upload a weather image", type=["jpg", "jpeg", "png", "bmp"])

def import_and_predict(image_data, model):
    size = model.input_shape[1:3]  # Resize to match model input
    image = ImageOps.fit(image_data, size, Image.LANCZOS)
    img_array = np.asarray(image).astype('float32') / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    return prediction

if file is None:
    st.info("🖼️ Please upload a weather image to proceed.")
else:
    image = Image.open(file).convert('RGB')
    st.image(image, caption="Uploaded Image", use_container_width=True)

    with st.spinner("🔍 Classifying weather..."):
        prediction = import_and_predict(image, model)

    predicted_index = int(np.argmax(prediction))
    predicted_class = class_names[predicted_index]
    confidence = round(100 * np.max(prediction), 2)

    st.markdown(f"### 🌈 Predicted Weather: **{predicted_class}**")

    if predicted_class == 'Rainy':
        st.warning("🌧️ It's likely raining — don't forget your umbrella!")
    elif predicted_class == 'Sunshine':
        st.success("☀️ Bright and sunny day ahead!")
    elif predicted_class == 'Cloudy':
        st.info("☁️ It's cloudy — keep an eye on the sky.")
    elif predicted_class == 'Sunrise':
        st.info("🌅 A beautiful sunrise detected.")
