import streamlit as st
from PIL import Image
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import os

# Set page configuration
st.set_page_config(
    page_title="Pneumonia Detector",
    page_icon="🩺",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Load the model
# Get the absolute path of the current file
current_dir = os.path.dirname(os.path.abspath(__file__))

# Build the absolute path to the model file
model_path = os.path.join(current_dir, 'models', 'vgg16_pneumonia.h5')

# load model
model = load_model(model_path)

# Set a stylish header
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>Pneumonia Detector 🩺</h1>", unsafe_allow_html=True)

# Introduction text
st.write("Welcome! Upload a chest X-ray image, and our AI model will predict whether it shows signs of **Pneumonia** or is **Normal**.")

# File uploader
uploaded_file = st.file_uploader("Choose an X-ray image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    st.write("### File uploaded successfully! 📂")
    
        # Display the uploaded image
    image = Image.open(uploaded_file).convert("RGB")  # Ensure the image has 3 channels (RGB)
    st.image(image, caption='Uploaded Image.', width=300)
    st.write("### Image displayed successfully!")
    
    # Image preprocessing and prediction
    try:
        with st.spinner('Processing image... 🌀'):
            # Convert image to a numpy array and resize to (224, 224)
            image = np.array(image)
            image = tf.image.resize(image, (224, 224))
            
            # Ensure the image is scaled correctly for model input
            image = np.expand_dims(image / 255.0, axis=0)  # Expand dimensions to add batch size

            # Model prediction
            y_pred = model.predict(image)
            y_pred_prob = y_pred[0][0]
            y_pred = y_pred_prob > 0.5
            
            # Show prediction result
            if y_pred == 0:
                pred = 'Normal 🟢'
            else:
                pred = 'Pneumonia 🔴'

            # Display result with confidence score
            st.success(f"Our model predicts: **{pred}** with a confidence of {y_pred_prob*100:.2f}%")
    except Exception as e:
        st.error("Error during image processing or prediction:")
        st.write(e)
else:
    st.warning("Please upload an X-ray image to proceed.")
