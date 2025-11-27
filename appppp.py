
import streamlit as st
import os
from validation import validate_image
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from load_member2_model import load_member2_model

st.set_page_config(page_title="Brain Tumor Detection", layout="centered")
st.title("Brain Tumor MRI Classification")

uploaded_file = st.file_uploader("Choose MRI image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    file_path = "temp_upload.jpg"
    with open(file_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Member 3: Validation
    is_valid, message = validate_image(file_path)
    if not is_valid:
        st.error(f"REJECTED: {message}")
        st.stop()

    st.success("Valid Brain MRI! Analyzing...")

    # Load Member 2's model
    with st.spinner("Loading AI model..."):
        model = load_member2_model()

    # Preprocess: force RGB (3 channels)
    img = load_img(file_path, target_size=(224, 224), color_mode="rgb")
    img_array = img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    with st.spinner("Predicting..."):
        pred = model.predict(img_array, verbose=0)
        classes = ['glioma', 'meningioma', 'notumor', 'pituitary']
        result = classes[np.argmax(pred)]
        confidence = float(np.max(pred) * 100)

    # Display result
    st.subheader(f"**PREDICTION: {result.upper()}**")
    st.progress(confidence / 100)
    st.write(f"**Confidence: {confidence:.2f}%**")
    

    # Cleanup
    if os.path.exists(file_path):
        os.remove(file_path)

st.caption("Team Project: Member 1 • Member 2 • Member 3")
