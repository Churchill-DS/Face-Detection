import streamlit as st
import cv2
import numpy as np
from PIL import Image
import tempfile
import os

st.set_page_config(page_title="Face Detection App", layout="centered")
st.title("📸 Face Detection App (Viola-Jones)")

# Instructions
st.markdown("""
### 🧾 Instructions:
- Upload an image or use your webcam.
- Adjust detection parameters (`scaleFactor`, `minNeighbors`).
- Choose the color for rectangles around detected faces.
- Download or save the processed image after detection.
""")

# Sidebar controls
st.sidebar.header("🔧 Detection Settings")

color = st.sidebar.color_picker("Pick Rectangle Color", "#00ff00")
bgr_color = tuple(int(color.lstrip("#")[i:i+2], 16) for i in (4, 2, 0))  # Convert to BGR

scaleFactor = st.sidebar.slider("Scale Factor", 1.1, 2.0, 1.1, 0.1)
minNeighbors = st.sidebar.slider("Min Neighbors", 1, 10, 5)

mode = st.radio("📷 Choose Mode", ["Upload Image", "Use Webcam"])

# Load Haar cascade
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def detect_faces(image_cv):
    gray = cv2.cvtColor(image_cv, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
    for (x, y, w, h) in faces:
        cv2.rectangle(image_cv, (x, y), (x + w, y + h), bgr_color, 2)
    return image_cv, len(faces)


if mode == "Upload Image":
    uploaded_file = st.file_uploader("📤 Upload an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image_cv = cv2.imdecode(file_bytes, 1)
        processed_image, face_count = detect_faces(image_cv)
        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption=f"Detected {face_count} face(s)", use_column_width=True)

        if st.checkbox("💾 Save and Download Image"):
            result_path = "detected_faces.jpg"
            cv2.imwrite(result_path, processed_image)
            with open(result_path, "rb") as file:
                st.download_button("📥 Download Image", file, file_name="detected_faces.jpg")

else:
    st.markdown("### Live Webcam (Capture and Detect)")
    picture = st.camera_input("Take a picture")

    if picture:
        img = Image.open(picture)
        image_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
        processed_image, face_count = detect_faces(image_cv)
        st.image(cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB), caption=f"Detected {face_count} face(s)", use_column_width=True)

        if st.checkbox("💾 Save and Download Webcam Image"):
            result_path = "webcam_faces.jpg"
            cv2.imwrite(result_path, processed_image)
            with open(result_path, "rb") as file:
                st.download_button("📥 Download Image", file, file_name="webcam_faces.jpg")
