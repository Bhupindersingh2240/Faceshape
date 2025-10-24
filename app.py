import streamlit as st
import numpy as np
import cv2
import tensorflow as tf
from keras.models import load_model
from sklearn.preprocessing import LabelEncoder




# --- Config ---
IMG_SIZE = (128, 128)
CLASS_NAMES = ['oval', 'Round', 'Square', 'Heart', 'Long']
CONFIDENCE_THRESHOLD = 0.5  # Change this if needed

# --- Spectacle Recommendations ---
spectacle_suggestions = {
    "Oval": ["Square", "Rectangle", "Geometric"],
    "Long": ["Tall frames", "Wayfarers", "Oversized"],
    "Round": ["Rectangle", "Angular", "Cat-eye"],
    "Square": ["Round", "Oval", "Rimless"],
    "Heart": ["Light-colored", "Rimless", "Bottom-heavy"],
}

# --- Load model and label encoder ---
model = load_model("simple_face_shape_model.h5")
label_encoder = LabelEncoder()
label_encoder.fit(CLASS_NAMES)

import cv2

def detect_face(image):
    face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    return len(faces) > 0

def predict_face_shape(image):
    if detect_face(image):
        # Proceed with face shape prediction
        shape = model.predict(image)
        return shape
    else:
        return "No face detected. Please upload a valid image."
# --- Preprocess uploaded or captured image ---
def preprocess_image(uploaded_file):
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, 1)
    if image is not None:
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_resized = cv2.resize(image_rgb, IMG_SIZE)
        image_normalized = image_resized.astype('float32') / 255.0
        return image_rgb, np.expand_dims(image_normalized, axis=0)
    return None, None

# --- Streamlit UI ---
st.title("📸 Face Shape & Spectacle Style Recommender")
st.write("Upload or capture a face image to predict the face shape and get perfect spectacle suggestions! 👓")

# === Upload or Camera input ===
upload_option = st.radio("Choose input method:", ["Upload Image", "Use Camera"])
image_rgb = None
image_input = None

if upload_option == "Upload Image":
    uploaded_file = st.file_uploader("Choose an image", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        image_rgb, image_input = preprocess_image(uploaded_file)

elif upload_option == "Use Camera":
    captured_image = st.camera_input("Take a picture")
    if captured_image:
        image_rgb, image_input = preprocess_image(captured_image)

# === If image is processed successfully ===
if image_input is not None:
    st.image(image_rgb, caption="Input Image", use_container_width=True)

    if not detect_face(image_rgb):
        st.sidebar.warning("No face detected. Please upload an image with a clear face.")
    else:
        prediction = model.predict(image_input)
        predicted_index = np.argmax(prediction)
        confidence = prediction[0][predicted_index]


        if confidence < CONFIDENCE_THRESHOLD:
            st.sidebar.warning("Face not detected. Please try again with a clearer image.")
        else:
            predicted_label = label_encoder.inverse_transform([predicted_index])
            st.sidebar.markdown(f"<h3 style='color:#7c4dff;'>🧠 Face Shape:</h3>", unsafe_allow_html=True)
            st.sidebar.markdown(f"<h2 style='color:#512da8'>{predicted_label.upper()}</h2>", unsafe_allow_html=True)
            st.sidebar.markdown(f"<p style='color:#6a1b9a'><b>Confidence:</b> {confidence:.2f}</p>", unsafe_allow_html=True)
            suggestions = spectacle_suggestions.get(predicted_label, [])
            if suggestions:
                st.sidebar.markdown("<h3 style='color:#7c4dff;'>👓 Spectacle Suggestions:</h3>", unsafe_allow_html=True)
                for style in suggestions:
                    st.sidebar.markdown(f"<li style='color:#5e35b1'>{style}</li>", unsafe_allow_html=True)
            else:
                st.sidebar.warning("No suggestion found for this face shape.")

