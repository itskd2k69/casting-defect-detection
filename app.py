# import streamlit as st
# import numpy as np
# import tensorflow as tf
# from PIL import Image
# import cv2

# # App title
# st.title("🧠 Casting Defect Detection")
# st.markdown("Upload a casting image to predict whether it is **OK** or **Defective**.")

# # Load model
# @st.cache_resource
# def load_model():
#     return tf.keras.models.load_model("casting_defect_model.h5")

# model = load_model()
# img_size = 128  # Image size used during training

# # File uploader
# uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

# if uploaded_file is not None:
#     # Read image
#     image = Image.open(uploaded_file).convert("L")  # Convert to grayscale
#     st.image(image, caption="Uploaded Image", use_column_width=True)

#     # Preprocess image
#     img_array = np.array(image)
#     img_resized = cv2.resize(img_array, (img_size, img_size))
#     img_normalized = img_resized / 255.0
#     img_reshaped = img_normalized.reshape(1, img_size, img_size, 1)

#     # Predict
#     prediction = model.predict(img_reshaped)
#     predicted_class = np.argmax(prediction, axis=1)[0]
#     confidence = prediction[0][predicted_class]

#     # Result
#     labels = ['OK', 'Defective']
#     result_label = labels[predicted_class]
#     st.subheader(f"🧾 Prediction: **{result_label}**")
#     st.write(f"🔍 Confidence: `{confidence * 100:.2f}%`")

import streamlit as st
import numpy as np
import tensorflow as tf
import cv2
from PIL import Image
import base64

# Set page config
st.set_page_config(page_title="Casting Defect Detector", page_icon="🧠", layout="centered")

# Load the model once
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("casting_defect_model.h5")

model = load_model()
img_size = 128

# Header section
st.markdown("""
    <style>
    .main-title {
        font-size: 40px;
        text-align: center;
        color: #4CAF50;
    }
    .footer {
        margin-top: 50px;
        text-align: center;
        font-size: 13px;
        color: gray;
    }
    .prediction-box {
        border: 2px solid #ddd;
        padding: 20px;
        border-radius: 12px;
        background-color: #f8f9fa;
        text-align: center;
    }
    </style>
    <h1 class="main-title">🧠 Casting Defect Detection</h1>
    <p style='text-align:center;'>Upload a casting image to detect if it's <strong>OK</strong> or <strong>Defective</strong>.</p>
""", unsafe_allow_html=True)

# Image uploader
uploaded_file = st.file_uploader("📤 Upload Image", type=["png", "jpg", "jpeg"], label_visibility="collapsed")

if uploaded_file:
    col1, col2 = st.columns(2)

    with col1:
        st.image(uploaded_file, caption="📸 Uploaded Image", use_column_width=True)

    # Preprocess the image
    image = Image.open(uploaded_file).convert("L")
    img_array = np.array(image)
    img_resized = cv2.resize(img_array, (img_size, img_size))
    img_normalized = img_resized / 255.0
    img_reshaped = img_normalized.reshape(1, img_size, img_size, 1)

    # Predict
    prediction = model.predict(img_reshaped)[0]
    predicted_class = np.argmax(prediction)
    confidence = prediction[predicted_class] * 100
    label = "OK ✅" if predicted_class == 0 else "Defective ❌"
    color = "green" if predicted_class == 0 else "red"

    with col2:
        st.markdown(f"""
            <div class="prediction-box">
                <h3 style='color:{color};'>🧾 Prediction: {label}</h3>
                <p>Confidence:</p>
                <progress value="{confidence}" max="100" style="width: 100%; height: 20px;"></progress>
                <p><strong>{confidence:.2f}%</strong> sure</p>
            </div>
        """, unsafe_allow_html=True)

else:
    st.markdown("<p style='text-align:center;'>⬆️ Upload a grayscale image of a casting part to begin analysis.</p>", unsafe_allow_html=True)

# Footer
st.markdown("<div class='footer'>Built with ❤️ using Streamlit & TensorFlow</div>", unsafe_allow_html=True)
