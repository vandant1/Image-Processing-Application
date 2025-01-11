
#image_processing_application


import streamlit as st
import cv2
import numpy as np
from PIL import Image

# Set the page layout
st.set_page_config(
    page_title="Image Processing App",
    page_icon=":camera:",
    layout="centered",
    initial_sidebar_state="expanded",
)

# Style the app
st.markdown(
    """
    <style>
    .main {
        background-color: #000000;
        color: white;
    }
    h1, h2, h3, h4 {
        color: #1E3A8A;
    }
    .css-10trblm {
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# Helper functions
def apply_blur(image, ksize=(5, 5)):
    return cv2.GaussianBlur(np.array(image), ksize, 0)

def apply_sharpen(image):
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    return cv2.filter2D(np.array(image), -1, kernel)

def apply_edge_detection(image):
    return cv2.Canny(np.array(image), 100, 200)

def apply_thresholding(image, threshold=127):
    gray_image = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2GRAY)
    _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
    return binary_image

def adjust_brightness(image, value=30):
    hsv = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    v = cv2.add(v, value)
    v[v > 255] = 255
    v[v < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

def adjust_contrast(image, alpha=1.5):
    return cv2.convertScaleAbs(np.array(image), alpha=alpha, beta=0)

def adjust_saturation(image, value=30):
    hsv = cv2.cvtColor(np.array(image), cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)
    s = cv2.add(s, value)
    s[s > 255] = 255
    s[s < 0] = 0
    final_hsv = cv2.merge((h, s, v))
    return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)

# App title
st.title("Image Processing App")

# Upload image
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    image = Image.open(uploaded_image)
    st.image(image, caption='Uploaded Image', use_column_width=True)
    
    # Choose processing option
    option = st.selectbox(
        "Choose an Image Processing Technique",
        ["None", "Blur", "Sharpen", "Edge Detection", "Thresholding", "Brightness", "Contrast", "Saturation"]
    )

    processed_image = None

    if option == "Blur":
        processed_image = apply_blur(image)
    elif option == "Sharpen":
        processed_image = apply_sharpen(image)
    elif option == "Edge Detection":
        processed_image = apply_edge_detection(image)
    elif option == "Thresholding":
        threshold = st.slider("Threshold", 0, 255, 127)
        processed_image = apply_thresholding(image, threshold)
    elif option == "Brightness":
        value = st.slider("Brightness", -100, 100, 30)
        processed_image = adjust_brightness(image, value)
    elif option == "Contrast":
        alpha = st.slider("Contrast", 0.0, 3.0, 1.5)
        processed_image = adjust_contrast(image, alpha)
    elif option == "Saturation":
        value = st.slider("Saturation", -100, 100, 30)
        processed_image = adjust_saturation(image, value)

    if processed_image is not None:
        st.image(processed_image, caption='Processed Image', use_column_width=True)
