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

# Custom CSS for UI styling
st.markdown(
    """
    <style>
    .main {background-color: #1E293B; color: white;}
    h1, h2, h3, h4 {color: #3B82F6;}
    .uploadedFile {text-align: center;}
    .stButton>button {background-color: #3B82F6; color: white;}
    .stButton>button:hover {background-color: #2563EB;}
    </style>
    """,
    unsafe_allow_html=True,
)

# Caching function for performance

def convert_to_array(image):
    return np.array(image)

# Image Processing Functions
def apply_blur(image):
    """Apply Gaussian Blur to the image"""
    try:
        return cv2.GaussianBlur(image, (5, 5), 0)
    except Exception as e:
        st.error(f"Error applying blur: {e}")
        return image

def apply_sharpen(image):
    """Apply sharpen filter to the image"""
    try:
        kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
        return cv2.filter2D(image, -1, kernel)
    except Exception as e:
        st.error(f"Error applying sharpen: {e}")
        return image

def apply_edge_detection(image):
    """Apply edge detection using Canny algorithm"""
    try:
        return cv2.Canny(image, 100, 200)
    except Exception as e:
        st.error(f"Error detecting edges: {e}")
        return image

def apply_thresholding(image, threshold):
    """Apply thresholding to the image"""
    try:
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, binary_image = cv2.threshold(gray_image, threshold, 255, cv2.THRESH_BINARY)
        return binary_image
    except Exception as e:
        st.error(f"Error applying thresholding: {e}")
        return image

def adjust_brightness(image, value):
    """Adjust brightness of the image"""
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = np.clip(v + value, 0, 255)
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    except Exception as e:
        st.error(f"Error adjusting brightness: {e}")
        return image

def adjust_contrast(image, alpha):
    """Adjust contrast of the image"""
    try:
        return cv2.convertScaleAbs(image, alpha=alpha, beta=0)
    except Exception as e:
        st.error(f"Error adjusting contrast: {e}")
        return image

def adjust_saturation(image, value):
    """Adjust saturation of the image"""
    try:
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        s = np.clip(s + value, 0, 255)
        final_hsv = cv2.merge((h, s, v))
        return cv2.cvtColor(final_hsv, cv2.COLOR_HSV2BGR)
    except Exception as e:
        st.error(f"Error adjusting saturation: {e}")
        return image

# Streamlit App Header
st.title("📸 Image Processing App")
st.markdown("Upload an image and apply various processing techniques to enhance it.")

# Upload Image
uploaded_image = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"], key="fileUploader")

if uploaded_image:
    try:
        image = Image.open(uploaded_image)
        image_np = convert_to_array(image)
        st.image(image, caption='Uploaded Image', use_container_width=True)  # Updated here
        
        # Processing Option Selection
        option = st.selectbox(
            "Choose an Image Processing Technique",
            ["None", "Blur", "Sharpen", "Edge Detection", "Thresholding", "Brightness", "Contrast", "Saturation"]
        )
        
        processed_image = None
        if option == "Blur":
            processed_image = apply_blur(image_np)
        elif option == "Sharpen":
            processed_image = apply_sharpen(image_np)
        elif option == "Edge Detection":
            processed_image = apply_edge_detection(image_np)
        elif option == "Thresholding":
            threshold = st.slider("Threshold", 0, 255, 127)
            processed_image = apply_thresholding(image_np, threshold)
        elif option == "Brightness":
            value = st.slider("Brightness Adjustment", -100, 100, 30)
            processed_image = adjust_brightness(image_np, value)
        elif option == "Contrast":
            alpha = st.slider("Contrast Level", 0.0, 3.0, 1.5)
            processed_image = adjust_contrast(image_np, alpha)
        elif option == "Saturation":
            value = st.slider("Saturation Adjustment", -100, 100, 30)
            processed_image = adjust_saturation(image_np, value)
        
        if processed_image is not None:
            st.image(processed_image, caption='Processed Image', use_container_width=True)  # Updated here
            st.download_button(
                label="Download Processed Image",
                data=cv2.imencode(".png", processed_image)[1].tobytes(),
                file_name="processed_image.png",
                mime="image/png"
            )
    except Exception as e:
        st.error(f"An error occurred while processing the image: {e}")
