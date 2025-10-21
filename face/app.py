import streamlit as st
import mediapipe as mp
import numpy as np
from PIL import Image, ImageDraw
from io import BytesIO
import base64

# ---- Page Config ----
st.set_page_config(page_title="Face Detection App", page_icon="🧠", layout="wide")

# ---- Custom CSS ----
st.markdown("""
<style>
.stApp { background: linear-gradient(135deg,#141E30,#243B55); color:white; }
.title{text-align:center;font-size:2.5em;font-weight:bold;margin-bottom:0.3em;}
.subtitle{text-align:center;font-size:1.1em;color:#b0bec5;margin-bottom:2em;}
.footer{text-align:center;color:#90caf9;margin-top:3em;font-size:0.9em;}
button{background-color:#1E88E5;color:white;padding:8px 20px;border:none;border-radius:5px;}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='title'>🧠 Face Detection Web App</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Detect faces from images using MediaPipe (Python 3.13 compatible)</div>", unsafe_allow_html=True)

st.sidebar.header("⚙️ Controls")
st.sidebar.info("Upload an image or use webcam. This version avoids OpenCV import errors.")

# ---- MediaPipe Face Detection ----
mp_face = mp.solutions.face_detection

def detect_faces_pillow(image):
    img_rgb = np.array(image.convert("RGB"))
    h, w, _ = img_rgb.shape
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(img_rgb)
        draw = ImageDraw.Draw(image)
        count = 0
        if results.detections:
            for det in results.detections:
                box = det.location_data.relative_bounding_box
                xmin, ymin = int(box.xmin * w), int(box.ymin * h)
                width, height = int(box.width * w), int(box.height * h)
                draw.rectangle([xmin, ymin, xmin + width, ymin + height], outline=(46,204,113), width=4)
                count += 1
        return image, count

def download_button(image, filename="face_detected.png"):
    buf = BytesIO()
    image.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}"><button>💾 Download Processed Image</button></a>'
    return href

st.markdown("### 🖼️ Upload an Image")
uploaded = st.file_uploader("Upload a JPG/PNG image:", type=["jpg","jpeg","png"])

if uploaded:
    image = Image.open(uploaded)
    result_img, faces = detect_faces_pillow(image.copy())
    st.success(f"✅ Detected {faces} face(s) in the image.")
    st.image(result_img, use_container_width=True)
    st.markdown(download_button(result_img), unsafe_allow_html=True)

st.markdown("""
<div class='footer'>
Made with ❤️ using <b>Python</b>, <b>MediaPipe</b> & <b>Streamlit</b><br>
Developed by <b>Kmpl Youth</b>
</div>
""", unsafe_allow_html=True)
