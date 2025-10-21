import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
from PIL import Image
from io import BytesIO
import base64

# ---- Streamlit Page Config ----
st.set_page_config(
    page_title="Face Detection App",
    page_icon="🧠",
    layout="wide",
)

# ---- Custom CSS Styling ----
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #141E30, #243B55);
        color: white;
    }
    .title {
        text-align: center;
        font-size: 2.5em;
        font-weight: bold;
        margin-bottom: 0.3em;
    }
    .subtitle {
        text-align: center;
        font-size: 1.1em;
        color: #b0bec5;
        margin-bottom: 2em;
    }
    .footer {
        text-align: center;
        color: #90caf9;
        margin-top: 3em;
        font-size: 0.9em;
    }
    button[kind="primary"] {
        background-color: #1E88E5 !important;
        color: white !important;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Title and Description ----
st.markdown("<div class='title'>🧠 Face Detection Web App</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Detect faces from uploaded images or webcam feed using MediaPipe and Streamlit</div>", unsafe_allow_html=True)

# ---- Sidebar ----
st.sidebar.header("⚙️ Controls")
mode = st.sidebar.radio("Choose Input Mode:", ("📸 Upload Image", "🎥 Use Webcam"))
st.sidebar.markdown("---")
st.sidebar.info("You can switch between image upload and webcam anytime.")

# ---- MediaPipe Setup ----
mp_face = mp.solutions.face_detection
mp_draw = mp.solutions.drawing_utils

# ---- Helper Functions ----
def detect_faces_mediapipe(image):
    with mp_face.FaceDetection(model_selection=0, min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        if results.detections:
            for detection in results.detections:
                mp_draw.draw_detection(image, detection)
        count = len(results.detections) if results.detections else 0
    return image, count

def get_image_download_link(img, filename="face_detected.png", text="💾 Download Processed Image"):
    buffered = BytesIO()
    img.save(buffered, format="PNG")
    img_bytes = buffered.getvalue()
    b64 = base64.b64encode(img_bytes).decode()
    href = f'<a href="data:file/png;base64,{b64}" download="{filename}" style="text-decoration:none;"><button style="background-color:#1E88E5;color:white;padding:8px 20px;border:none;border-radius:5px;">{text}</button></a>'
    return href

# ---- Option 1: Upload Image ----
if mode == "📸 Upload Image":
    st.markdown("### 🖼️ Upload an Image")
    uploaded_file = st.file_uploader("Upload a clear image containing faces:", type=["jpg", "jpeg", "png"])

    if uploaded_file:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        image = cv2.imdecode(file_bytes, 1)
        result_img, faces = detect_faces_mediapipe(image)

        st.success(f"✅ Detected {faces} face(s) in the image.")
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        # Convert result to PIL for download
        result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        st.markdown(get_image_download_link(result_pil), unsafe_allow_html=True)

# ---- Option 2: Webcam ----
elif mode == "🎥 Use Webcam":
    st.markdown("### 🎥 Live Webcam Face Detection")
    run = st.checkbox("Start Camera")

    FRAME_WINDOW = st.image([])

    camera = cv2.VideoCapture(0)

    while run:
        ret, frame = camera.read()
        if not ret:
            st.error("Unable to access the webcam. Please check permissions.")
            break
        frame, faces = detect_faces_mediapipe(frame)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    else:
        camera.release()
        st.warning("🛑 Webcam stopped.")

# ---- Footer ----
st.markdown("""
<div class="footer">
Made with ❤️ using <b>Python</b>, <b>MediaPipe</b> & <b>Streamlit</b><br>
Developed by <b>Kmpl Youth</b>
</div>
""", unsafe_allow_html=True)
