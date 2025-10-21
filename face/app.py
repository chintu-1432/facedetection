import cv2
import streamlit as st
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

# ---- Custom CSS for UI/UX ----
st.markdown("""
    <style>
    body {
        background-color: #0e1117;
        color: #FAFAFA;
    }
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
        font-size: 1.2em;
        color: #b0bec5;
        margin-bottom: 2em;
    }
    .footer {
        text-align: center;
        color: #90caf9;
        margin-top: 3em;
        font-size: 0.9em;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Title and Description ----
st.markdown("<div class='title'>🧠 Face Detection Web App</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Detect faces from uploaded images or live webcam feed using OpenCV and Streamlit</div>", unsafe_allow_html=True)

# ---- Sidebar Options ----
st.sidebar.header("⚙️ Controls")
mode = st.sidebar.radio("Choose Input Mode:", ("📸 Upload Image", "🎥 Use Webcam"))
st.sidebar.markdown("---")
st.sidebar.info("Tip: You can switch between image upload and webcam mode anytime.")

# ---- Load the Face Classifier ----
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# ---- Face Detection Function ----
def detect_faces(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    for (x, y, w, h) in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (46, 204, 113), 2)
        cv2.putText(image, "Face", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (46, 204, 113), 2)
    return image, len(faces)

# ---- Function to Create Download Link ----
def get_image_download_link(img, filename, text):
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
        result_img, faces = detect_faces(image)

        st.success(f"✅ Detected {faces} face(s) in the image.")
        st.image(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)

        # Convert result image for download
        result_pil = Image.fromarray(cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB))
        st.markdown(get_image_download_link(result_pil, "face_detected.png", "💾 Download Processed Image"), unsafe_allow_html=True)

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
        frame, faces = detect_faces(frame)
        FRAME_WINDOW.image(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), channels="RGB", use_container_width=True)
    else:
        camera.release()
        st.warning("🛑 Webcam stopped.")

# ---- Footer ----
st.markdown("""
<div class="footer">
Made with ❤️ using <b>Python</b>, <b>OpenCV</b> & <b>Streamlit</b><br>
Developed by <b>Kmpl Youth</b>
</div>
""", unsafe_allow_html=True)
