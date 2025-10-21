


📸 Overview

The Face Detection Web App uses Python, OpenCV, and Streamlit to detect human faces in both uploaded images and live webcam feeds.
It leverages OpenCV’s Haar Cascade Classifier for fast and accurate real-time face detection.

This project also features a modern UI/UX, a download button for saving processed images, and can easily be deployed to Streamlit Cloud or other platforms.

✨ Features

✅ Detect faces from uploaded images
✅ Real-time face detection via webcam
✅ Clean modern UI (dark gradient theme)
✅ Display number of detected faces
✅ Download processed image with bounding boxes
✅ Responsive and user-friendly interface
✅ Built using Python, OpenCV, and Streamlit

🧰 Tech Stack
Technology	Description
Python	Programming language
OpenCV	Image processing & face detection
Streamlit	Web app framework
NumPy	Image array manipulation
Pillow (PIL)	Image conversion for download
⚙️ Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/yourusername/face-detection-webapp.git
cd face-detection-webapp

2️⃣ Install dependencies
pip install streamlit opencv-python numpy pillow

3️⃣ Run the application
streamlit run app.py



💻 Usage

Select Upload Image or Use Webcam mode from the sidebar.

If you choose Upload Image, upload a photo (JPG, PNG, or JPEG).

The app detects faces and displays bounding boxes.

Click the 💾 Download Processed Image button to save your result.

For Webcam Mode, click Start Camera and watch detection happen live.

📂 Project Structure
face-detection-webapp/
│
├── app.py                  # Main Streamlit application
├── README.md               # Project documentation
├── requirements.txt        # (Optional) Dependency list
└── assets/                 # (Optional) Screenshots or icons

🧩 Code Snippet Example
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 5)
for (x, y, w, h) in faces:
    cv2.rectangle(image, (x, y), (x + w, y + h), (46, 204, 113), 2)

🖼️ Screenshot Preview
Upload Mode	Webcam Mode

	


☁️ Deployment

You can easily deploy this project on:

Streamlit Cloud

Render or Railway (with a Procfile)

Hugging Face Spaces

🤖 Future Enhancements

🚀 Add smile & eye detection
🎭 Integrate emotion recognition (DeepFace)
🪄 Add face blurring for privacy mode
📱 Mobile-friendly optimizations

🧑‍💻 Author

Developed by: Mallaiah Appaneni

📧 Feel free to fork, star ⭐, and contribute!
