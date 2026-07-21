import streamlit as st
from streamlit_webrtc import (
    webrtc_streamer,
    VideoProcessorBase,
    RTCConfiguration
)
import av
import cv2
import os

st.set_page_config(page_title="Smart Face Detection", page_icon="😊")

st.title("😊 Smart Face Detection")
st.write("Real-time Face, Eye and Smile Detection using OpenCV Haar Cascades")

# ------------------ Load Haar Cascade Files ------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CASCADE_DIR = os.path.join(BASE_DIR, "haarcascades")

face_cacade = cv2.CascadeClassifier(
    os.path.join(CASCADE_DIR, "haarcascade_frontalface_default.xml")
)

eyes_cascade = cv2.CascadeClassifier(
    os.path.join(CASCADE_DIR, "haarcascade_eye.xml")
)

smile_cascade = cv2.CascadeClassifier(
    os.path.join(CASCADE_DIR, "haarcascade_smile.xml")
)

eye_glasses_cascade = cv2.CascadeClassifier(
    os.path.join(CASCADE_DIR, "haarcascade_eye_tree_eyeglasses.xml")
)

# ------------------ Check XML Files ------------------

if face_cacade.empty():
    st.error("❌ Face Cascade XML not found.")
    st.stop()

if eyes_cascade.empty():
    st.error("❌ Eye Cascade XML not found.")
    st.stop()

if smile_cascade.empty():
    st.error("❌ Smile Cascade XML not found.")
    st.stop()

if eye_glasses_cascade.empty():
    st.error("❌ Eye Glasses Cascade XML not found.")
    st.stop()

# ------------------ Video Processor ------------------

class VideoProcessor(VideoProcessorBase):

    def recv(self, frame):

        frame = frame.to_ndarray(format="bgr24")

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = face_cacade.detectMultiScale(gray, 1.1, 5)

        for (x, y, w, h) in faces:

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            roi_gray = gray[y:y+h, x:x+w]

            # Eye Detection
            eyes = eyes_cascade.detectMultiScale(roi_gray, 1.1, 10)

            if len(eyes) > 0:
                cv2.putText(
                    frame,
                    "Eyes Detection",
                    (x, y - 30),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            # Eye Glasses Detection
            eyes_glasses = eye_glasses_cascade.detectMultiScale(
                roi_gray,
                1.1,
                10
            )

            if len(eyes_glasses) > 0:
                cv2.putText(
                    frame,
                    "Eyes (Without Glasses)",
                    (x, y - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 0, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "Eyes (With Glasses)",
                    (x, y - 50),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )

            # Smile Detection
            smiles = smile_cascade.detectMultiScale(
                roi_gray,
                1.3,
                minNeighbors=35,
                minSize=(30, 30)
            )

            if len(smiles) > 0:
                cv2.putText(
                    frame,
                    "Smile Detected",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 255, 0),
                    2
                )
            else:
                cv2.putText(
                    frame,
                    "No Smile Detected",
                    (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (0, 0, 255),
                    2
                )

        return av.VideoFrame.from_ndarray(frame, format="bgr24")


# ------------------ WebRTC Configuration ------------------

RTC_CONFIGURATION = RTCConfiguration(
    {
        "iceServers": [
            {"urls": ["stun:stun.l.google.com:19302"]}
        ]
    }
)

# ------------------ Streamlit WebRTC ------------------

webrtc_streamer(
    key="smart-face-detection",
    video_processor_factory=VideoProcessor,
    media_stream_constraints={
        "video": True,
        "audio": False
    },
    rtc_configuration=RTC_CONFIGURATION
)
