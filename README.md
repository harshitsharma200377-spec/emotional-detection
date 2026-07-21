# 😊 Smart Face Detection using OpenCV
## Live demo : https://emotional-detection-q5t8wjuxocntenxroum9y8.streamlit.app/

A real-time face detection application built with **Python** and **OpenCV** using Haar Cascade Classifiers.

This project detects:

- 😀 Face
- 👀 Eyes
- 😊 Smile
- 👓 Eye Detection using Eyeglasses Haar Cascade

The application captures live video from your webcam and performs real-time detection.

---

## 📸 Features

- Real-time webcam detection
- Face Detection
- Eye Detection
- Smile Detection
- Eye Detection using Eyeglasses Haar Cascade
- Bounding box around detected faces
- Lightweight and beginner-friendly

---

## 🛠️ Technologies Used

- Python 3
- OpenCV
- Haar Cascade Classifiers

---

## 📂 Project Structure

```
Smart-Face-Detection/
│
├── main.py
├── README.md
├── requirements.txt
│
└── face and object detection/
    ├── haarcascade_frontalface_default.xml
    ├── haarcascade_eye.xml
    ├── haarcascade_smile.xml
    └── haarcascade_eye_tree_eyeglasses.xml
```

---

## 🚀 Installation

### 1. Clone the repository

```bash
git clone https://github.com/yourusername/Smart-Face-Detection.git
```

### 2. Move into the project

```bash
cd Smart-Face-Detection
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
python main.py
```

Press **Q** to exit.

---

## 📦 Requirements

```text
opencv-python
numpy
```

---

## 📖 How It Works

1. Captures live webcam video.
2. Converts each frame to grayscale.
3. Detects faces using Haar Cascade.
4. Detects eyes within the detected face.
5. Detects smiles within the detected face.
6. Displays the detection results on the video stream.

---

## ⚠️ Limitations

- Haar Cascade smile detection is not an emotion detector.
- "No Smile" does **not** mean the person is sad.
- Eyeglasses Haar Cascade detects eyes and may also work when glasses are present, but it **cannot determine** whether a person is actually wearing glasses.
- Performance depends on lighting conditions and camera quality.

---

## 📸 Output

The application displays:

- Face Detection
- Eyes Detection
- Smile Detected / No Smile Detected

in real time using your webcam.

---

## 🔮 Future Improvements

- MediaPipe Face Mesh
- Blink Detection
- Emotion Detection
- Face Recognition
- Drowsiness Detection
- YOLO-based Face Detection
- Streamlit Web Application
- FPS Counter

---

## 🤝 Contributing

Contributions are welcome!

Feel free to fork the repository, improve the project, and submit a pull request.

---

## 📜 License

This project is licensed under the MIT License.

---

## 👨‍💻 Author

**Harshit Sharma**

GitHub: https://github.com/harshitsharma200377-spec
