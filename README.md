# AirWriter ✨

A computer vision-powered application that lets users write letters in the air using hand gestures, and converts those air-traced characters into typed digital text using AI.

## 🧠 Core Idea

A futuristic, touchless typing experience that allows users to write invisible characters in the air and see them appear as text — no keyboard, no stylus, just gestures.

## 🧰 Tech Stack

| Component | Tech Used |
|-----------|-----------|
| Hand Tracking | Mediapipe (Hands module) |
| Image Processing | OpenCV |
| Gesture Logic | Python (custom logic for start/stop draw) |
| Drawing to Image | NumPy + OpenCV |
| Character Recognition | TensorFlow/Keras (CNN trained on EMNIST or custom air-writing data) |
| Real-time UI | OpenCV window (for now), optionally Streamlit or Pygame |
| Packaging | PyInstaller or Docker (optional) |
| (Optional) Hosting | Render / Hugging Face Spaces / Streamlit Cloud |


## 🔄 System Workflow

1. User draws a letter in the air with their finger
2. Mediapipe tracks the fingertip and logs (x, y) points
3. Points are drawn on a virtual canvas using OpenCV
4. When gesture ends, the canvas is converted into a grayscale image
5. The image is passed to a CNN model that predicts the letter
6. The predicted letter is added to a growing output string and displayed

## 💡 Features

- ✅ Real-time hand gesture tracking
- ✅ Air-drawing canvas that simulates writing on air
- ✅ Character recognition using a trained CNN
- ✅ Live output display (typed text)


