# Real-Time MNIST Digit Recognizer 🧠📷

A full-stack computer vision project that recognizes handwritten digits (0-9) in real-time using a custom Convolutional Neural Network (CNN) and a live webcam feed. 

## 🎯 Overview
While training a model on the MNIST dataset is a classic deep learning milestone, getting that model to perform accurately in the real world—with varying lighting, shadows, and camera angles—is a completely different engineering challenge. This project bridges the "domain gap" by using OpenCV for live image processing and a PyTorch CNN for real-time inference.

**Key Features:**
* **Custom CNN Architecture:** A 3-block Convolutional Neural Network built from scratch, utilizing Batch Normalization and Dropout (0.5) to prevent overfitting.
* **Live Webcam Integration:** Real-time OpenCV video capture with an interactive Region of Interest (ROI).
* **Advanced Preprocessing:** Uses Adaptive Otsu Thresholding to isolate the digit from background noise, and pads the bounding box to $20 \times 20$ pixels to perfectly mimic the original MNIST dataset distribution.
* **Dynamic UI:** An interactive heads-up display showing rolling average predictions, confidence scores, and a live 28x28 "What the Model Sees" thumbnail.

## 🛠️ Tech Stack
* **Deep Learning:** PyTorch, Torchvision
* **Computer Vision:** OpenCV (cv2)
* **Data Processing:** NumPy
* **Language:** Python 3.x

## 🚀 Installation & Usage

1. **Clone the repository:**
   ```bash
   git clone [https://github.com/](https://github.com/)[YourUsername]/mnist-realtime-recognizer.git
   cd mnist-realtime-recognizer
