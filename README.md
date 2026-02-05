# 😷 Real-Time Face Mask Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Detection-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Status](https://img.shields.io/badge/Status-Working-success)

A **real-time face mask detection system** using **MediaPipe for face detection** and a **PyTorch deep learning model** for mask classification.  
Designed to work robustly even when the face is partially occluded (mask worn).


Project Overview

This project is a real-time Face Mask Detection system built using Deep Learning and Computer Vision techniques. The system is capable of detecting whether a person is wearing a mask or not, both from single images and live webcam video.

The core classification model is trained using PyTorch, while MediaPipe Face Detection is used for fast and accurate face localization in real-time. Once a face is detected, it is preprocessed and passed to the trained model, which predicts the class (Mask / No Mask) along with the confidence percentage.

The application supports:

🖼️ Single image mask detection

🎥 Live webcam mask detection using OpenCV

📊 Confidence-based predictions displayed on screen

This project demonstrates a complete end-to-end ML pipeline, including model training, inference, real-time deployment, and system integration. It is designed to be lightweight, efficient, and suitable for real-world applications such as public safety monitoring, healthcare environments, and access control systems.

---

## 🚀 Demo

<p align="center">
  <img src="assets/demo.gif" width="600"/>
</p>

---

## 🧠 Key Features

- ✅ Real-time webcam detection
- ✅ Robust face detection using **MediaPipe**
- ✅ Accurate classification: **Mask / No Mask**
- ✅ Works with head movement & partial occlusion
- ✅ Confidence score visualization
- ✅ Lightweight and fast

---

## 🏗️ Architecture

Webcam Feed
↓
MediaPipe Face Detection
↓
Face ROI Extraction
↓
PyTorch CNN Model
↓
Mask / No Mask Prediction
↓
OpenCV Visualization

---

## 📁 Recommended Project Structure
Face-Mask-Detection/
│
├── app.py                  # Main application (MediaPipe + OpenCV)
├── model.pth               # Trained PyTorch model (optional upload)
├── labels.txt              # Class labels (Mask / No Mask)
│
├── requirements.txt
├── README.md
├── .gitignore
│
├── assets/
│   ├── demo.gif             # Demo video/gif
│   └── architecture.png     # Pipeline diagram (optional)
│
└── models/
    └── README.md            # (optional) model description
