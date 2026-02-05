# 😷 Real-Time Face Mask Detection System

![Python](https://img.shields.io/badge/Python-3.11-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![MediaPipe](https://img.shields.io/badge/MediaPipe-Face%20Detection-orange)
![PyTorch](https://img.shields.io/badge/PyTorch-Deep%20Learning-red)
![Status](https://img.shields.io/badge/Status-Working-success)

A **real-time face mask detection system** using **MediaPipe for face detection** and a **PyTorch deep learning model** for mask classification.  
Designed to work robustly even when the face is partially occluded (mask worn).

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
