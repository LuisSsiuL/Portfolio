# 🧠 Machine Learning Portfolio

Welcome to my Machine Learning Portfolio – a curated collection of projects showcasing my journey and skills in applied AI and data science. I'm a Computer Science student at Bina Nusantara University specializing in Artificial Intelligence. This repository demonstrates not only my academic progress but also my passion for solving real-world problems with data and machine learning.

🔗 [LinkedIn](https://www.linkedin.com/in/christian-luis-efendy-53b25a217/)

---

## 🩺 Skin Disease Detection – Hybrid CNN & Transformer (Thesis)

This project is part of my undergraduate thesis and focuses on classifying skin lesions using a **hybrid CNN–Transformer architecture**. The model combines CNN-based feature extraction with Transformer-based global context modeling to improve classification performance on dermatoscopic images.

**Highlights**:
- Medical imaging classification
- Hybrid deep learning architecture
- Designed for real-world diagnostic support
- Published in a scientific journal

**Skills**: Computer Vision, CNNs, Vision Transformers, Medical Imaging, Model Evaluation

---

## 🧠 IPC Similarity Verifier – Face Matching & Liveness Detection

This project verifies whether the person in a selfie is the same as in an uploaded KTP (Indonesian ID card) using facial recognition, and checks for liveness (anti-spoofing) using DeepFace.

🔗 **Live demo on Hugging Face**: https://c-luis-e-ipc-similarity-verifier.hf.space

> ⚠️ **Privacy Notice:** All images uploaded via the web or API are processed **in memory only**. **No images are stored or collected.** Once processed, the data is discarded immediately.

### 🔍 API: `/api/verify` – Identity Similarity Check
- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Body Parameters:**
  - `ktp_image`: image file (JPG/PNG) of the KTP
  - `selfie_image`: image file (JPG/PNG) of the user’s selfie

### 🧬 API: `/api/liveness` – Liveness Detection
- Anti-spoofing using deep learning
- Confidence-based verification

**Technologies**: Flask, OpenCV, DeepFace, Gunicorn  
**Skills**: Face Recognition, Liveness Detection, API Design, Deployment, Privacy-aware ML

---

## 🕶️ Wise Frame – AI-powered Face & Eyewear Matching
💡 https://github.com/celine1906/C8S2-MLChallenge-WiseFrame

Wise Frame is an AI-powered mobile app that helps users discover the perfect eyeglasses based on their **face shape**, **skin tone**, and **facial proportions**. It integrates an **ML model** built in Python with a native **iOS frontend** built in SwiftUI.

**Features**:
- Face shape detection via facial landmark extraction
- ARKit-powered virtual try-on
- Personalized frame recommendations
- In-app product listings and onboarding

**Tech Stack**:  
SwiftUI · ARKit · Vision Framework · MediaPipe · Python · CoreML · Xcode

**Skills**: Computer Vision, iOS App Development, ML Integration, UX Design

---

## 🧍 Pose to Impress – Real-time Human Pose Estimation
💡 https://github.com/LuisSsiuL/pose-to-impress

Pose to Impress is a real-time pose correction system built using webcam or mobile input, designed for posture correction in fitness and dance contexts.

**Highlights**:
- Real-time pose estimation
- Visual feedback for posture correction
- Designed for accessibility and physical activity

**Tech Stack**: OpenCV, MediaPipe  
**Skills**: Human Pose Estimation, Computer Vision, Real-Time Systems, HCI

---

## 🖼️ Vision Transformers vs EfficientNet – Model Comparison

A comparative study analyzing Vision Transformers, EfficientNet, and hybrid architectures for image classification tasks.

**Insights**:
- Accuracy vs computational efficiency
- Architectural trade-offs
- Visualization of training behavior

**Skills**: Transfer Learning, Model Evaluation, Experiment Design

---

## 🩻 Pneumonia Detection from Chest X-Rays

Developed a CNN-based model to classify chest X-rays as either **Normal** or **Pneumonia**, focusing on medical image preprocessing and evaluation metrics.

**Skills**: Deep Learning, CNNs, Medical Imaging, Model Evaluation

---

## 🎧 DeepFake Audio Detection

Built a deep learning system to detect synthetic audio using CNNs and audio features such as MFCCs.

**Skills**: Audio Signal Processing, Spectrograms, Deep Learning

---

## 🧑‍🎨 Face Shape Classification (with iOS Integration)

Created a facial landmark–based classification pipeline to determine face shapes and integrate the trained model into an iOS application using Apple’s Vision framework.

**Skills**: Facial Landmark Detection, CoreML, iOS Integration

---

## 🧾 NLP & Text Analysis Projects

### 🔹 Text Summarizer – Extractive Summarization
Built a summarizer for technical documents using TF-IDF and TextRank.

### 🔹 Sentiment Analysis – Satria Data Competition
Performed exploratory data analysis and sentiment analysis on competition-related datasets.

**Skills**: NLP, TF-IDF, TextRank, EDA

---

## 📊 Foundational Machine Learning Projects

### 🔹 Linear Regression – From Scratch
Implemented linear regression using NumPy and visualized predictions using Matplotlib.

### 🔹 Logistic Regression – Diabetes Prediction
Binary classification using health indicators with evaluation via precision, recall, and F1-score.

### 🔹 Customer Segmentation – Clustering
Applied K-Means, DBSCAN, and hierarchical clustering with silhouette analysis.

**Skills**: Classical ML, Feature Engineering, Model Evaluation

---

## 🚀 Summary

This portfolio highlights my focus on **applied AI**, **computer vision**, and **end-to-end machine learning systems**, with a strong emphasis on deploying models into real-world applications, particularly within the Apple ecosystem.
