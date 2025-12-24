# 🧠 Machine Learning Portfolio

Welcome to my Machine Learning Portfolio – a curated collection of projects showcasing my journey and skills in applied AI and data science. I'm a Computer Science student at Bina Nusantara University specializing in Artificial Intelligence. This repository demonstrates not only my academic progress but also my passion for solving real-world problems with data and machine learning.

🔗 LinkedIn: https://www.linkedin.com/in/christian-luis-efendy-53b25a217/

---

## 🩺 Skin Disease Detection using Hybrid CNN–Transformer (Thesis)

<img width="1189" height="490" alt="vit-efficientnetb1" src="https://github.com/user-attachments/assets/804e08c1-a36e-43b8-a019-f690e75002fc" />

This project is part of my undergraduate thesis and focuses on classifying skin lesions using a hybrid deep learning architecture that combines Convolutional Neural Networks (CNNs) for local feature extraction and Transformer models for global context understanding.

Highlights:
- Medical image classification using dermatoscopic images  
- Hybrid CNN–Transformer architecture  
- Designed for real-world diagnostic assistance  
- Published in a scientific journal  

Skills: Computer Vision, CNNs, Vision Transformers, Medical Imaging, Model Evaluation

---

## 🧠 IPC Similarity Verifier – Face Matching & Liveness Detection

This project verifies whether the person in a selfie is the same as in an uploaded KTP (Indonesian ID card) using facial recognition, and checks for liveness (anti-spoofing).

Live demo (Hugging Face):  
https://c-luis-e-ipc-similarity-verifier.hf.space

Privacy Notice:  
All images uploaded via the web or API are processed in memory only. No images are stored or collected.

---

### 🔍 API: /api/verify – Identity Similarity Check

Method: POST  
Content-Type: multipart/form-data  

Body Parameters:
- ktp_image: image file (JPG/PNG) of the KTP  
- selfie_image: image file (JPG/PNG) of the user’s selfie  

Example curl command:

    curl -X POST https://c-luis-e-ipc-similarity-verifier.hf.space/api/verify \
      -F "ktp_image=@/path/to/ktp.jpg" \
      -F "selfie_image=@/path/to/selfie.jpg"

Example success response:

    {
      "matched": true,
      "distance": 0.31,
      "threshold": 0.4,
      "message": "Face match successful"
    }

---

### 🧬 API: /api/liveness – Liveness Detection (Anti-Spoofing)

Method: POST  
Content-Type: multipart/form-data  

Body Parameter:
- image: selfie image file (JPG/PNG)

Example curl command:

    curl -X POST https://c-luis-e-ipc-similarity-verifier.hf.space/api/liveness \
      -F "image=@/path/to/selfie.jpg"

Example success response:

    {
      "liveness_passed": true,
      "confidence": 0.9832
    }

---

Technologies Used:
- Flask
- OpenCV
- DeepFace
- Gunicorn

---

### 🔹 Wise Frame – AI-powered Face & Eyewear Matching

<img width="1489" height="1189" alt="face_shape_detector" src="https://github.com/user-attachments/assets/c7cc456d-5ea2-4176-944b-eb7486891ce0" />

GitHub Repository:  
https://github.com/celine1906/C8S2-MLChallenge-WiseFrame

Wise Frame is an AI-powered mobile app that helps users discover the perfect eyeglasses based on face shape, skin tone, and facial proportions. It integrates a machine learning model built in Python with a native iOS frontend built in SwiftUI.

Features:
- Face shape detection via facial landmark extraction  
- ARKit-powered virtual try-on  
- Personalized frame recommendations  
- In-app product listings and user onboarding  

Tech Stack:
- SwiftUI
- ARKit
- Vision Framework
- MediaPipe
- Python
- Xcode
- CoreML

Skills: Computer Vision, iOS App Development, ML Model Integration, UX Design, A/B Testing

---

### 🔹 Pose to Impress

GitHub Repository:  
https://github.com/LuisSsiuL/pose-to-impress

Real-time pose correction system using webcam or mobile input, built for fitness and dance posture tracking.

Tech Stack:
- OpenCV
- MediaPipe
- Real-Time Feedback

Skills: Human Pose Estimation, Computer Vision, User Interaction

---

## 🩻 Medical Imaging

### 🔹 Pneumonia Detection from Chest X-Rays
<img width="1567" height="1582" alt="Chest_Xray" src="https://github.com/user-attachments/assets/49bb545c-1173-4966-88ff-ed8a5a806cc6" />

Developed a CNN-based model to classify chest X-rays as either Normal or Pneumonia.

Tech Stack:
- CNN
- Image Augmentation
- Evaluation Metrics

Skills: Deep Learning, CNNs, Medical Imaging, Model Evaluation

---

## 🎧 Audio & Speech

### 🔹 DeepFake Audio Detection
<img width="1033" height="470" alt="DeepFakeDetection" src="https://github.com/user-attachments/assets/ecb74e9d-c6d3-48af-8316-90f6515ff8d5" />

Built a deep learning system to detect synthetic audio using CNNs and audio features such as MFCCs.

Skills: Audio Signal Processing, Spectrograms, Deep Learning, CNN/RNN

---

## 🧾 NLP Projects

### 🔹 Text Summarizer – Extractive Summarization for Technical Docs

Built a summarizer tailored to software engineering documents using TF-IDF and TextRank.

Skills: NLP Preprocessing, Summarization Algorithms, Tokenization, Text Cleaning

---

### 🔹 Sentiment Analysis – Satria Data Competition Insights

<img width="567" height="455" alt="sentiment_analysis" src="https://github.com/user-attachments/assets/ed7fc982-c0fb-4c9a-80fb-b41866276464" />

Analyzed participant demographics and trends in an Indonesian data competition.

Skills: EDA, Categorical/Numerical Analysis, Matplotlib, Seaborn, Plotly

---

## 📈 Clustering & Unsupervised Learning

### 🔹 Customer Segmentation with K-Means, DBSCAN, and Hierarchical Clustering

Segmented mall customers into distinct groups based on demographics and spending habits.

Highlights:
- Elbow Method and Silhouette Score  
- DBSCAN for outlier detection  
- Hierarchical clustering with dendrograms  

Skills: Clustering, Dimensionality Reduction, Unsupervised ML, Data Visualization

---

## ✅ Classification Models

### 🔹 Logistic Regression – Diabetes Prediction

Binary classifier to predict diabetes likelihood using health indicators.

Skills: Logistic Regression, Feature Engineering, Model Evaluation

---

## 📊 Regression Models

### 🔹 Linear Regression – Predictive Modeling from Scratch

Implemented linear regression using NumPy and Matplotlib.

Skills: Linear Algebra, NumPy, Data Visualization, Predictive Modeling

---

## 🚀 Summary

This portfolio demonstrates a strong focus on applied machine learning and computer vision, spanning medical imaging, biometric verification, real-time systems, NLP, audio processing, and mobile integration, with an emphasis on building deployable and impactful AI solutions.
