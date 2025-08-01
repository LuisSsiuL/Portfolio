# 🧠 Machine Learning Portfolio

Welcome to my Machine Learning Portfolio – a curated collection of projects showcasing my journey and skills in applied AI and data science. I'm a Computer Science student at Bina Nusantara University specializing in Artificial Intelligence. This repository demonstrates not only my academic progress but also my passion for solving real-world problems with data and machine learning.

🔗 [LinkedIn](https://www.linkedin.com/in/christian-luis-efendy-53b25a217/)

---

## 📊 Regression Models

### 🔹 Linear Regression – Predictive Modeling from Scratch  
A foundational project where I implemented simple linear regression using only NumPy and Matplotlib. The model was trained using the least squares method and visualized to show the best-fit line.  
**Skills**: Linear algebra, NumPy, Data Visualization, Predictive Modeling

---

## ✅ Classification Models

### 🔹 Logistic Regression – Diabetes Prediction  
Developed a binary classifier to predict the likelihood of diabetes using health indicators from the `diabetes.csv` dataset.  
**Key Features**:  
- Feature selection based on correlation  
- Model training with `scikit-learn`  
- Evaluated using confusion matrix, precision, recall, F1-score  
**Skills**: Logistic Regression, Feature Engineering, Model Evaluation

---

## 📈 Clustering & Unsupervised Learning

### 🔹 Customer Segmentation with K-Means, DBSCAN, and Hierarchical Clustering  
Segmented mall customers into distinct groups based on demographics and spending habits.  
**Highlights**:  
- Elbow Method and Silhouette Score for K selection  
- DBSCAN for outlier detection  
- Dendrogram for hierarchical clustering  
**Skills**: Clustering, Dimensionality Reduction, Unsupervised ML, Data Visualization

---

## 🧾 NLP Projects

### 🔹 Text Summarizer – Extractive Summarization for Technical Docs  
Built a summarizer tailored to software engineering documents using TF-IDF and TextRank.  
**Skills**: NLP Preprocessing, Summarization Algorithms, Tokenization, Text Cleaning

### 🔹 Sentiment Analysis – Satria Data Competition Insights  
Analyzed participant demographics and trends in an Indonesian data competition.  
**Skills**: EDA, Categorical/Numerical Analysis, Matplotlib, Seaborn, Plotly

---

## 🩺 Medical Imaging

### 🔹 Pneumonia Detection from Chest X-Rays  
Developed a CNN-based model to classify chest X-rays as either 'Normal' or 'Pneumonia'.  
**Tech Stack**: CNN, Image Augmentation, Evaluation Metrics  
**Skills**: Deep Learning, CNNs, Medical Imaging, Model Evaluation

---

## 🤖 AI vs Human Text Detector

### 🔹 AI & Plagiarism Text Classification  
Trained models to differentiate between AI-generated and human-written text.  
**Techniques**: TF-IDF, SVM, Logistic Regression, Ensemble Models  
**Skills**: Text Classification, NLP, Data Cleaning, Evaluation Metrics

---

## 🎧 Audio & Speech

### 🔹 DeepFake Audio Detection  
Built a deep learning system to detect synthetic audio using CNNs and audio features like MFCCs.  
**Skills**: Audio Signal Processing, Spectrograms, Deep Learning, CNN/RNN

---

## 🖼️ Vision Architectures

### 🔹 Vision Transformers vs EfficientNet  
Comparative analysis of ViT, EfficientNet, and hybrid models for image classification.  
**Insights**: Accuracy vs Efficiency, Architectural Trade-offs, Visualization  
**Skills**: Transfer Learning, Model Comparison, Experiment Design

---

## 🧑‍🎨 Computer Vision Projects

### 🔹 Face Shape Classification (with iOS integration)  
Created a pipeline using facial landmarks to classify face shapes.  
- Used Python for model training  
- Integrated with iOS using Swift and Apple’s Vision framework  
**Skills**: Facial Landmark Detection, iOS Integration, ML for Apps

### 🔹 Pose to Impress  
💡 [GitHub Repo](https://github.com/LuisSsiuL/pose-to-impress)  
Real-time pose correction system using webcam/mobile input, built for fitness/dance posture tracking.  
**Tech Stack**: OpenCV, MediaPipe, Real-Time Feedback  
**Skills**: Human Pose Estimation, Computer Vision, User Interaction

---

## 🕶️ AI in Fashion

### 🔹 Wise Frame – AI-powered Face & Eyewear Matching  
💡 [GitHub Repo](https://github.com/celine1906/C8S2-MLChallenge-WiseFrame)

Wise Frame is an AI-powered mobile app that helps users discover the perfect eyeglasses based on their **face shape**, **skin tone**, and **facial proportions**. It integrates an **ML model** built in Python with a native **iOS frontend** built in SwiftUI.

**Features**:
- Face shape detection via facial landmark extraction
- ARKit-powered virtual try-on
- Personalized frame recommendations
- In-app product listings and user onboarding

**Tech Stack**:  
`SwiftUI` · `ARKit` · `Vision Framework` · `MediaPipe` · `Python` · `Xcode` · `CoreML`

**Skills**: Computer Vision, iOS App Development, ML Model Integration, UX Design, A/B Testing

---

## 🧠 IPC Similarity Verifier – Face Matching & Liveness Detection

This project verifies whether the person in a selfie is the same as in an uploaded KTP (Indonesian ID card) using facial recognition, and checks for liveness (anti-spoofing) using DeepFace.

🔗 **Live demo on Hugging Face**: [IPC Similarity Verifier](https://c-luis-e-ipc-similarity-verifier.hf.space)

> ⚠️ **Privacy Notice:** All images uploaded via the web or API are processed **in memory only**. **No images are stored or collected.** Once processed, the data is discarded immediately.

---

### 🔍 API: `/api/verify` – Identity Similarity Check

- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Body Parameters:**
  - `ktp_image`: image file (JPG/PNG) of the KTP
  - `selfie_image`: image file (JPG/PNG) of the user’s selfie

#### ✅ Example `curl` Command:
```bash
curl -X POST https://c-luis-e-ipc-similarity-verifier.hf.space/api/verify \
  -F "ktp_image=@/path/to/ktp.jpg" \
  -F "selfie_image=@/path/to/selfie.jpg"
```

#### ✅ Example Success Response:
```json
{
  "matched": true,
  "distance": 0.31,
  "threshold": 0.4,
  "message": "Face match successful"
}
```

---

### 🧬 API: `/api/liveness` – Liveness Detection (Anti-Spoofing)

- **Method:** `POST`
- **Content-Type:** `multipart/form-data`
- **Body Parameter:**
  - `image`: selfie image file (JPG/PNG)

#### ✅ Example `curl` Command:
```bash
curl -X POST https://c-luis-e-ipc-similarity-verifier.hf.space/api/liveness \
  -F "image=@/path/to/selfie.jpg"
```

#### ✅ Example Success Response:
```json
{
  "liveness_passed": true,
  "confidence": 0.9832
}
```

---

### 🛠️ Technologies Used

- `Flask` (API server)
- `OpenCV` (image decoding)
- `DeepFace` (face recognition + anti-spoofing)
- `Gunicorn` (production server for Hugging Face Spaces)

---

Feel free to clone the repository and run it locally, or test it using the hosted demo.

---


## 🚀 Summary

This portfolio demonstrates a well-rounded application of machine learning and AI across various domains—NLP, computer vision, healthcare, fashion, and human-computer interaction. My goal is to keep learning, contributing, and building impactful AI solutions.

Thanks for visiting!
