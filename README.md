# 🧠 Machine Learning Portfolio

Welcome to my Machine Learning Portfolio – a curated collection of projects showcasing my journey and skills in applied AI and data science. I'm a Computer Science student at Bina Nusantara University specializing in Artificial Intelligence. This repository demonstrates not only my academic progress but also my passion for solving real-world problems with data and machine learning.

🔗 [LinkedIn](https://www.linkedin.com/in/christian-luis-efendy-53b25a217/)

---

## 🩺 Medical Imaging

### 🔹 Pneumonia Detection from Chest X-Rays  
Developed a CNN-based model to classify chest X-rays as either 'Normal' or 'Pneumonia'.  
**Tech Stack**: CNN, Image Augmentation, Evaluation Metrics  
**Skills**: Deep Learning, CNNs, Medical Imaging, Model Evaluation

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
