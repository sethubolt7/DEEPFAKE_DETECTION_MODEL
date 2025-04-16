# 🧠 DEEPFAKE_DETECTION_MODEL

A full-fledged system for detecting deepfakes in images and videos using a custom-trained deep learning model powered by EfficientNetB0, wrapped in an interactive Streamlit interface.

---

## 📌 About

This project implements a deepfake detection pipeline from scratch. It includes:

- A training pipeline using EfficientNetB0 with fine-tuning.
- Real-time image and video classification using a Streamlit web app.
- Frame-by-frame video analysis for deepfake frame percentage.
- Robust handling of grayscale, RGB, and RGBA inputs.

The goal is to showcase applied machine learning, model deployment, and frontend integration for a practical AI application.

---

## 🗂 Repository Structure

├── train/ # Model architecture, training scripts 
├── test data/ # Sample test videos and images
├── streamlit/app.py # Streamlit frontend code for inference 
├── streamlit/my_model.keras # Saved trained model


---

## 📚 Topics Covered

- ✅ Deepfake Image & Video Detection
- 🧠 Transfer Learning with EfficientNetB0
- 🖼️ Image Preprocessing & Channel Normalization
- 🎞️ OpenCV Video Frame Extraction
- 🔄 Batch Prediction on Video Frames
- 📈 Classification: Real vs. Fake
- 🧪 Streamlit App for Real-time Inference
- 🧼 Robust Handling of Image Formats (RGB, RGBA, Grayscale)
- 💾 Model Exporting and Deployment (`.keras`)

---

## 🚀 How It Works

1. **Training Phase**
   - EfficientNetB0 (imagenet weights) is used as a base model.
   - Custom dense layers are added for binary classification.
   - Trained on labeled real vs fake datasets.

2. **Streamlit App**
   - Users upload an image or video.
   - For images: Model instantly classifies as **Fake** or **Real**.
   - For videos: Each frame is predicted, and a percentage of fake frames is shown.

---

## 🛠️ Tech Stack

- **Python**
- **TensorFlow / Keras**
- **Streamlit**
- **OpenCV**
- **NumPy / PIL**

---

> ⚠️ This project is built for educational purposes and is not intended for production-grade deepfake detection.
