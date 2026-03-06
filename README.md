# farming-hub-ml-
# 🌱 AgriSense – AI Plant Disease Detection System

AgriSense is an **AI-powered plant health monitoring system** that detects crop diseases using image analysis.  
It integrates **ESP32-CAM, TensorFlow, and Flask** to provide real-time disease detection and treatment recommendations for farmers.

---

## 🚀 Features

- 📷 **ESP32-CAM Integration** – Capture plant images directly from field cameras  
- 🤖 **AI Disease Detection** – Deep learning model identifies plant diseases  
- 🌿 **Real-Time Analysis** – Instant disease prediction with confidence score  
- 📊 **Detailed Plant Health Report** – Severity, treatment, and prevention methods  
- 📱 **Multiple Capture Options**
  - Upload image
  - Device camera
  - ESP32-CAM live stream
- 🔬 **Smart Agriculture Support** – Helps farmers detect diseases early

---

## 🧠 AI Model

The system uses a **TensorFlow deep learning model** trained for plant disease classification.

**Model Architecture**
- MobileNetV2
- Image size: 256x256
- Output: Disease classification + confidence score

**Supported Classes**
- Healthy
- Powdery Mildew
- Rust Disease

---

## 🏗️ System Architecture
