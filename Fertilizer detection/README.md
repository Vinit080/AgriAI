<div align="center">
  <img src="https://raw.githubusercontent.com/Vinit080/AgriAI/master/Fertilizer%20detection/static/img/dashboard_hero.png" alt="AgriAI Hero" width="800" />
  
  # 🌿 AgriAI

  **AI-Powered Smart Farming Assistant**<br>
  *A comprehensive platform for crop disease detection, fertilizer prediction, and seed quality analysis.*

  [![Python](https://img.shields.io/badge/Python-3.12-blue.svg?style=flat-square&logo=python&logoColor=white)](https://www.python.org)
  [![Flask](https://img.shields.io/badge/Flask-3.1.3-green.svg?style=flat-square&logo=flask&logoColor=white)](https://flask.palletsprojects.com/)
  [![TensorFlow](https://img.shields.io/badge/TensorFlow-2.21.0-orange.svg?style=flat-square&logo=tensorflow&logoColor=white)](https://tensorflow.org)
  [![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9.0-yellow.svg?style=flat-square&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
  [![License](https://img.shields.io/badge/License-MIT-purple.svg?style=flat-square)](LICENSE)

  [Features](#-key-features) • [Installation](#-installation) • [AI Models](#-ai-models) • [Tech Stack](#-tech-stack) • [Documentation](#-documentation)
</div>

---

## 🎯 Overview

AgriAI is a full-stack, production-grade web application that brings deep learning and machine learning to the fingertips of farmers, agronomists, and agricultural researchers. The platform integrates three independent AI engines into a single, secure, multi-lingual interface to optimize farm inputs, verify seed quality, and diagnose crop diseases.

## ✨ Key Features

- **🧪 NPK Fertilizer Optimization**: Predicts optimal Nitrogen, Phosphorus, and Potassium requirements and computes the minimum-cost procurement plan using linear programming. Includes SHAP explainability charts.
- **🌾 Seed Quality Analysis**: Uses an EfficientNetB0 deep learning model to classify seed images into *High*, *Average*, or *Poor* quality tiers. Features Grad-CAM heatmaps for visual explainability.
- **🍃 Crop Disease Detection**: Employs a fine-tuned MobileNetV2 model capable of identifying 38 distinct crop diseases from leaf images with 96.7% accuracy. Also generates Grad-CAM overlays.
- **🌍 Multi-Lingual Support**: Accessible in 7 Indian regional languages (English, Hindi, Marathi, Tamil, Telugu, Bengali, Gujarati) via the Google Translate API.
- **📊 Real-time Dashboard**: Interactive Chart.js analytics for crop yields and fertilizer distribution.
- **🔐 Secure Authentication**: PBKDF2:SHA256 password hashing, CSRF protection, rate-limiting, and strict Content Security Policies (CSP).

---

## 🧠 AI Models

### 1. Crop Disease Detector
- **Architecture**: MobileNetV2 (Pre-trained on ImageNet, top 30 layers fine-tuned).
- **Dataset**: PlantVillage (54,000+ images).
- **Performance**: 96.7% Validation Accuracy across 38 classes.
- **Input**: 224x224 RGB image.

### 2. Seed Quality Classifier
- **Architecture**: EfficientNetB0.
- **Classes**: 3 tiers (High, Average, Poor) mapped from pure, discolored, broken, and silkcut seeds.
- **Performance**: 72.3% Validation Accuracy.
- **Input**: 128x128 RGB image.

### 3. Fertilizer NPK Predictor
- **Architecture**: Random Forest Regressor inside a Scikit-Learn Pipeline.
- **Features**: State, Season, Crop, Area, Yield, Rainfall, Temperature, pH, Soil Type.
- **Output**: Multi-target N, P, K requirements + SHAP Feature Importance.

---

## 💻 Tech Stack

- **Backend**: Flask, SQLAlchemy (SQLite), Flask-Login, Flask-WTF, Flask-Limiter, Flask-Talisman
- **Machine Learning**: TensorFlow, Keras, scikit-learn, SHAP
- **Image Processing**: OpenCV (Grad-CAM), Pillow, Matplotlib
- **Frontend**: HTML5, Vanilla CSS (Dark Mode), Chart.js, Font Awesome

---

## 🚀 Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Vinit080/AgriAI.git
cd AgriAI/"Fertilizer detection"
```

### 2. Set up the Environment
Create a virtual environment and install dependencies:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 3. Configure Environment Variables
Create a `.env` file in the root directory based on `.env.example`:
```ini
FLASK_SECRET_KEY=your_secure_random_key_here
FLASK_ENV=development
DATABASE_URL=sqlite:///farm.db
```

### 4. Run the Application
```bash
python app.py
```
The app will be available at `http://127.0.0.1:5000`.

---

## 📚 Documentation

For a deep dive into the system architecture, use cases, API references, and security models, please check out the official **Technical Report**:

👉 [**AgriAI Technical Report (HTML)**](AgriAI_Technical_Report.html)  
👉 [**AgriAI Technical Report (PDF)**](AgriAI_Technical_Report.pdf)

---

<div align="center">
  <p>Built with ❤️ for sustainable agriculture.</p>
</div>
