# 🩺 Kidney Stone Prediction System

A machine learning application that predicts kidney stone risk using:
1. **Urine Analysis** - Parameter-based prediction with Random Forest
2. **Medical Image Analysis** - CT/X-ray scan classification with CNN
3. **Urine Strip OCR** - AI-powered parameter extraction using Gemini AI

## ✨ Features

- 🚀 Real-time ML predictions with confidence scores
- ️ CT/X-ray scan analysis with deep learning
- 📷 Urine test strip image analysis with OCR
- � Personalized health recommendations
- 📜 Prediction history tracking
- 🎨 Modern responsive UI with Next.js and TypeScript

## 🏗️ Tech Stack

**Frontend:** Next.js 15, React 19, TypeScript, Tailwind CSS  
**Backend:** FastAPI, Python 3.12  
**ML Models:** Random Forest (urine), CNN (images), Google Gemini AI (OCR)  
**Libraries:** TensorFlow, scikit-learn, OpenCV, Pandas

## 📋 Prerequisites

- Node.js 18+ and npm
- Python 3.8+ (3.12 recommended)
- Git

## 🚀 Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/Sharath-Devadiga/KidneyStonePrediction.git
cd KidneyStonePrediction
```

### 2. Backend Setup

```bash
cd ml_api

# Create virtual environment (Windows PowerShell)
python -m venv .venv-api
.\.venv-api\Scripts\Activate.ps1

# Install dependencies
pip install -r requirements.txt

# Configure Gemini AI (optional - for urine strip OCR)
# Get API key from: https://makersuite.google.com/app/apikey
# Create .env file and add: GEMINI_API_KEY=your_key_here

# Start backend server
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

**Expected output:**
```
✓ Urine analysis model loaded successfully!
✓ Image classification model loaded successfully!
INFO: Uvicorn running on http://127.0.0.1:8000
```

### 3. Frontend Setup

Open a new terminal:

```bash
cd webapp

# Install dependencies
npm install

# Start development server
npm run dev
```

**Expected output:**
```
▲ Next.js 15.x.x
- Local: http://localhost:3000
```

### 4. Access Application

Open browser and go to: **http://localhost:3000**

Click "Start Prediction" to use the app.


## 🧪 Testing the App

### Manual Entry Tab
1. Enter sample urine test values:
   - Gravity: 1.015, pH: 6.0, Osmolarity: 400
   - Conductivity: 15.0, Urea: 150, Calcium: 2.0
2. Click "Predict Risk"

### Urine Strip Image Tab
1. Upload a urine test strip image
2. AI will extract parameters automatically
3. Click "Predict with Extracted Parameters"

### CT/X-Ray Scan Tab
1. Upload kidney scan from `ml_api/CT_images/Test/Stone/` or `Normal/`
2. Click "Analyze CT Scan"
3. View classification and recommendations

## 📊 API Endpoints

- `GET /health` - Health check
- `POST /predict` - Urine analysis prediction
- `POST /predict/image` - CT/X-ray classification
- `POST /predict/urine-strip` - Urine strip OCR
- `GET /docs` - Interactive API documentation at http://127.0.0.1:8000/docs

## 📁 Project Structure

```
kidney_stone_project/
├── ml_api/                   # Backend (FastAPI + ML models)
│   ├── main.py              # API server
│   ├── models/              # Trained models
│   └── CT_images/           # Medical image dataset
├── webapp/                   # Frontend (Next.js + React)
│   ├── app/                 # Pages and API routes
│   └── components/          # UI components
└── data/                     # Training datasets
```

## ⚠️ Disclaimer

**This application is for educational purposes only.** It is NOT a substitute for professional medical advice. Always consult qualified healthcare professionals for medical concerns.

