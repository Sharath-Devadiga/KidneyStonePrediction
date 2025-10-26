# 🩺 Kidney Stone Prediction System

A comprehensive machine learning application that predicts kidney stone risk using **dual detection methods**: 
1. **Urine Analysis** - Traditional parameter-based prediction
2. **Medical Image Analysis** - CNN-based CT/X-ray scan classification

Built with modern technologies including Next.js, TypeScript, FastAPI, TensorFlow, and scikit-learn.

## ✨ Features

### Urine Analysis Module
- 🚀 **Real-time ML Predictions** - Instant kidney stone risk assessment
- 📊 **Confidence Scores** - Transparent probability-based predictions
- 💡 **Smart Recommendations** - Personalized health advice
- 📜 **History Tracking** - Monitor trends over time
- 📥 **Export Functionality** - Download prediction data as JSON

### Medical Image Analysis Module (NEW!)
- 🖼️ **CT/X-Ray Scan Analysis** - Deep learning-based kidney stone detection
- 🧠 **CNN Model** - Trained on 3,300+ medical images
- 📸 **Image Upload** - Drag-and-drop or click to upload
- 🎯 **High Accuracy** - 100% confidence on kidney stone detection
- 💊 **Medical Recommendations** - Immediate action guidance

### UI/UX
- 🎨 **Modern UI/UX** - Beautiful, responsive design with smooth animations
- 🔄 **Batch Processing** - Analyze multiple samples at once (API)
- 📈 **Model Statistics** - View detailed model performance metrics
- 🔒 **Type-Safe** - Full TypeScript implementation
- ⚡ **Fast & Efficient** - Sub-second prediction times

## 🏗️ Tech Stack

### Frontend
- **Next.js 15** - React framework with App Router
- **React 19** - UI library
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Framer Motion** - Animations
- **Radix UI** - Accessible components
- **Axios** - HTTP client

### Backend
- **FastAPI** - Modern Python API framework
- **scikit-learn** - Machine learning (Random Forest for urine analysis)
- **TensorFlow 2.18** - Deep learning framework (CNN for image classification)
- **Keras 3.11** - Neural network API
- **OpenCV 4.10** - Image processing
- **Pillow** - Image manipulation
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Pydantic** - Data validation
- **Joblib** - Model serialization

### Machine Learning Models
- **Random Forest Classifier** - Urine analysis predictions
- **CNN (Convolutional Neural Network)** - Medical image classification
  - 4 Convolutional layers
  - BatchNormalization & Dropout
  - ~7M parameters
  - Input: 150x150 RGB images

## 📋 Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.8+ (3.12 recommended)
- **pip** (Python package manager)
- **Git** (for cloning the repository)

## 🚀 Quick Start Guide

Follow these steps to set up and run the application on your local machine.

### Step 1: Clone the Repository

```bash
git clone https://github.com/Sharath-Devadiga/KidneyStonePrediction.git
cd KidneyStonePrediction
```

### Step 2: Backend Setup (FastAPI + ML Models)

#### 2.1 Navigate to Backend Directory
```bash
cd ml_api
```

#### 2.2 Create Virtual Environment (Recommended)

**Windows (PowerShell):**
```powershell
python -m venv .venv-api
.\.venv-api\Scripts\Activate.ps1
```

**Windows (CMD):**
```cmd
python -m venv .venv-api
.venv-api\Scripts\activate.bat
```

**macOS/Linux:**
```bash
python3 -m venv .venv-api
source .venv-api/bin/activate
```

#### 2.3 Install Python Dependencies
```bash
pip install -r requirements.txt
```

This will install:
- FastAPI & Uvicorn (API framework)
- scikit-learn, pandas, numpy (Urine analysis ML)
- TensorFlow, Keras, OpenCV (Image analysis ML)
- Pillow, python-multipart (Image processing)
- And other required packages

#### 2.4 Verify Models Exist

The trained models should be in `ml_api/models/`:
- ✅ `kidney_stone_model_extended.joblib` (Urine analysis model)
- ✅ `kidney_stone_cnn_model.keras` (Image classification model)

If models are missing, you'll need to train them (see [Model Training](#-model-training) section).

#### 2.5 Start the Backend Server
```bash
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

✅ **Expected Output:**
```
✓ Urine analysis model loaded successfully!
✓ Image classification model loaded successfully! (.keras format)
INFO: Uvicorn running on http://127.0.0.1:8000
INFO: Application startup complete.
```

🌐 **API Endpoints:**
- Main API: `http://127.0.0.1:8000`
- Interactive Docs: `http://127.0.0.1:8000/docs`
- Alternative Docs: `http://127.0.0.1:8000/redoc`
- Health Check: `http://127.0.0.1:8000/health`

**⚠️ Keep this terminal open!** The API must stay running.

---

### Step 3: Frontend Setup (Next.js)

#### 3.1 Open a NEW Terminal

Keep the backend running and open a new terminal window.

#### 3.2 Navigate to Frontend Directory
```bash
cd webapp
# If you were in ml_api: cd ../webapp
```

#### 3.3 Install Node Dependencies
```bash
npm install
```

This will install Next.js, React, TypeScript, Tailwind CSS, and all required packages.

#### 3.4 Start the Development Server
```bash
npm run dev
```

✅ **Expected Output:**
```
▲ Next.js 15.x.x
- Local:        http://localhost:3000
- Ready in X ms
```

**⚠️ Keep this terminal open too!**

---

### Step 4: Access the Application

Open your web browser and navigate to:

```
http://localhost:3000
```

Click **"Start Prediction"** or go directly to:
```
http://localhost:3000/predict
```

You'll see **3 prediction modes**:
1. **📝 Manual Entry** - Enter urine test parameters manually
2. **📷 Urine Test Image** - Upload urine test strip images (future feature)
3. **🏥 CT/X-Ray Scan** - Upload kidney CT/X-ray scans for AI analysis

---

## 🧪 Testing the Application

### Test Urine Analysis

1. Go to the **Manual Entry** tab
2. Enter sample values:
   - Gravity: 1.015
   - pH: 6.0
   - Osmolarity: 400
   - Conductivity: 15.0
   - Urea: 150
   - Calcium: 2.0
3. Click **"Predict Risk"**
4. View the results with confidence score and recommendations

### Test Image Analysis

1. Go to the **CT/X-Ray Scan** tab
2. Use test images from: `ml_api/CT_images/Test/`
   - **Stone images:** `ml_api/CT_images/Test/Stone/Stone- (1001).jpg`
   - **Normal images:** `ml_api/CT_images/Test/Normal/Normal- (1001).jpg`
3. Upload an image (drag & drop or click to browse)
4. Click **"Analyze CT Scan"**
5. View classification with confidence and medical recommendations

### Test API Directly

Run the test script:
```bash
cd ml_api
python test_image_api.py
```

This will test all API endpoints with sample images.

---

## 📊 API Endpoints

### Health & Info
- `GET /` - API welcome message and capabilities
- `GET /health` - Health check (shows which models are loaded)
- `GET /metadata` - Model metadata and parameter ranges
- `GET /statistics` - Model performance statistics

### Predictions
- `POST /predict` - Urine analysis prediction (single sample)
- `POST /predict/batch` - Batch urine analysis (multiple samples)
- `POST /predict/image` - Medical image classification (CT/X-ray)




## 📊 Input Parameters

### Urine Analysis Parameters

| Parameter | Description | Normal Range | Unit |
|-----------|-------------|--------------|------|
| **gravity** | Specific Gravity | 1.005-1.030 | sg |
| **ph** | pH Level | 4.5-8.0 | pH |
| **osmo** | Osmolarity | 150-1200 | mOsm/kg |
| **cond** | Conductivity | 5-35 | mS/cm |
| **urea** | Urea Concentration | 50-500 | mg/dL |
| **calc** | Calcium Level | 0-15 | mg/dL |

### Image Analysis

**Supported Image Types:**
- CT scan images of kidneys
- KUB (Kidney, Ureter, Bladder) X-rays
- Abdominal imaging showing kidney region

**Supported Formats:** JPG, JPEG, PNG  
**Image Size:** Any (automatically resized to 150x150)  
**Color Mode:** RGB or Grayscale (automatically converted)

---

## 🧪 Model Training

### Urine Analysis Model (Random Forest)
- **Dataset:** 10,416 urine analysis samples
- **Features:** 6 key parameters
- **Algorithm:** Random Forest Classifier
- **Target:** Binary classification (0: Low Risk, 1: High Risk)
- **Training:** See `notebooks/1_Data_Exploration_and_Cleaning.ipynb`

### Image Classification Model (CNN)

If you need to retrain the image model:

```bash
cd ml_api
python train_image_model.py
```

**Training Details:**
- **Dataset:** 3,300 CT/X-ray images (2,400 train + 600 validation)
- **Architecture:** 4 Convolutional blocks + Dense layers
- **Optimizer:** Adam with learning rate scheduling
- **Callbacks:** EarlyStopping, ReduceLROnPlateau
- **Epochs:** Up to 50 (with early stopping)
- **Training Time:** ~30-60 minutes (depends on hardware)

**Dataset Structure:**
```
ml_api/CT_images/
├── Train/
│   ├── Normal/  (1,800 images)
│   └── Stone/   (600 images)
└── Test/
    ├── Normal/  (900 images)
    └── Stone/   (900 images)
```

---

## 🎯 Usage Examples

### Command Line Usage

#### 1. Start Both Servers Simultaneously

**Terminal 1 - Backend:**
```powershell
cd ml_api
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

**Terminal 2 - Frontend:**
```powershell
cd webapp
npm run dev
```

#### 2. Test API with cURL

**Urine Analysis:**
```bash
curl -X POST "http://127.0.0.1:8000/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "gravity": 1.015,
    "ph": 6.0,
    "osmo": 400,
    "cond": 15.0,
    "urea": 150,
    "calc": 2.0
  }'
```

**Image Analysis:**
```bash
curl -X POST "http://127.0.0.1:8000/predict/image" \
  -F "file=@path/to/kidney_scan.jpg"
```

### Web Interface Usage

1. **Navigate:** `http://localhost:3000/predict`
2. **Choose Mode:**
   - Manual Entry for parameter-based prediction
   - CT/X-Ray Scan for image-based prediction
3. **Submit:** Enter data or upload image
4. **Review:** See results, confidence, and recommendations
5. **Track:** View prediction history and trends

## 🧪 Development

### Project Structure

```
kidney_stone_project/
├── ml_api/                    # Backend (FastAPI + ML)
│   ├── main.py               # API endpoints
│   ├── requirements.txt      # Python dependencies
│   ├── train_image_model.py  # CNN training script
│   ├── test_image_api.py     # API testing script
│   ├── models/               # Trained ML models
│   │   ├── kidney_stone_model_extended.joblib
│   │   └── kidney_stone_cnn_model.keras
│   └── CT_images/            # Medical image dataset
│       ├── Train/
│       └── Test/
├── webapp/                    # Frontend (Next.js)
│   ├── app/                  # Next.js App Router
│   │   ├── page.tsx          # Home page
│   │   ├── predict/          # Prediction interface
│   │   └── api/              # API routes
│   ├── components/           # React components
│   ├── lib/                  # Utilities
│   └── package.json          # Node dependencies
├── notebooks/                 # Jupyter notebooks
└── data/                     # Training datasets
```

### Running Tests

**Backend API Tests:**
```bash
cd ml_api
python test_image_api.py
```

**Frontend Linting:**
```bash
cd webapp
npm run lint
```

### Building for Production

**Frontend Build:**
```bash
cd webapp
npm run build
npm start  # Runs on port 3000
```

**Backend Production:**
```bash
cd ml_api
uvicorn main:app --host 0.0.0.0 --port 8000 --workers 4
```

---

## 🔧 Configuration

### Environment Variables

**Frontend** (Optional - create `webapp/.env.local`):
```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

### Port Configuration

- **Backend:** Default `8000` (can be changed in uvicorn command)
- **Frontend:** Default `3000` (can be changed with `-p` flag: `npm run dev -- -p 3001`)

---

## � Troubleshooting

### Common Issues & Solutions

#### 1. "Module not found" errors (Backend)

**Problem:** Python packages not installed  
**Solution:**
```bash
cd ml_api
pip install -r requirements.txt
```

#### 2. "Model not found" errors

**Problem:** Trained models missing  
**Solution:** 
- Check `ml_api/models/` directory
- Retrain models if needed: `python train_image_model.py`

#### 3. "Connection refused" (Frontend can't reach API)

**Problem:** Backend not running  
**Solution:**
```bash
cd ml_api
uvicorn main:app --host 127.0.0.1 --port 8000 --reload
```

#### 4. Port already in use

**Problem:** Port 8000 or 3000 already occupied  
**Solution:**
```bash
# Change backend port
uvicorn main:app --host 127.0.0.1 --port 8001 --reload

# Change frontend port
npm run dev -- -p 3001
```

#### 5. TensorFlow installation issues (Windows)

**Problem:** TensorFlow won't install  
**Solution:**
```bash
pip install tensorflow==2.18.0 --no-cache-dir
```

#### 6. Image upload not working

**Problem:** python-multipart not installed  
**Solution:**
```bash
pip install python-multipart
```

---

## 📚 Additional Resources

### API Documentation

Once the backend is running, visit:
- **Swagger UI:** http://127.0.0.1:8000/docs
- **ReDoc:** http://127.0.0.1:8000/redoc

### Dataset Information

**Urine Analysis Dataset:**
- Source: `data/kidney_stone_urine_analysis_extended.csv`
- Samples: 10,416 records
- Features: 6 numerical parameters

**Medical Image Dataset:**
- Location: `ml_api/CT_images/`
- Total Images: 3,300
- Classes: Normal (1,800) and Stone (1,500)
- Format: JPG images

---

## 🤝 Contributing

Contributions are welcome! To contribute:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📄 License

This project is created for educational purposes.

---

## ⚠️ Important Disclaimer

**This application is for educational and informational purposes only.**

It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals regarding any medical concerns. The predictions made by this system should be used as a supplementary tool, not as a definitive medical diagnosis.







