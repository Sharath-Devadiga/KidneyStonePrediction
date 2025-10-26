# 🩺 Kidney Stone Prediction System

A professional, production-ready machine learning application that predicts kidney stone risk based on urine analysis parameters. Built with modern technologies including Next.js, TypeScript, FastAPI, and scikit-learn.

## ✨ Features

- 🚀 **Real-time ML Predictions** - Instant kidney stone risk assessment
- 📊 **Confidence Scores** - Transparent probability-based predictions
- 💡 **Smart Recommendations** - Personalized health advice
- 📜 **History Tracking** - Monitor trends over time
- 📥 **Export Functionality** - Download prediction data as JSON
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
- **scikit-learn** - Machine learning (Random Forest)
- **Pandas** - Data manipulation
- **NumPy** - Numerical computing
- **Pydantic** - Data validation
- **Joblib** - Model serialization

## 📋 Prerequisites

- **Node.js** 18+ and npm
- **Python** 3.8+
- **pip** (Python package manager)

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone <your-repo-url>
cd kidney_stone_project
```

### 2. Backend Setup (FastAPI)

```bash
# Navigate to the ML API directory
cd ml_api

# Create a virtual environment (recommended)
python -m venv venv

# Activate virtual environment
# Windows (PowerShell):
.\venv\Scripts\Activate.ps1
# Windows (CMD):
.\venv\Scripts\activate.bat
# macOS/Linux:
source venv/bin/activate

# Install Python dependencies
pip install fastapi uvicorn pandas numpy scikit-learn joblib

# Start the FastAPI server
uvicorn main:app --reload --port 8000
```

The API will be available at `http://127.0.0.1:8000`

API Documentation:
- Interactive docs: `http://127.0.0.1:8000/docs`
- ReDoc: `http://127.0.0.1:8000/redoc`

### 3. Frontend Setup (Next.js)

```bash
# Navigate to the webapp directory (in a new terminal)
cd webapp

# Install Node dependencies
npm install

# Start the development server
npm run dev
```

The web application will be available at `http://localhost:3000`




## 📊 Input Parameters

| Parameter | Description | Normal Range | Unit |
|-----------|-------------|--------------|------|
| **gravity** | Specific Gravity | 1.005-1.030 | sg |
| **ph** | pH Level | 4.5-8.0 | pH |
| **osmo** | Osmolarity | 150-1200 | mOsm/kg |
| **cond** | Conductivity | 5-35 | mS/cm |
| **urea** | Urea Concentration | 50-500 | mg/dL |
| **calc** | Calcium Level | 0-15 | mg/dL |

## 🎯 Usage

1. **Start Both Servers**
   - Run FastAPI backend on port 8000
   - Run Next.js frontend on port 3000

2. **Navigate to the App**
   - Open browser to `http://localhost:3000`
   - Click "Start Prediction" or navigate to `/predict`

3. **Enter Test Results**
   - Input your urine analysis values
   - Click "Predict Risk"

4. **View Results**
   - See risk assessment (High/Low)
   - Review confidence score
   - Read personalized recommendations
   - Export or track history

## 🧪 Development

### Running Tests
```bash
# Frontend
cd webapp
npm run lint

# Backend
cd ml_api
pytest  # (if tests are implemented)
```

### Building for Production

```bash
# Frontend
cd webapp
npm run build
npm start

# Backend
cd ml_api
uvicorn main:app --host 0.0.0.0 --port 8000
```

## 🔧 Configuration

### Environment Variables (optional)

Create `.env.local` in webapp directory:
```env
NEXT_PUBLIC_API_URL=http://127.0.0.1:8000
```

## 📝 Model Training

The model was trained using:
- **Algorithm**: Random Forest Classifier
- **Dataset**: 10,416 urine analysis samples
- **Features**: 6 key parameters
- **Target**: Binary classification (0: Low Risk, 1: High Risk)
- **Validation**: Cross-validation applied

To retrain the model, see `notebooks/1_Data_Exploration_and_Cleaning.ipynb`

## ⚠️ Important Disclaimer

**This application is for educational and informational purposes only.**

It is NOT a substitute for professional medical advice, diagnosis, or treatment. Always consult with qualified healthcare professionals regarding any medical concerns. The predictions made by this system should be used as a supplementary tool, not as a definitive medical diagnosis.







