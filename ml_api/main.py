from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
from datetime import datetime

# 1. Initialize the FastAPI app
app = FastAPI(
    title="Kidney Stone Prediction API", 
    description="Advanced API for predicting kidney stone risk based on urine analysis",
    version="2.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:3001"],  # Frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the trained model
try:
    model = joblib.load("models/kidney_stone_model_extended.joblib")
    MODEL_LOADED = True
except Exception as e:
    print(f"Warning: Could not load model - {e}")
    model = None
    MODEL_LOADED = False

# 3. Define the input data structures using Pydantic
class UrineData(BaseModel):
    gravity: float = Field(..., ge=1.0, le=1.05, description="Specific gravity (1.005-1.035)")
    ph: float = Field(..., ge=4.0, le=9.0, description="pH level (4.5-8.0)")
    osmo: int = Field(..., ge=0, le=1500, description="Osmolarity in mOsm/kg (150-1200)")
    cond: float = Field(..., ge=0, le=50, description="Conductivity in mS/cm (5-40)")
    urea: int = Field(..., ge=0, le=700, description="Urea concentration in mmol/L (100-600)")
    calc: float = Field(..., ge=0, le=20, description="Calcium level in mmol/L (0.5-15)")
    
    class Config:
        json_schema_extra = {
            "example": {
                "gravity": 1.015,
                "ph": 6.0,
                "osmo": 400,
                "cond": 15.0,
                "urea": 150,
                "calc": 2.0
            }
        }

class BatchUrineData(BaseModel):
    samples: List[UrineData]
    
class PredictionResponse(BaseModel):
    prediction: int
    risk_level: str
    confidence: Optional[float] = None
    recommendations: List[str]
    timestamp: str

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    summary: dict

# 4. Helper functions
def get_recommendations(prediction: int, data: dict) -> List[str]:
    """Generate personalized recommendations based on prediction and parameters"""
    recommendations = []
    
    if prediction == 1:
        recommendations.append("⚠️ High risk detected - Consult a healthcare professional immediately")
        recommendations.append("💧 Increase water intake to at least 2-3 liters per day")
        recommendations.append("🥗 Consider dietary modifications to reduce stone formation risk")
    else:
        recommendations.append("✅ Current parameters show low risk")
        recommendations.append("💪 Maintain healthy hydration levels")
        recommendations.append("🏃 Continue healthy lifestyle habits")
    
    # Specific parameter recommendations
    if data["ph"] < 5.5:
        recommendations.append("📊 pH is low - Consider alkalizing foods")
    elif data["ph"] > 7.0:
        recommendations.append("📊 pH is high - Monitor alkaline levels")
    
    if data["calc"] > 10:
        recommendations.append("🥛 Calcium levels elevated - Reduce high-calcium foods")
    
    if data["urea"] > 400:
        recommendations.append("🍖 High urea levels - Consider reducing protein intake")
    
    return recommendations

# 5. Create the prediction endpoints
@app.post("/predict", response_model=PredictionResponse)
def predict_stone(data: UrineData):
    """
    Receives urine parameters and predicts the presence of a kidney stone.
    Returns detailed prediction with recommendations.
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    # Convert incoming data into a pandas DataFrame
    input_data = pd.DataFrame([data.dict()])
    
    # Make a prediction
    prediction = model.predict(input_data)
    result = int(prediction[0])
    
    # Get probability if available
    confidence = None
    if hasattr(model, 'predict_proba'):
        proba = model.predict_proba(input_data)
        confidence = float(proba[0][result])
    
    # Determine risk level
    risk_level = "High Risk" if result == 1 else "Low Risk"
    
    # Get recommendations
    recommendations = get_recommendations(result, data.dict())
    
    return {
        "prediction": result,
        "risk_level": risk_level,
        "confidence": confidence,
        "recommendations": recommendations,
        "timestamp": datetime.now().isoformat()
    }

@app.post("/predict/batch", response_model=BatchPredictionResponse)
def predict_batch(data: BatchUrineData):
    """
    Process multiple urine samples at once for batch predictions.
    """
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
    
    predictions = []
    high_risk_count = 0
    
    for sample in data.samples:
        input_data = pd.DataFrame([sample.dict()])
        prediction = model.predict(input_data)
        result = int(prediction[0])
        
        if result == 1:
            high_risk_count += 1
        
        confidence = None
        if hasattr(model, 'predict_proba'):
            proba = model.predict_proba(input_data)
            confidence = float(proba[0][result])
        
        risk_level = "High Risk" if result == 1 else "Low Risk"
        recommendations = get_recommendations(result, sample.dict())
        
        predictions.append({
            "prediction": result,
            "risk_level": risk_level,
            "confidence": confidence,
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat()
        })
    
    summary = {
        "total_samples": len(data.samples),
        "high_risk_count": high_risk_count,
        "low_risk_count": len(data.samples) - high_risk_count,
        "high_risk_percentage": round((high_risk_count / len(data.samples)) * 100, 2)
    }
    
    return {
        "predictions": predictions,
        "summary": summary
    }

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy" if MODEL_LOADED else "degraded",
        "model_loaded": MODEL_LOADED,
        "timestamp": datetime.now().isoformat(),
        "version": "2.0.0"
    }

# Statistics endpoint
@app.get("/statistics")
def get_statistics():
    """
    Get model performance statistics and information
    """
    # Return basic info even if model isn't loaded
    stats = {
        "model_type": "Random Forest Classifier",
        "version": "2.0.0",
        "features": ["gravity", "ph", "osmo", "cond", "urea", "calc"],
        "feature_count": 6,
        "classes": ["Low Risk (0)", "High Risk (1)"],
        "training_info": {
            "algorithm": "Random Forest",
            "target_accuracy": "> 85%",
            "cross_validation": "Applied",
        },
        "performance_metrics": {
            "note": "Actual metrics depend on your trained model evaluation"
        }
    }
    
    # Add feature importances if model is loaded and available
    if MODEL_LOADED and hasattr(model, 'feature_importances_'):
        feature_names = ["gravity", "ph", "osmo", "cond", "urea", "calc"]
        importances = model.feature_importances_
        stats["feature_importance"] = {
            name: round(float(imp), 4) 
            for name, imp in zip(feature_names, importances)
        }
        stats["model_status"] = "loaded"
    else:
        stats["model_status"] = "not_loaded"
        stats["warning"] = "Model is not loaded. Please retrain with current scikit-learn version."
    
    return stats

# Model metadata endpoint
@app.get("/metadata")
def get_metadata():
    param_ranges = {
        "gravity": {"min": 1.005, "max": 1.035, "unit": "sg"},
        "ph": {"min": 4.5, "max": 8.0, "unit": "pH"},
        "osmo": {"min": 150, "max": 1200, "unit": "mOsm/kg"},
        "cond": {"min": 5, "max": 40, "unit": "mS/cm"},
        "urea": {"min": 100, "max": 600, "unit": "mmol/L"},
        "calc": {"min": 0.5, "max": 15, "unit": "mmol/L"}
    }
    
    return {
        "model_version": "1.0.0",
        "model_type": "Random Forest Classifier",
        "input_parameters": param_ranges,
        "output": {
            "type": "binary",
            "values": {
                "0": "Low Risk",
                "1": "High Risk"
            }
        }
    }

# Root endpoint
@app.get("/")
def read_root():
    return {
        "message": "Welcome to the Kidney Stone Prediction API",
        "version": "2.0.0",
        "status": "operational" if MODEL_LOADED else "degraded",
        "endpoints": {
            "predict": "/predict - POST endpoint for single predictions",
            "batch_predict": "/predict/batch - POST endpoint for batch predictions",
            "metadata": "/metadata - GET endpoint for model information",
            "statistics": "/statistics - GET endpoint for model statistics",
            "health": "/health - GET endpoint for API health status",
            "docs": "/docs - Interactive API documentation",
            "redoc": "/redoc - Alternative API documentation"
        },
        "documentation": "Visit /docs for interactive API documentation"
    }