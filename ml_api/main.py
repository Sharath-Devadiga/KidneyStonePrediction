from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Initialize the FastAPI app
app = FastAPI(title="Kidney Stone Prediction API", 
             description="API for predicting kidney stone risk based on urine analysis",
             version="1.0.0")

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 2. Load the trained model
model = joblib.load("models/kidney_stone_model_extended.joblib")

# 3. Define the input data structure using Pydantic
class UrineData(BaseModel):
    gravity: float
    ph: float
    osmo: int
    cond: float
    urea: int
    calc: float

# 4. Create the prediction endpoint
@app.post("/predict")
def predict_stone(data: UrineData):
    """
    Receives urine parameters and predicts the presence of a kidney stone.
    """
    # Convert incoming data into a pandas DataFrame
    # The model expects a DataFrame with specific column names
    input_data = pd.DataFrame([data.dict()])
    
    # Make a prediction
    prediction = model.predict(input_data)
    
    # The result is a NumPy array, so we get the first element
    result = int(prediction[0])
    
    # Return the prediction in a JSON response
    return {"prediction": result}

# Health check endpoint
@app.get("/health")
def health_check():
    return {
        "status": "healthy",
        "model_loaded": model is not None
    }

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
        "version": "1.0.0",
        "endpoints": {
            "predict": "/predict - POST endpoint for predictions",
            "metadata": "/metadata - GET endpoint for model information",
            "health": "/health - GET endpoint for API health status"
        }
    }