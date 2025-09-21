from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import pandas as pd

# 1. Initialize the FastAPI app
app = FastAPI(title="Kidney Stone Prediction API")

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

# A simple root endpoint to check if the API is running
@app.get("/")
def read_root():
    return {"message": "Welcome to the Kidney Stone Prediction API"}