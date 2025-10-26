from fastapi import FastAPI, HTTPException, File, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field, validator
from typing import List, Optional
import joblib
import pandas as pd
import numpy as np
from datetime import datetime
import os
import io
from PIL import Image
import json
import base64

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Gemini AI for OCR
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
    # Configure Gemini API
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
    if GEMINI_API_KEY:
        genai.configure(api_key=GEMINI_API_KEY)
        print("✓ Gemini AI configured successfully!")
    else:
        print("⚠️ Warning: GEMINI_API_KEY not found in environment variables")
        print("   Urine strip OCR will not be available.")
        print("   Get a free API key from: https://makersuite.google.com/app/apikey")
        GEMINI_AVAILABLE = False
except Exception as e:
    print(f"⚠️ Warning: Gemini AI not available - {e}")
    GEMINI_AVAILABLE = False


def gemini_generate(prompt, img=None):
    """
    Try calling Gemini with several candidate model names. If none succeed,
    call list_models() and raise an informative exception.

    Returns the `response` object from the successful model.generate_content call.
    """
    if not GEMINI_AVAILABLE:
        raise RuntimeError("Gemini SDK not available or not configured")

    candidates = [
        "gemini-pro-vision",
        "gemini-1.5-pro",
        "gemini-1.5-flash-latest",
        "gemini-2.0-flash-exp",
        "gemini-1.5-flash"
    ]

    last_exc = None
    for candidate in candidates:
        try:
            model = genai.GenerativeModel(candidate)
            # generate_content accepts a list of inputs; include image if provided
            if img is not None:
                resp = model.generate_content([prompt, img])
            else:
                resp = model.generate_content([prompt])
            return resp
        except Exception as e:
            last_exc = e
            # If model not found or not supported, try next candidate
            err_msg = str(e).lower()
            if (
                "not found" in err_msg
                or "not supported" in err_msg
                or "404" in err_msg
                or ("model" in err_msg and "not" in err_msg)
            ):
                continue
            # For other errors, raise immediately
            raise

    # If we get here, none of the candidates worked. Try listing available models
    try:
        available = genai.list_models()
        # available might be a list-like object; extract names safely
        model_names = []
        for m in available:
            try:
                # m may be dict-like or object
                name = m.get("name") if isinstance(m, dict) else getattr(m, "name", None)
                if name:
                    model_names.append(name)
            except Exception:
                continue
        sample = ", ".join(model_names[:10]) if model_names else "(no models returned)"
    except Exception:
        sample = "(could not list models)"

    raise RuntimeError(
        f"No working Gemini models found among candidates. Last error: {last_exc}. Available models: {sample}"
    )

# 1. Initialize the FastAPI app
app = FastAPI(
    title="Kidney Stone Prediction API", 
    description="Advanced API for predicting kidney stone risk based on urine analysis and medical images",
    version="3.0.0",
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

# 2. Load the trained models
try:
    import warnings
    warnings.filterwarnings('ignore')
    model = joblib.load("models/kidney_stone_model_extended.joblib")
    MODEL_LOADED = True
    print("✓ Urine analysis model loaded successfully!")
except Exception as e:
    print(f"Warning: Could not load urine analysis model - {e}")
    model = None
    MODEL_LOADED = False

# Load image model
IMAGE_MODEL_LOADED = False
image_model = None
img_size = (150, 150)

try:
    import tensorflow as tf
    from tensorflow.keras.models import load_model as keras_load_model
    from tensorflow.keras.preprocessing import image as keras_image
    
    # Try loading .keras format first, then .h5
    if os.path.exists("models/kidney_stone_cnn_model.keras"):
        image_model = keras_load_model("models/kidney_stone_cnn_model.keras")
        IMAGE_MODEL_LOADED = True
        print("✓ Image classification model loaded successfully! (.keras format)")
    elif os.path.exists("models/kidney_stone_cnn_model.h5"):
        image_model = keras_load_model("models/kidney_stone_cnn_model.h5")
        IMAGE_MODEL_LOADED = True
        print("✓ Image classification model loaded successfully! (.h5 format)")
    else:
        print("⚠️ Warning: Image model not found. Please train the model first using train_image_model.py")
    
    # Load model info
    if os.path.exists("models/image_model_info.json"):
        with open("models/image_model_info.json", 'r') as f:
            model_info = json.load(f)
            img_size = tuple(model_info.get('img_size', [150, 150]))
            print(f"✓ Image model info loaded: img_size={img_size}")
except Exception as e:
    print(f"⚠️ Warning: Could not load image model - {e}")
    IMAGE_MODEL_LOADED = False

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

class ImagePredictionResponse(BaseModel):
    prediction: int
    classification: str
    confidence: float
    probability_normal: float
    probability_stone: float
    recommendations: List[str]
    timestamp: str
    model_type: str = "CNN Image Classifier"

# 4. Helper functions
def preprocess_image(image_data: bytes) -> np.ndarray:
    """
    Preprocess image for model prediction
    
    Args:
        image_data: Raw image bytes
    
    Returns:
        Preprocessed image array ready for prediction
    """
    try:
        # Open image from bytes
        img = Image.open(io.BytesIO(image_data))
        
        # Convert to RGB if necessary
        if img.mode != 'RGB':
            img = img.convert('RGB')
        
        # Resize to model input size
        img = img.resize(img_size)
        
        # Convert to array and normalize
        img_array = np.array(img)
        img_array = img_array / 255.0  # Normalize to [0, 1]
        
        # Add batch dimension
        img_array = np.expand_dims(img_array, axis=0)
        
        return img_array
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

def get_image_recommendations(classification: str, confidence: float) -> List[str]:
    """
    Generate recommendations based on image classification
    
    Args:
        classification: 'Stone' or 'Normal'
        confidence: Confidence score (0-1)
    
    Returns:
        List of recommendations
    """
    recommendations = []
    
    if classification == "Stone":
        recommendations.append("🔴 Kidney stone detected in the image")
        recommendations.append("⚠️ Immediate medical consultation is strongly recommended")
        recommendations.append("💊 Follow up with a urologist for treatment options")
        recommendations.append("🔬 Consider additional diagnostic tests (CT scan, ultrasound)")
        
        if confidence > 0.9:
            recommendations.append("📊 High confidence detection - take action promptly")
        elif confidence > 0.7:
            recommendations.append("📊 Moderate-high confidence - medical verification advised")
        else:
            recommendations.append("📊 Lower confidence - seek professional medical imaging interpretation")
            
        recommendations.append("💧 Increase water intake to 2-3 liters per day")
        recommendations.append("🥗 Discuss dietary modifications with your healthcare provider")
        
    else:  # Normal
        recommendations.append("✅ No kidney stone detected in the image")
        recommendations.append("👍 Kidney appears normal in this scan")
        
        if confidence > 0.9:
            recommendations.append("📊 High confidence - maintain healthy lifestyle")
        else:
            recommendations.append("📊 Consider follow-up imaging if symptoms persist")
            
        recommendations.append("💪 Continue preventive measures: adequate hydration")
        recommendations.append("🏃 Maintain regular physical activity")
        recommendations.append("🥗 Follow a balanced diet low in sodium and animal protein")
    
    return recommendations
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

@app.post("/predict/image", response_model=ImagePredictionResponse)
async def predict_image(file: UploadFile = File(...)):
    """
    Predict kidney stone presence from CT/X-ray image
    
    Args:
        file: Uploaded image file (JPG, PNG, etc.)
    
    Returns:
        ImagePredictionResponse with classification and recommendations
    """
    if not IMAGE_MODEL_LOADED:
        raise HTTPException(
            status_code=503, 
            detail="Image model not loaded. Please train the model first using train_image_model.py"
        )
    
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Validate if image is a medical scan using Gemini AI
        if GEMINI_AVAILABLE and GEMINI_API_KEY:
            try:
                img_pil = Image.open(io.BytesIO(image_data))
                
                validation_prompt = """Look at this image and identify what type of image it is.
                
Is this:
A) A medical scan (CT scan, X-ray, ultrasound, MRI) showing internal body parts/organs
B) A urine test strip (colorful strip with multiple test pads)
C) Something else

Respond with ONLY one letter: A, B, or C"""

                validation_response = gemini_generate(validation_prompt, img_pil)
                image_type = validation_response.text.strip().upper()
                
                # If it's a urine test strip, reject it
                if 'B' in image_type:
                    raise HTTPException(
                        status_code=400, 
                        detail="❌ Invalid image type. This appears to be a urine test strip. Please use the 'Urine Test Image' tab for urine strip analysis, or upload a CT/X-ray scan image here."
                    )
                
                # If it's neither medical scan nor urine strip
                if 'C' in image_type:
                    raise HTTPException(
                        status_code=400,
                        detail="❌ Invalid image type. Please upload a CT scan or X-ray image of kidneys."
                    )
            except HTTPException:
                raise
            except Exception as e:
                # If validation fails, continue with prediction (don't block valid images)
                print(f"Warning: Image validation failed: {e}")
        
        # Preprocess image
        img_array = preprocess_image(image_data)
        
        # Make prediction
        prediction = image_model.predict(img_array, verbose=0)
        
        # Extract probability (sigmoid output for binary classification)
        probability_stone = float(prediction[0][0])
        probability_normal = 1.0 - probability_stone
        
        # Classify (threshold at 0.5)
        predicted_class = 1 if probability_stone > 0.5 else 0
        classification = "Stone" if predicted_class == 1 else "Normal"
        
        # Confidence is the probability of the predicted class
        confidence = probability_stone if predicted_class == 1 else probability_normal
        
        # Get recommendations
        recommendations = get_image_recommendations(classification, confidence)
        
        return {
            "prediction": predicted_class,
            "classification": classification,
            "confidence": round(confidence, 4),
            "probability_normal": round(probability_normal, 4),
            "probability_stone": round(probability_stone, 4),
            "recommendations": recommendations,
            "timestamp": datetime.now().isoformat(),
            "model_type": "CNN Image Classifier"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")

class UrineStripOCRResponse(BaseModel):
    success: bool
    parameters: Optional[dict] = None
    message: str
    confidence: Optional[float] = None
    missing_parameters: Optional[List[str]] = None
    prediction: Optional[PredictionResponse] = None
    prediction_method: Optional[str] = None  # "ml_model" or "gemini_ai"

def predict_with_gemini(img: Image.Image, parameters: dict) -> dict:
    """
    Use Gemini AI to predict kidney stone risk when ML model cannot be used
    (because some parameters are missing)
    
    Args:
        img: PIL Image object
        parameters: Extracted parameters (may have null values)
    
    Returns:
        Prediction result from Gemini AI
    """
    try:
        # Create a detailed prompt for Gemini to predict kidney stone risk
        prompt = f"""You are a medical AI assistant specialized in kidney stone risk assessment.

Based on the following urine test parameters extracted from an image, predict the risk of kidney stones:

**Extracted Parameters:**
- Specific Gravity: {parameters.get('gravity', 'Not detected')}
- pH Level: {parameters.get('ph', 'Not detected')}
- Osmolarity (mOsm/kg): {parameters.get('osmo', 'Not detected')}
- Conductivity (mS/cm): {parameters.get('cond', 'Not detected')}
- Urea (mg/dL): {parameters.get('urea', 'Not detected')}
- Calcium (mg/dL): {parameters.get('calc', 'Not detected')}

**Normal Ranges:**
- Specific Gravity: 1.005-1.030
- pH: 4.5-8.0
- Osmolarity: 150-1200 mOsm/kg
- Conductivity: 5-35 mS/cm
- Urea: 50-500 mg/dL
- Calcium: 0-15 mg/dL

**Task:**
Analyze these parameters and predict the kidney stone risk.

**Return your response as a JSON object with this EXACT structure:**
{{
  "prediction": 0 or 1,
  "risk_level": "Low Risk" or "High Risk",
  "confidence": 0.0 to 1.0,
  "recommendations": ["recommendation 1", "recommendation 2", "..."]
}}

Where:
- prediction: 0 = Low Risk, 1 = High Risk
- risk_level: Either "Low Risk" or "High Risk"
- confidence: Your confidence level (0.0 to 1.0)
- recommendations: Array of 4-6 specific health recommendations

Consider that some parameters may be missing. Base your prediction on available data and known kidney stone risk factors.

Return ONLY the JSON object, no other text."""

        # Call Gemini API (use helper that tries several model names)
        response = gemini_generate(prompt, img)

        # Parse response
        response_text = response.text.strip()

        # Extract JSON from response
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        # Parse JSON
        prediction_result = json.loads(response_text)

        # Add timestamp
        prediction_result["timestamp"] = datetime.now().isoformat()

        return prediction_result
        
    except Exception as e:
        # Fallback prediction if Gemini fails
        return {
            "prediction": 1,
            "risk_level": "High Risk",
            "confidence": 0.5,
            "recommendations": [
                "⚠️ Could not complete full analysis with available data",
                "💧 Increase water intake to 2-3 liters per day as a precaution",
                "🏥 Consult a healthcare professional for complete urine analysis",
                "📊 Consider getting a complete metabolic panel test",
                "🔬 Request a 24-hour urine collection test for accurate diagnosis"
            ],
            "timestamp": datetime.now().isoformat()
        }

@app.post("/predict/urine-strip", response_model=UrineStripOCRResponse)
async def analyze_urine_strip(file: UploadFile = File(...)):
    """
    Analyze urine test strip image and extract parameters using Gemini AI
    
    Args:
        file: Uploaded urine test strip image
    
    Returns:
        Extracted parameters and prediction if all parameters found
    """
    # Validate file type
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    if not GEMINI_AVAILABLE or not GEMINI_API_KEY:
        return {
            "success": False,
            "parameters": None,
            "message": "Gemini AI is not configured. Please set GEMINI_API_KEY environment variable.",
            "confidence": None,
            "missing_parameters": None,
            "prediction": None
        }
    
    try:
        # Read image data
        image_data = await file.read()
        
        # Convert to PIL Image
        img = Image.open(io.BytesIO(image_data))
        
        # Validate image type using Gemini AI
        validation_prompt = """Look at this image and identify what type of image it is.

Is this:
A) A medical scan (CT scan, X-ray, ultrasound, MRI) showing internal body parts/organs
B) A urine test strip (colorful strip with multiple test pads/color patches)
C) Something else

Respond with ONLY one letter: A, B, or C"""

        try:
            validation_response = gemini_generate(validation_prompt, img)
            image_type = validation_response.text.strip().upper()
            
            # If it's a medical scan, reject it
            if 'A' in image_type:
                return {
                    "success": False,
                    "parameters": None,
                    "message": "❌ Invalid image type. This appears to be a CT/X-ray scan. Please use the 'CT/X-Ray Scan' tab for medical scan analysis, or upload a urine test strip image here.",
                    "confidence": None,
                    "missing_parameters": None,
                    "prediction": None
                }
            
            # If it's neither urine strip nor medical scan
            if 'C' in image_type:
                return {
                    "success": False,
                    "parameters": None,
                    "message": "❌ Invalid image type. Please upload a urine test strip image showing color patches for analysis.",
                    "confidence": None,
                    "missing_parameters": None,
                    "prediction": None
                }
        except Exception as e:
            # If validation fails, continue with extraction (don't block valid images)
            print(f"Warning: Image validation failed: {e}")
        
        # Prepare prompt for Gemini
        prompt = """Analyze this urine test strip image and extract the following parameters with their numeric values:

1. **gravity** (Specific Gravity): Normal range 1.005-1.030
2. **ph** (pH Level): Normal range 4.5-8.0
3. **osmo** (Osmolarity in mOsm/kg): Normal range 150-1200
4. **cond** (Conductivity in mS/cm): Normal range 5-35
5. **urea** (Urea Concentration in mg/dL): Normal range 50-500
6. **calc** (Calcium Level in mg/dL): Normal range 0-15

IMPORTANT INSTRUCTIONS:
- Look for numeric values on the test strip or any labels/text in the image
- If you can clearly see a value for a parameter, extract it
- Return ONLY valid numeric values that fall within the normal ranges specified
- If you cannot find or read a value clearly, mark it as null
- Return your response as a JSON object with this exact structure:

{
  "gravity": <number or null>,
  "ph": <number or null>,
  "osmo": <number or null>,
  "cond": <number or null>,
  "urea": <number or null>,
  "calc": <number or null>
}

Do not include any other text, just the JSON object."""

        # Call Gemini API (use helper that tries several model names)
        response = gemini_generate(prompt, img)

        # Parse response
        response_text = response.text.strip()
        
        # Extract JSON from response (it might be wrapped in markdown code blocks)
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()
        
        # Parse JSON
        extracted_params = json.loads(response_text)
        
        # Check which parameters are missing
        required_params = ["gravity", "ph", "osmo", "cond", "urea", "calc"]
        missing = [param for param in required_params if extracted_params.get(param) is None]
        
        # If all parameters found, use ML model prediction
        if not missing:
            # Validate and prepare data for prediction
            try:
                urine_data = UrineData(**extracted_params)
                
                # Make prediction using existing ML model
                if not MODEL_LOADED:
                    return {
                        "success": False,
                        "parameters": extracted_params,
                        "message": "Parameters extracted but prediction model not loaded",
                        "confidence": None,
                        "missing_parameters": [],
                        "prediction": None,
                        "prediction_method": None
                    }
                
                # Convert to DataFrame for prediction
                input_data = pd.DataFrame([urine_data.dict()])
                prediction = model.predict(input_data)
                result = int(prediction[0])
                
                # Get probability
                confidence = None
                if hasattr(model, 'predict_proba'):
                    proba = model.predict_proba(input_data)
                    confidence = float(proba[0][result])
                
                risk_level = "High Risk" if result == 1 else "Low Risk"
                recommendations = get_recommendations(result, urine_data.dict())
                
                prediction_result = {
                    "prediction": result,
                    "risk_level": risk_level,
                    "confidence": confidence,
                    "recommendations": recommendations,
                    "timestamp": datetime.now().isoformat()
                }
                
                return {
                    "success": True,
                    "parameters": extracted_params,
                    "message": "✅ All parameters extracted successfully! Prediction completed.",
                    "confidence": 0.95,  # High confidence since we got all params
                    "missing_parameters": [],
                    "prediction": prediction_result,
                    "prediction_method": "ml_model"
                }
                
            except Exception as e:
                return {
                    "success": False,
                    "parameters": extracted_params,
                    "message": f"Parameters extracted but validation failed: {str(e)}",
                    "confidence": None,
                    "missing_parameters": [],
                    "prediction": None,
                    "prediction_method": None
                }
        else:
            # Some parameters missing - use Gemini AI to predict directly
            try:
                gemini_prediction = predict_with_gemini(img, extracted_params)
                
                return {
                    "success": True,
                    "parameters": extracted_params,
                    "message": f"⚠️ Some parameters missing ({', '.join(missing)}). Prediction completed based on available data.",
                    "confidence": gemini_prediction.get('confidence', 0.7),
                    "missing_parameters": missing,
                    "prediction": gemini_prediction,
                    "prediction_method": "gemini_ai"
                }
            except Exception as e:
                return {
                    "success": False,
                    "parameters": extracted_params,
                    "message": f"Could not extract all required parameters. Missing: {', '.join(missing)}. Prediction failed: {str(e)}",
                    "confidence": 0.5,
                    "missing_parameters": missing,
                    "prediction": None,
                    "prediction_method": None
                }
        
    except json.JSONDecodeError as e:
        return {
            "success": False,
            "parameters": None,
            "message": f"Could not parse image data. Please try a clearer image or use manual entry. Error: {str(e)}",
            "confidence": None,
            "missing_parameters": None,
            "prediction": None
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing urine strip image: {str(e)}")

# Health check endpoint
@app.get("/health")
def health_check():
    overall_status = "healthy" if (MODEL_LOADED or IMAGE_MODEL_LOADED) else "degraded"
    return {
        "status": overall_status,
        "urine_model_loaded": MODEL_LOADED,
        "image_model_loaded": IMAGE_MODEL_LOADED,
        "timestamp": datetime.now().isoformat(),
        "version": "3.0.0"
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
    overall_status = "operational" if (MODEL_LOADED or IMAGE_MODEL_LOADED) else "degraded"
    return {
        "message": "Welcome to the Kidney Stone Prediction API",
        "version": "3.0.0",
        "status": overall_status,
        "capabilities": {
            "urine_analysis": MODEL_LOADED,
            "image_classification": IMAGE_MODEL_LOADED
        },
        "endpoints": {
            "predict_urine": "/predict - POST endpoint for urine analysis predictions",
            "predict_image": "/predict/image - POST endpoint for CT/X-ray image classification",
            "predict_urine_strip": "/predict/urine-strip - POST endpoint for urine test strip OCR (experimental)",
            "batch_predict": "/predict/batch - POST endpoint for batch urine predictions",
            "metadata": "/metadata - GET endpoint for model information",
            "statistics": "/statistics - GET endpoint for model statistics",
            "health": "/health - GET endpoint for API health status",
            "docs": "/docs - Interactive API documentation",
            "redoc": "/redoc - Alternative API documentation"
        },
        "documentation": "Visit /docs for interactive API documentation"
    }