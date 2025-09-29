#!/usr/bin/env python3
"""
FastAPI Serving Application for Intent Classification
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
import uvicorn
from datetime import datetime
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Intent Classification API",
    description="API for classifying user intents in Turkish and English",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models
class IntentRequest(BaseModel):
    text: str = Field(..., description="Text to classify", min_length=1, max_length=1000)
    model_type: Optional[str] = Field("baseline", description="Model type: baseline or transformer")
    return_probabilities: Optional[bool] = Field(True, description="Whether to return all class probabilities")
    confidence_threshold: Optional[float] = Field(0.7, description="Minimum confidence threshold for prediction")

class IntentResponse(BaseModel):
    intent: str = Field(..., description="Predicted intent")
    confidence: float = Field(..., description="Confidence score")
    probabilities: Optional[Dict[str, float]] = Field(None, description="All class probabilities")
    model_type: str = Field(..., description="Model type used")
    processing_time: float = Field(..., description="Processing time in seconds")
    timestamp: str = Field(..., description="Prediction timestamp")

class BatchIntentRequest(BaseModel):
    texts: List[str] = Field(..., description="List of texts to classify", min_length=1, max_length=100)
    model_type: Optional[str] = Field("baseline", description="Model type: baseline or transformer")
    return_probabilities: Optional[bool] = Field(True, description="Whether to return all class probabilities")
    confidence_threshold: Optional[float] = Field(0.7, description="Minimum confidence threshold for prediction")

class BatchIntentResponse(BaseModel):
    predictions: List[IntentResponse] = Field(..., description="List of predictions")
    total_processing_time: float = Field(..., description="Total processing time in seconds")
    model_type: str = Field(..., description="Model type used")

class HealthResponse(BaseModel):
    status: str = Field(..., description="Service status")
    models_loaded: Dict[str, bool] = Field(..., description="Model loading status")
    timestamp: str = Field(..., description="Health check timestamp")

# Global model storage
models = {}

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load models on startup"""
    logger.info("Starting up...")
    
    # Try to load both models
    baseline_loaded = load_model("baseline")
    transformer_loaded = load_model("transformer")
    
    logger.info(f"Startup completed. Models loaded: baseline={baseline_loaded}, transformer={transformer_loaded}")
    yield
    logger.info("Shutting down...")

# Initialize FastAPI app
app = FastAPI(
    title="Intent Classification API",
    description="API for classifying user intents in Turkish and English",
    version="1.0.0",
    lifespan=lifespan
)

def load_model(model_type: str = "baseline"):
    """Load model based on type"""
    try:
        if model_type == "baseline":
            from src.training.train_baseline import BaselineIntentClassifier
            model = BaselineIntentClassifier()
            model_dir = Path("artifacts/baseline_model")
            model.load_model(model_dir)
        elif model_type == "transformer":
            from src.models.transformer_classifier import TransformerIntentClassifier
            model = TransformerIntentClassifier()
            model_dir = Path("artifacts/transformer_model")
            model.load_model(model_dir)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
        
        models[model_type] = model
        logger.info(f"Model {model_type} loaded successfully")
        return True
    except Exception as e:
        logger.error(f"Error loading model {model_type}: {e}")
        return False

def get_model(model_type: str = "baseline"):
    """Get loaded model"""
    if model_type not in models:
        if not load_model(model_type):
            raise HTTPException(status_code=500, detail=f"Model {model_type} not available")
    return models[model_type]

@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint"""
    return HealthResponse(
        status="healthy",
        models_loaded={
            "baseline": "baseline" in models,
            "transformer": "transformer" in models
        },
        timestamp=datetime.now().isoformat()
    )

@app.post("/intent", response_model=IntentResponse)
async def classify_intent(request: IntentRequest):
    """Classify single text intent"""
    start_time = time.perf_counter()
    
    try:
        # Get model
        model = get_model(request.model_type)
        
        # Make prediction
        prediction = model.predict(
            request.text, 
            return_confidence=request.return_probabilities
        )
        
        # Check confidence threshold
        if prediction['confidence'] < request.confidence_threshold:
            prediction['intent'] = 'abstain'
            prediction['confidence'] = 1.0 - prediction['confidence']
        
        # Calculate processing time (ensure > 0)
        processing_time = max(1e-6, time.perf_counter() - start_time)
        
        # Prepare response
        response = IntentResponse(
            intent=prediction['intent'],
            confidence=prediction['confidence'],
            probabilities=prediction.get('probabilities') if request.return_probabilities else None,
            model_type=request.model_type,
            processing_time=processing_time,
            timestamp=datetime.now().isoformat()
        )
        
        logger.info(f"Prediction completed: {request.text[:50]}... -> {prediction['intent']}")
        return response
        
    except Exception as e:
        logger.error(f"Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/intent/batch", response_model=BatchIntentResponse)
async def classify_intent_batch(request: BatchIntentRequest):
    """Classify multiple texts"""
    start_time = time.perf_counter()
    
    try:
        # Get model
        model = get_model(request.model_type)
        
        # Make predictions
        predictions = []
        for text in request.texts:
            prediction = model.predict(
                text, 
                return_confidence=request.return_probabilities
            )
            
            # Check confidence threshold
            if prediction['confidence'] < request.confidence_threshold:
                prediction['intent'] = 'abstain'
                prediction['confidence'] = 1.0 - prediction['confidence']
            
            # Create response
            response = IntentResponse(
                intent=prediction['intent'],
                confidence=prediction['confidence'],
                probabilities=prediction.get('probabilities') if request.return_probabilities else None,
                model_type=request.model_type,
                processing_time=0.0,  # Individual processing time not calculated
                timestamp=datetime.now().isoformat()
            )
            predictions.append(response)
        
        # Calculate total processing time
        total_processing_time = max(1e-6, time.perf_counter() - start_time)
        
        return BatchIntentResponse(
            predictions=predictions,
            total_processing_time=total_processing_time,
            model_type=request.model_type
        )
        
    except Exception as e:
        logger.error(f"Error in batch prediction: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/intents")
async def get_intents():
    """Get list of available intents"""
    try:
        # Load intents from file
        intents_file = Path("data/intents.yaml")
        if intents_file.exists():
            import yaml
            with open(intents_file, 'r', encoding='utf-8') as f:
                intents = yaml.safe_load(f)
            return {"intents": intents}
        else:
            # Fallback to hardcoded intents
            intents = {
                "opening_hours": "Kütüphane çalışma saatleri hakkında sorular",
                "fine_policy": "Ceza politikası, gecikme ücretleri hakkında sorular",
                "borrow_limit": "Ödünç alma limitleri hakkında sorular",
                "room_booking": "Oda rezervasyonu, çalışma alanı rezervasyonu",
                "ebook_access": "E-kitap erişimi, dijital kaynaklar",
                "wifi": "WiFi bağlantı sorunları",
                "lost_card": "Kayıp kart, kart yenileme",
                "renewal": "Üyelik yenileme, kart yenileme",
                "ill": "Hastalık, sağlık durumu",
                "events": "Etkinlikler, seminerler, workshoplar",
                "complaint": "Şikayetler, memnuniyetsizlik",
                "other": "Diğer konular, belirsiz sorular"
            }
            return {"intents": intents}
    except Exception as e:
        logger.error(f"Error getting intents: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/models")
async def get_models():
    """Get available models"""
    return {
        "available_models": list(models.keys()),
        "default_model": "baseline"
    }

@app.get("/")
async def root():
    """Root endpoint"""
    return {
        "message": "Intent Classification API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info"
    )

# Explicit OPTIONS handler for CORS preflight used by tests
@app.options("/intent")
async def options_intent():
    from fastapi import Response
    response = Response(content="{}", media_type="application/json")
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "*"
    return response
