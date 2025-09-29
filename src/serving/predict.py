#!/usr/bin/env python3
"""
Prediction utilities for intent classification
"""

import json
import time
from pathlib import Path
from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)

class IntentPredictor:
    def __init__(self, model_type: str = "baseline"):
        """Initialize predictor with model type"""
        self.model_type = model_type
        self.model = None
        self.is_loaded = False
        
    def load_model(self):
        """Load the specified model"""
        try:
            if self.model_type == "baseline":
                from src.training.train_baseline import BaselineIntentClassifier
                self.model = BaselineIntentClassifier()
                model_dir = Path("artifacts/baseline_model")
                self.model.load_model(model_dir)
            elif self.model_type == "transformer":
                from src.training.train_transformer import TransformerIntentClassifier
                self.model = TransformerIntentClassifier()
                model_dir = Path("artifacts/transformer_model")
                self.model.load_model(model_dir)
            else:
                raise ValueError(f"Unknown model type: {self.model_type}")
            
            self.is_loaded = True
            logger.info(f"Model {self.model_type} loaded successfully")
            return True
        except Exception as e:
            logger.error(f"Error loading model {self.model_type}: {e}")
            return False
    
    def predict(self, text: str, confidence_threshold: float = 0.5) -> Dict[str, Any]:
        """Predict intent for a single text"""
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError(f"Model {self.model_type} not available")
        
        start_time = time.time()
        
        try:
            # Make prediction
            prediction = self.model.predict(text, return_confidence=True)
            
            # Apply confidence threshold
            if prediction['confidence'] < confidence_threshold:
                prediction['intent'] = 'abstain'
                prediction['confidence'] = 1.0 - prediction['confidence']
            
            # Add processing time
            prediction['processing_time'] = time.time() - start_time
            
            return prediction
            
        except Exception as e:
            logger.error(f"Error in prediction: {e}")
            raise RuntimeError(f"Prediction failed: {e}")
    
    def predict_batch(self, texts: List[str], confidence_threshold: float = 0.5) -> List[Dict[str, Any]]:
        """Predict intents for multiple texts"""
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError(f"Model {self.model_type} not available")
        
        start_time = time.time()
        predictions = []
        
        try:
            for text in texts:
                prediction = self.predict(text, confidence_threshold)
                predictions.append(prediction)
            
            # Add total processing time
            total_time = time.time() - start_time
            for prediction in predictions:
                prediction['total_processing_time'] = total_time
            
            return predictions
            
        except Exception as e:
            logger.error(f"Error in batch prediction: {e}")
            raise RuntimeError(f"Batch prediction failed: {e}")
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get model information"""
        if not self.is_loaded:
            return {"status": "not_loaded"}
        
        return {
            "model_type": self.model_type,
            "status": "loaded",
            "intent_labels": getattr(self.model, 'intent_labels', None),
            "label_to_intent": getattr(self.model, 'label_to_intent', None)
        }

def main():
    """Main function for testing"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Intent Classification Predictor")
    parser.add_argument("--text", type=str, help="Text to classify")
    parser.add_argument("--model", type=str, default="baseline", choices=["baseline", "transformer"])
    parser.add_argument("--threshold", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--batch", type=str, help="JSON file with texts to classify")
    
    args = parser.parse_args()
    
    # Initialize predictor
    predictor = IntentPredictor(model_type=args.model)
    
    if args.batch:
        # Batch prediction
        with open(args.batch, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if isinstance(data, list):
            texts = data
        elif isinstance(data, dict) and 'texts' in data:
            texts = data['texts']
        else:
            raise ValueError("Invalid batch file format")
        
        predictions = predictor.predict_batch(texts, args.threshold)
        
        # Save results
        output_file = Path("predictions.json")
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(predictions, f, ensure_ascii=False, indent=2)
        
        print(f"Batch predictions saved to {output_file}")
        
    elif args.text:
        # Single prediction
        prediction = predictor.predict(args.text, args.threshold)
        print(json.dumps(prediction, ensure_ascii=False, indent=2))
        
    else:
        # Interactive mode
        print("Interactive mode. Type 'quit' to exit.")
        while True:
            text = input("\nEnter text to classify: ").strip()
            if text.lower() == 'quit':
                break
            
            if text:
                try:
                    prediction = predictor.predict(text, args.threshold)
                    print(f"Intent: {prediction['intent']}")
                    print(f"Confidence: {prediction['confidence']:.4f}")
                    if 'probabilities' in prediction:
                        print("All probabilities:")
                        for intent, prob in prediction['probabilities'].items():
                            print(f"  {intent}: {prob:.4f}")
                except Exception as e:
                    print(f"Error: {e}")

if __name__ == "__main__":
    main()
