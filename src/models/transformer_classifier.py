#!/usr/bin/env python3
"""
Transformer-based Intent Classifier
Uses DistilBERT for intent classification
"""

import json
import torch
import numpy as np
from pathlib import Path
from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransformerIntentClassifier:
    def __init__(self, model_path="artifacts/transformer_model"):
        self.model_path = Path(model_path)
        self.model = None
        self.tokenizer = None
        self.label2id = {}
        self.id2label = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
    def load_model(self, model_path: str = None):
        """Load the trained transformer model"""
        if model_path:
            self.model_path = Path(model_path)
        
        try:
            # Load tokenizer
            self.tokenizer = DistilBertTokenizer.from_pretrained(str(self.model_path))
            
            # Load model
            self.model = DistilBertForSequenceClassification.from_pretrained(str(self.model_path))
            self.model.to(self.device)
            self.model.eval()
            
            # Load label mappings
            with open(self.model_path / "label2id.json", 'r', encoding='utf-8') as f:
                self.label2id = json.load(f)
            
            with open(self.model_path / "id2label.json", 'r', encoding='utf-8') as f:
                self.id2label = json.load(f)
            
            logger.info(f"Transformer model loaded from {self.model_path}")
            logger.info(f"Using device: {self.device}")
            
        except Exception as e:
            logger.error(f"Error loading transformer model: {e}")
            raise
    
    def predict(self, text: str, return_confidence: bool = True) -> Dict[str, Any]:
        """Predict intent for a single text"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            padding=True,
            max_length=128
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        
        # Get predictions
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Get predicted class
        predicted_id = torch.argmax(probabilities, dim=-1).item()
        predicted_intent = self.id2label[str(predicted_id)]
        confidence = probabilities[0][predicted_id].item()
        
        result = {
            "intent": predicted_intent,
            "confidence": confidence
        }
        
        if return_confidence:
            # Get all probabilities
            all_probs = probabilities[0].cpu().numpy()
            probabilities_dict = {
                self.id2label[str(i)]: float(prob) 
                for i, prob in enumerate(all_probs)
            }
            result["probabilities"] = probabilities_dict
        
        return result
    
    def predict_batch(self, texts: List[str], return_confidence: bool = True) -> List[Dict[str, Any]]:
        """Predict intents for multiple texts"""
        results = []
        for text in texts:
            try:
                result = self.predict(text, return_confidence)
                results.append(result)
            except Exception as e:
                logger.error(f"Error predicting for text '{text}': {e}")
                results.append({
                    "intent": "error",
                    "confidence": 0.0,
                    "probabilities": {} if return_confidence else None
                })
        
        return results

def main():
    """Test the transformer classifier"""
    classifier = TransformerIntentClassifier()
    
    try:
        classifier.load_model()
        
        # Test prediction
        test_text = "Kütüphane saat kaçta açılıyor?"
        result = classifier.predict(test_text)
        
        print(f"Text: {test_text}")
        print(f"Predicted Intent: {result['intent']}")
        print(f"Confidence: {result['confidence']:.4f}")
        
        if 'probabilities' in result:
            print("Top 3 probabilities:")
            sorted_probs = sorted(result['probabilities'].items(), 
                                key=lambda x: x[1], reverse=True)[:3]
            for intent, prob in sorted_probs:
                print(f"  {intent}: {prob:.4f}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()
