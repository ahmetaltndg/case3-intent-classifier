#!/usr/bin/env python3
"""
Model Calibration: Platt Scaling and Temperature Scaling
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, brier_score_loss
import matplotlib.pyplot as plt
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
import joblib

class ModelCalibrator:
    def __init__(self, model_type: str = "baseline"):
        """Initialize calibrator"""
        self.model_type = model_type
        self.calibrated_model = None
        self.calibration_method = None
        self.is_calibrated = False
        
    def load_model(self, model_dir: Path):
        """Load trained model"""
        if self.model_type == "baseline":
            self._load_baseline_model(model_dir)
        elif self.model_type == "transformer":
            self._load_transformer_model(model_dir)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _load_baseline_model(self, model_dir: Path):
        """Load baseline model"""
        import pickle
        
        # Load vectorizer
        vectorizer_file = model_dir / "vectorizer.pkl"
        with open(vectorizer_file, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Load classifier
        classifier_file = model_dir / "classifier.pkl"
        with open(classifier_file, 'rb') as f:
            self.classifier = pickle.load(f)
        
        # Load label mappings
        label_file = model_dir / "labels.json"
        with open(label_file, 'r', encoding='utf-8') as f:
            labels = json.load(f)
            self.intent_labels = labels['intent_to_label']
            self.label_to_intent = labels['label_to_intent']
    
    def _load_transformer_model(self, model_dir: Path):
        """Load transformer model"""
        from transformers import DistilBertTokenizer, DistilBertForSequenceClassification
        
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        
        # Load label mappings
        label_file = model_dir / "labels.json"
        with open(label_file, 'r', encoding='utf-8') as f:
            labels = json.load(f)
            self.intent_labels = labels['intent_to_label']
            self.label_to_intent = labels['label_to_intent']
    
    def load_validation_data(self, data_file: Path) -> Tuple[List[str], List[str]]:
        """Load validation data"""
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = [item['text'] for item in data]
        intents = [item['intent'] for item in data]
        
        return texts, intents
    
    def get_model_probabilities(self, texts: List[str]) -> np.ndarray:
        """Get model probabilities for texts"""
        if self.model_type == "baseline":
            return self._get_baseline_probabilities(texts)
        elif self.model_type == "transformer":
            return self._get_transformer_probabilities(texts)
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
    
    def _get_baseline_probabilities(self, texts: List[str]) -> np.ndarray:
        """Get baseline model probabilities"""
        # Vectorize texts
        X = self.vectorizer.transform(texts)
        
        # Get probabilities
        probabilities = self.classifier.predict_proba(X)
        
        return probabilities
    
    def _get_transformer_probabilities(self, texts: List[str]) -> np.ndarray:
        """Get transformer model probabilities"""
        probabilities = []
        
        for text in texts:
            # Tokenize
            inputs = self.tokenizer(
                text,
                truncation=True,
                padding=True,
                max_length=128,
                return_tensors="pt"
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                logits = outputs.logits
                probs = torch.softmax(logits, dim=-1)
                probabilities.append(probs[0].numpy())
        
        return np.array(probabilities)
    
    def calibrate_platt_scaling(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """Apply Platt scaling calibration"""
        print("Applying Platt scaling calibration...")
        
        # Use CalibratedClassifierCV for Platt scaling
        self.calibrated_model = CalibratedClassifierCV(
            self.classifier, 
            method='sigmoid', 
            cv=3
        )
        self.calibrated_model.fit(X_cal, y_cal)
        self.calibration_method = "platt_scaling"
        self.is_calibrated = True
        
        print("Platt scaling calibration completed")
    
    def calibrate_temperature_scaling(self, logits: np.ndarray, y_true: np.ndarray):
        """Apply temperature scaling calibration"""
        print("Applying temperature scaling calibration...")
        
        # Convert to torch tensors
        logits_tensor = torch.tensor(logits, dtype=torch.float32)
        y_true_tensor = torch.tensor(y_true, dtype=torch.long)
        
        # Initialize temperature parameter
        temperature = torch.nn.Parameter(torch.ones(1))
        
        # Optimizer
        optimizer = torch.optim.LBFGS([temperature], lr=0.01, max_iter=50)
        
        def eval_loss():
            optimizer.zero_grad()
            loss = torch.nn.functional.cross_entropy(logits_tensor / temperature, y_true_tensor)
            loss.backward()
            return loss
        
        # Optimize temperature
        optimizer.step(eval_loss)
        
        # Store calibrated temperature
        self.calibrated_temperature = temperature.item()
        self.calibration_method = "temperature_scaling"
        self.is_calibrated = True
        
        print(f"Temperature scaling calibration completed. Temperature: {self.calibrated_temperature:.4f}")
    
    def calibrate_isotonic_regression(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """Apply isotonic regression calibration"""
        print("Applying isotonic regression calibration...")
        
        # Use CalibratedClassifierCV for isotonic regression
        self.calibrated_model = CalibratedClassifierCV(
            self.classifier, 
            method='isotonic', 
            cv=3
        )
        self.calibrated_model.fit(X_cal, y_cal)
        self.calibration_method = "isotonic_regression"
        self.is_calibrated = True
        
        print("Isotonic regression calibration completed")
    
    def predict_calibrated(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Get calibrated predictions"""
        if not self.is_calibrated:
            raise ValueError("Model not calibrated yet")
        
        if self.calibration_method == "temperature_scaling":
            return self._predict_temperature_scaled(texts)
        else:
            return self._predict_platt_isotonic(texts)
    
    def _predict_temperature_scaled(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Get temperature scaled predictions"""
        results = []
        
        for text in texts:
            # Get logits
            if self.model_type == "baseline":
                X = self.vectorizer.transform([text])
                logits = self.classifier.decision_function(X)[0]
            else:
                # Transformer case
                inputs = self.tokenizer(
                    text,
                    truncation=True,
                    padding=True,
                    max_length=128,
                    return_tensors="pt"
                )
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    logits = outputs.logits[0].numpy()
            
            # Apply temperature scaling
            scaled_logits = logits / self.calibrated_temperature
            probabilities = torch.softmax(torch.tensor(scaled_logits), dim=-1).numpy()
            
            # Get prediction
            predicted_label = np.argmax(probabilities)
            confidence = probabilities[predicted_label]
            predicted_intent = self.label_to_intent[predicted_label]
            
            result = {
                'intent': predicted_intent,
                'confidence': confidence,
                'probabilities': {
                    self.label_to_intent[i]: prob 
                    for i, prob in enumerate(probabilities)
                }
            }
            
            results.append(result)
        
        return results
    
    def _predict_platt_isotonic(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Get Platt/Isotonic scaled predictions"""
        if self.model_type == "baseline":
            X = self.vectorizer.transform(texts)
            probabilities = self.calibrated_model.predict_proba(X)
        else:
            # For transformer, we need to get probabilities first
            probabilities = self._get_transformer_probabilities(texts)
        
        results = []
        for i, text in enumerate(texts):
            probs = probabilities[i]
            predicted_label = np.argmax(probs)
            confidence = probs[predicted_label]
            predicted_intent = self.label_to_intent[predicted_label]
            
            result = {
                'intent': predicted_intent,
                'confidence': confidence,
                'probabilities': {
                    self.label_to_intent[j]: prob 
                    for j, prob in enumerate(probs)
                }
            }
            
            results.append(result)
        
        return results
    
    def evaluate_calibration(self, texts: List[str], true_intents: List[str]) -> Dict[str, float]:
        """Evaluate calibration quality"""
        # Get calibrated predictions
        predictions = self.predict_calibrated(texts)
        
        # Convert to arrays
        y_true = [self.intent_labels[intent] for intent in true_intents]
        y_pred = [self.intent_labels[pred['intent']] for pred in predictions]
        y_prob = [pred['confidence'] for pred in predictions]
        
        # Calculate metrics
        log_loss_score = log_loss(y_true, y_prob)
        brier_score = brier_score_loss(y_true, y_prob)
        
        # Calculate reliability diagram
        reliability = self._calculate_reliability_diagram(y_true, y_prob)
        
        return {
            'log_loss': log_loss_score,
            'brier_score': brier_score,
            'reliability': reliability
        }
    
    def _calculate_reliability_diagram(self, y_true: List[int], y_prob: List[float]) -> Dict[str, Any]:
        """Calculate reliability diagram"""
        # Bin probabilities
        bins = np.linspace(0, 1, 11)
        bin_centers = (bins[:-1] + bins[1:]) / 2
        
        # Calculate accuracy and confidence for each bin
        accuracies = []
        confidences = []
        counts = []
        
        for i in range(len(bins) - 1):
            mask = (np.array(y_prob) >= bins[i]) & (np.array(y_prob) < bins[i + 1])
            if i == len(bins) - 2:  # Last bin includes 1.0
                mask = (np.array(y_prob) >= bins[i]) & (np.array(y_prob) <= bins[i + 1])
            
            if np.sum(mask) > 0:
                bin_true = np.array(y_true)[mask]
                bin_prob = np.array(y_prob)[mask]
                
                accuracy = np.mean(bin_true == np.argmax([bin_prob]))  # Simplified
                confidence = np.mean(bin_prob)
                count = np.sum(mask)
                
                accuracies.append(accuracy)
                confidences.append(confidence)
                counts.append(count)
            else:
                accuracies.append(0)
                confidences.append(bin_centers[i])
                counts.append(0)
        
        return {
            'bin_centers': bin_centers.tolist(),
            'accuracies': accuracies,
            'confidences': confidences,
            'counts': counts
        }
    
    def save_calibrated_model(self, model_dir: Path):
        """Save calibrated model"""
        model_dir.mkdir(parents=True, exist_ok=True)
        
        if self.calibration_method == "temperature_scaling":
            # Save temperature parameter
            temp_file = model_dir / "temperature.json"
            with open(temp_file, 'w') as f:
                json.dump({
                    'temperature': self.calibrated_temperature,
                    'method': self.calibration_method
                }, f)
        else:
            # Save calibrated model
            calibrated_file = model_dir / "calibrated_model.pkl"
            joblib.dump(self.calibrated_model, calibrated_file)
            
            # Save calibration info
            info_file = model_dir / "calibration_info.json"
            with open(info_file, 'w') as f:
                json.dump({
                    'method': self.calibration_method,
                    'model_type': self.model_type
                }, f)
        
        print(f"Calibrated model saved to {model_dir}")
    
    def load_calibrated_model(self, model_dir: Path):
        """Load calibrated model"""
        if self.calibration_method == "temperature_scaling":
            temp_file = model_dir / "temperature.json"
            with open(temp_file, 'r') as f:
                temp_data = json.load(f)
                self.calibrated_temperature = temp_data['temperature']
        else:
            calibrated_file = model_dir / "calibrated_model.pkl"
            self.calibrated_model = joblib.load(calibrated_file)
            
            info_file = model_dir / "calibration_info.json"
            with open(info_file, 'r') as f:
                info = json.load(f)
                self.calibration_method = info['method']
        
        self.is_calibrated = True
        print(f"Calibrated model loaded from {model_dir}")

def main():
    """Main function"""
    print("Model calibration...")
    
    # Setup paths
    data_dir = Path("data")
    clean_dir = data_dir / "clean"
    model_dir = Path("artifacts/baseline_model")
    calibrated_model_dir = Path("artifacts/calibrated_model")
    
    # Load model
    calibrator = ModelCalibrator(model_type="baseline")
    calibrator.load_model(model_dir)
    
    # Load validation data
    validation_file = clean_dir / "deduplicated_data.json"
    texts, intents = calibrator.load_validation_data(validation_file)
    
    # Split for calibration
    from sklearn.model_selection import train_test_split
    X_cal, X_test, y_cal, y_test = train_test_split(
        texts, intents, test_size=0.3, random_state=42
    )
    
    # Get model probabilities for calibration
    X_cal_vectorized = calibrator.vectorizer.transform(X_cal)
    y_cal_numeric = [calibrator.intent_labels[intent] for intent in y_cal]
    
    # Apply Platt scaling
    calibrator.calibrate_platt_scaling(X_cal_vectorized, y_cal_numeric)
    
    # Evaluate calibration
    results = calibrator.evaluate_calibration(X_test, y_test)
    print(f"Calibration results: {results}")
    
    # Save calibrated model
    calibrator.save_calibrated_model(calibrated_model_dir)
    
    print("Model calibration completed!")

if __name__ == "__main__":
    main()
