#!/usr/bin/env python3
"""
Baseline Model Training: TF-IDF + Logistic Regression
"""

import json
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, f1_score
from sklearn.calibration import CalibratedClassifierCV
import joblib
from collections import Counter

class BaselineIntentClassifier:
    def __init__(self, max_features: int = 10000, ngram_range: Tuple[int, int] = (1, 2)):
        """Initialize baseline classifier"""
        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            ngram_range=ngram_range,
            stop_words=None,  # We'll handle Turkish/English stop words manually
            lowercase=True,
            strip_accents='unicode'
        )
        self.classifier = LogisticRegression(
            random_state=42,
            max_iter=1000,
            class_weight='balanced'
        )
        self.calibrated_classifier = None
        self.intent_labels = None
        self.is_trained = False
    
    def load_data(self, data_file: Path) -> Tuple[List[str], List[str]]:
        """Load and prepare data"""
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = [item['text'] for item in data]
        intents = [item['intent'] for item in data]
        
        print(f"Loaded {len(texts)} examples")
        print(f"Intent distribution: {Counter(intents)}")
        
        return texts, intents
    
    def preprocess_text(self, text: str) -> str:
        """Basic text preprocessing"""
        # Convert to lowercase
        text = text.lower()
        
        # Remove extra whitespace
        text = ' '.join(text.split())
        
        # Basic punctuation handling
        text = text.replace('?', ' ?')
        text = text.replace('!', ' !')
        text = text.replace('.', ' .')
        
        return text
    
    def prepare_data(self, texts: List[str], intents: List[str]) -> Tuple[List[str], List[str]]:
        """Prepare data for training"""
        # Preprocess texts
        processed_texts = [self.preprocess_text(text) for text in texts]
        
        # Get unique intents and create label mapping
        unique_intents = sorted(list(set(intents)))
        self.intent_labels = {intent: idx for idx, intent in enumerate(unique_intents)}
        self.label_to_intent = {idx: intent for intent, idx in self.intent_labels.items()}
        self.labels = list(self.intent_labels.keys())
        
        # Convert intents to numeric labels
        numeric_intents = [self.intent_labels[intent] for intent in intents]
        
        print(f"Intent labels: {self.intent_labels}")
        print(f"Number of classes: {len(unique_intents)}")
        
        return processed_texts, numeric_intents
    
    def train(self, texts: List[str], intents: List[str], test_size: float = 0.2):
        """Train the baseline model"""
        print("Training baseline model...")
        
        # Prepare data
        processed_texts, numeric_intents = self.prepare_data(texts, intents)
        
        # Ensure split works on very small datasets (skip stratify if any class has <2 samples)
        from collections import Counter as _Counter
        class_counts_before_split = _Counter(numeric_intents)
        min_per_class_overall = min(class_counts_before_split.values()) if len(class_counts_before_split) > 0 else 0
        use_stratify = min_per_class_overall >= 2
        if not use_stratify:
            print("Skipping stratified split due to classes with <2 samples; using random split.")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, numeric_intents,
            test_size=test_size,
            random_state=42,
            stratify=numeric_intents if use_stratify else None
        )
        
        print(f"Training set: {len(X_train)} examples")
        print(f"Test set: {len(X_test)} examples")
        
        # Vectorize texts
        print("Vectorizing texts...")
        X_train_vectorized = self.vectorizer.fit_transform(X_train)
        X_test_vectorized = self.vectorizer.transform(X_test)
        
        print(f"Feature matrix shape: {X_train_vectorized.shape}")
        
        # Train classifier
        print("Training logistic regression...")
        self.classifier.fit(X_train_vectorized, y_train)
        
        # Calibrate classifier for confidence scores
        # Attempt calibration with dynamic CV depending on class counts
        print("Calibrating classifier...")
        # Determine minimum samples per class in y_train
        from collections import Counter as _Counter
        class_counts = _Counter(y_train)
        min_per_class = min(class_counts.values()) if len(class_counts) > 0 else 0
        cv_folds = min(3, min_per_class)
        if cv_folds >= 2:
            self.calibrated_classifier = CalibratedClassifierCV(
                self.classifier,
                method='sigmoid',
                cv=cv_folds
            )
            self.calibrated_classifier.fit(X_train_vectorized, y_train)
        else:
            print("Not enough samples per class for calibration; using uncalibrated probabilities.")
            self.calibrated_classifier = None
        
        # Evaluate on test set
        y_pred = self.classifier.predict(X_test_vectorized)
        if self.calibrated_classifier is not None:
            y_pred_proba = self.calibrated_classifier.predict_proba(X_test_vectorized)
        elif hasattr(self.classifier, 'predict_proba'):
            y_pred_proba = self.classifier.predict_proba(X_test_vectorized)
        else:
            y_pred_proba = None
        
        # Calculate metrics
        f1 = f1_score(y_test, y_pred, average='weighted')
        print(f"Test F1 Score: {f1:.4f}")
        
        # Print classification report
        intent_names = [self.label_to_intent[label] for label in y_test]
        pred_intent_names = [self.label_to_intent[label] for label in y_pred]
        
        print("\nClassification Report:")
        print(classification_report(intent_names, pred_intent_names))
        
        # Cross-validation
        # Cross-validation with safe folds for small datasets
        from collections import Counter as _Counter
        class_counts_cv = _Counter(y_train)
        min_per_class_cv = min(class_counts_cv.values()) if len(class_counts_cv) > 0 else 0
        cv_folds = max(2, min(5, min_per_class_cv))
        try:
            cv_scores = cross_val_score(
                self.classifier, X_train_vectorized, y_train,
                cv=cv_folds, scoring='f1_weighted'
            )
            print(f"Cross-validation F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
        except Exception as _e:
            cv_scores = np.array([])
            print("Cross-validation skipped due to small dataset constraints.")
        
        self.is_trained = True
        
        return {
            'f1_score': f1,
            'cv_scores': cv_scores.tolist() if cv_scores.size > 0 else [],
            'classification_report': classification_report(intent_names, pred_intent_names, output_dict=True)
        }
    
    def predict(self, text: str, return_confidence: bool = True) -> Dict[str, Any]:
        """Predict intent for a single text"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Preprocess text
        processed_text = self.preprocess_text(text)
        
        # Vectorize
        text_vectorized = self.vectorizer.transform([processed_text])
        
        # Predict
        probabilities = None
        if return_confidence:
            if self.calibrated_classifier is not None:
                probabilities = self.calibrated_classifier.predict_proba(text_vectorized)[0]
            else:
                # Fall back to classifier probabilities
                if hasattr(self.classifier, 'predict_proba'):
                    probabilities = self.classifier.predict_proba(text_vectorized)[0]
        
        if probabilities is not None:
            predicted_label = int(np.argmax(probabilities))
            confidence = float(probabilities[predicted_label])
        else:
            predicted_label = int(self.classifier.predict(text_vectorized)[0])
            confidence = None
        
        predicted_intent = self.label_to_intent[predicted_label]
        
        result = {
            'intent': predicted_intent,
            'confidence': confidence,
            'probabilities': None
        }

        if return_confidence and probabilities is not None:
            # Add all class probabilities, ensure native floats
            result['probabilities'] = {
                self.label_to_intent[i]: float(prob)
                for i, prob in enumerate(probabilities)
            }
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict intents for multiple texts"""
        return [self.predict(text) for text in texts]
    
    def save_model(self, model_dir: Path):
        """Save trained model"""
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save vectorizer
        vectorizer_file = model_dir / "vectorizer.pkl"
        with open(vectorizer_file, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        # Save classifier
        classifier_file = model_dir / "classifier.pkl"
        with open(classifier_file, 'wb') as f:
            pickle.dump(self.classifier, f)
        
        # Save calibrated classifier
        if self.calibrated_classifier is not None:
            calibrated_file = model_dir / "calibrated_classifier.pkl"
            with open(calibrated_file, 'wb') as f:
                pickle.dump(self.calibrated_classifier, f)
        
        # Save label mappings
        label_file = model_dir / "labels.json"
        with open(label_file, 'w', encoding='utf-8') as f:
            json.dump({
                'intent_to_label': self.intent_labels,
                'label_to_intent': self.label_to_intent
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Model saved to {model_dir}")
    
    def load_model(self, model_dir: Path):
        """Load trained model"""
        # Load vectorizer
        vectorizer_file = model_dir / "vectorizer.pkl"
        with open(vectorizer_file, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        # Load classifier
        classifier_file = model_dir / "classifier.pkl"
        with open(classifier_file, 'rb') as f:
            self.classifier = pickle.load(f)
        
        # Load calibrated classifier if exists
        calibrated_file = model_dir / "calibrated_classifier.pkl"
        if calibrated_file.exists():
            with open(calibrated_file, 'rb') as f:
                self.calibrated_classifier = pickle.load(f)
        
        # Load label mappings
        label_file = model_dir / "labels.json"
        with open(label_file, 'r', encoding='utf-8') as f:
            labels = json.load(f)
            self.intent_labels = labels['intent_to_label']
            # JSON serializes int keys as strings; convert back to int
            self.label_to_intent = {int(k): v for k, v in labels['label_to_intent'].items()}
            self.labels = list(self.intent_labels.keys())
        
        self.is_trained = True
        print(f"Model loaded from {model_dir}")

def main():
    """Main function"""
    print("Training baseline intent classifier...")
    
    # Setup paths
    data_dir = Path("data")
    clean_dir = data_dir / "clean"
    model_dir = Path("artifacts/baseline_model")
    
    # Input file
    input_file = clean_dir / "deduplicated_data.json"
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found")
        print("Please run dedupe.py first")
        return
    
    # Initialize classifier
    classifier = BaselineIntentClassifier()
    
    # Load data
    texts, intents = classifier.load_data(input_file)
    
    # Train model
    results = classifier.train(texts, intents)
    
    # Save model
    classifier.save_model(model_dir)
    
    print("Baseline model training completed!")
    print(f"F1 Score: {results['f1_score']:.4f}")

if __name__ == "__main__":
    main()
