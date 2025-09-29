#!/usr/bin/env python3
"""
Transformer Model Training: DistilBERT Fine-tuning
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from sklearn.metrics import classification_report, f1_score, accuracy_score
from sklearn.model_selection import train_test_split
from datasets import Dataset
import joblib
from collections import Counter
import os

class TransformerIntentClassifier:
    def __init__(self, model_name: str = "distilbert-base-multilingual-cased", max_length: int = 128):
        """Initialize transformer classifier"""
        self.model_name = model_name
        self.max_length = max_length
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = None
        self.intent_labels = None
        self.label_to_intent = None
        self.is_trained = False
        
        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
    
    def load_data(self, data_file: Path) -> Tuple[List[str], List[str]]:
        """Load and prepare data"""
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = [item['text'] for item in data]
        intents = [item['intent'] for item in data]
        
        print(f"Loaded {len(texts)} examples")
        print(f"Intent distribution: {Counter(intents)}")
        
        return texts, intents
    
    def prepare_data(self, texts: List[str], intents: List[str]) -> Tuple[List[str], List[int]]:
        """Prepare data for training"""
        # Get unique intents and create label mapping
        unique_intents = sorted(list(set(intents)))
        self.intent_labels = {intent: idx for idx, intent in enumerate(unique_intents)}
        self.label_to_intent = {idx: intent for intent, idx in self.intent_labels.items()}
        
        # Convert intents to numeric labels
        numeric_intents = [self.intent_labels[intent] for intent in intents]
        
        print(f"Intent labels: {self.intent_labels}")
        print(f"Number of classes: {len(unique_intents)}")
        
        return texts, numeric_intents
    
    def tokenize_function(self, examples):
        """Tokenize texts for training"""
        return self.tokenizer(
            examples['text'],
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
    
    def create_dataset(self, texts: List[str], intents: List[int]) -> Dataset:
        """Create HuggingFace dataset"""
        # Create dataset dictionary
        dataset_dict = {
            'text': texts,
            'labels': intents
        }
        
        # Create dataset
        dataset = Dataset.from_dict(dataset_dict)
        
        # Tokenize
        tokenized_dataset = dataset.map(
            self.tokenize_function,
            batched=True,
            remove_columns=dataset.column_names
        )
        
        return tokenized_dataset
    
    def compute_metrics(self, eval_pred):
        """Compute metrics for evaluation"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=1)
        
        f1 = f1_score(labels, predictions, average='weighted')
        accuracy = accuracy_score(labels, predictions)
        
        return {
            'f1': f1,
            'accuracy': accuracy
        }
    
    def train(self, texts: List[str], intents: List[str], test_size: float = 0.2):
        """Train the transformer model"""
        print("Training transformer model...")
        
        # Prepare data
        processed_texts, numeric_intents = self.prepare_data(texts, intents)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            processed_texts, numeric_intents, 
            test_size=test_size, 
            random_state=42, 
            stratify=numeric_intents
        )
        
        print(f"Training set: {len(X_train)} examples")
        print(f"Test set: {len(X_test)} examples")
        
        # Create datasets
        train_dataset = self.create_dataset(X_train, y_train)
        test_dataset = self.create_dataset(X_test, y_test)
        
        # Initialize model
        num_labels = len(self.intent_labels)
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=num_labels,
            problem_type="single_label_classification"
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir="./artifacts/transformer_model",
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=500,
            weight_decay=0.01,
            logging_dir="./logs",
            logging_steps=100,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            save_total_limit=2,
            report_to=None  # Disable wandb
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=test_dataset,
            compute_metrics=self.compute_metrics,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=2)]
        )
        
        # Train model
        print("Starting training...")
        trainer.train()
        
        # Evaluate on test set
        print("Evaluating on test set...")
        eval_results = trainer.evaluate()
        
        print(f"Test F1 Score: {eval_results['eval_f1']:.4f}")
        print(f"Test Accuracy: {eval_results['eval_accuracy']:.4f}")
        
        # Get predictions for detailed analysis
        predictions = trainer.predict(test_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        
        # Print classification report
        intent_names = [self.label_to_intent[label] for label in y_test]
        pred_intent_names = [self.label_to_intent[label] for label in y_pred]
        
        print("\nClassification Report:")
        print(classification_report(intent_names, pred_intent_names))
        
        self.is_trained = True
        
        return {
            'f1_score': eval_results['eval_f1'],
            'accuracy': eval_results['eval_accuracy'],
            'classification_report': classification_report(intent_names, pred_intent_names, output_dict=True)
        }
    
    def predict(self, text: str, return_confidence: bool = True) -> Dict[str, Any]:
        """Predict intent for a single text"""
        if not self.is_trained:
            raise ValueError("Model not trained yet")
        
        # Tokenize text
        inputs = self.tokenizer(
            text,
            truncation=True,
            padding=True,
            max_length=self.max_length,
            return_tensors="pt"
        )
        
        # Move to device
        inputs = {k: v.to(self.device) for k, v in inputs.items()}
        self.model.to(self.device)
        
        # Predict
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits
            probabilities = torch.softmax(logits, dim=-1)
        
        # Get prediction
        predicted_label = torch.argmax(probabilities, dim=-1).item()
        confidence = probabilities[0][predicted_label].item()
        
        predicted_intent = self.label_to_intent[predicted_label]
        
        result = {
            'intent': predicted_intent,
            'confidence': confidence
        }
        
        if return_confidence:
            # Add all class probabilities
            result['probabilities'] = {
                self.label_to_intent[i]: prob.item() 
                for i, prob in enumerate(probabilities[0])
            }
        
        return result
    
    def predict_batch(self, texts: List[str]) -> List[Dict[str, Any]]:
        """Predict intents for multiple texts"""
        return [self.predict(text) for text in texts]
    
    def save_model(self, model_dir: Path):
        """Save trained model"""
        model_dir.mkdir(parents=True, exist_ok=True)
        
        # Save model and tokenizer
        self.model.save_pretrained(model_dir)
        self.tokenizer.save_pretrained(model_dir)
        
        # Save label mappings
        label_file = model_dir / "labels.json"
        with open(label_file, 'w', encoding='utf-8') as f:
            json.dump({
                'intent_to_label': self.intent_labels,
                'label_to_intent': self.label_to_intent,
                'model_name': self.model_name,
                'max_length': self.max_length
            }, f, ensure_ascii=False, indent=2)
        
        print(f"Model saved to {model_dir}")
    
    def load_model(self, model_dir: Path):
        """Load trained model"""
        # Load model and tokenizer
        self.model = DistilBertForSequenceClassification.from_pretrained(model_dir)
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_dir)
        
        # Load label mappings
        label_file = model_dir / "labels.json"
        with open(label_file, 'r', encoding='utf-8') as f:
            labels = json.load(f)
            self.intent_labels = labels['intent_to_label']
            self.label_to_intent = labels['label_to_intent']
            self.model_name = labels.get('model_name', self.model_name)
            self.max_length = labels.get('max_length', self.max_length)
        
        self.is_trained = True
        print(f"Model loaded from {model_dir}")

def main():
    """Main function"""
    print("Training transformer intent classifier...")
    
    # Setup paths
    data_dir = Path("data")
    clean_dir = data_dir / "clean"
    model_dir = Path("artifacts/transformer_model")
    
    # Input file
    input_file = clean_dir / "deduplicated_data.json"
    if not input_file.exists():
        print(f"Error: Input file {input_file} not found")
        print("Please run dedupe.py first")
        return
    
    # Initialize classifier
    classifier = TransformerIntentClassifier()
    
    # Load data
    texts, intents = classifier.load_data(input_file)
    
    # Train model
    results = classifier.train(texts, intents)
    
    # Save model
    classifier.save_model(model_dir)
    
    print("Transformer model training completed!")
    print(f"F1 Score: {results['f1_score']:.4f}")

if __name__ == "__main__":
    main()
