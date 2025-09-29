#!/usr/bin/env python3
"""
Transformer Model Training Script
Trains DistilBERT for intent classification
"""

import json
import torch
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from transformers import (
    DistilBertTokenizer, DistilBertForSequenceClassification,
    TrainingArguments, Trainer, EarlyStoppingCallback
)
from datasets import Dataset
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class TransformerTrainer:
    def __init__(self, model_name="distilbert-base-multilingual-cased"):
        self.model_name = model_name
        self.tokenizer = DistilBertTokenizer.from_pretrained(model_name)
        self.model = None
        self.label2id = {}
        self.id2label = {}
        
    def load_data(self, data_path="data/clean/deduplicated_data.json"):
        """Load and prepare data"""
        with open(data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = [item['text'] for item in data]
        labels = [item['intent'] for item in data]
        
        # Create label mappings
        unique_labels = sorted(list(set(labels)))
        self.label2id = {label: idx for idx, label in enumerate(unique_labels)}
        self.id2label = {idx: label for label, idx in self.label2id.items()}
        
        # Convert labels to IDs
        label_ids = [self.label2id[label] for label in labels]
        
        return texts, label_ids
    
    def tokenize_data(self, texts, labels):
        """Tokenize texts"""
        def tokenize_function(examples):
            return self.tokenizer(
                examples['text'],
                truncation=True,

                padding='max_length',
                max_length=128
            )
        
        # Create dataset
        dataset = Dataset.from_dict({
            'text': texts,
            'labels': labels
        })
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def train(self, data_path="data/clean/deduplicated_data.json", 
              output_dir="artifacts/transformer_model"):
        """Train the transformer model"""
        logger.info("Loading data...")
        texts, labels = self.load_data(data_path)
        
        # Split data (without stratification due to imbalanced classes)
        train_texts, val_texts, train_labels, val_labels = train_test_split(
            texts, labels, test_size=0.2, random_state=42
        )
        
        logger.info(f"Training samples: {len(train_texts)}")
        logger.info(f"Validation samples: {len(val_texts)}")
        
        # Tokenize data
        train_dataset = self.tokenize_data(train_texts, train_labels)
        val_dataset = self.tokenize_data(val_texts, val_labels)
        
        # Initialize model
        self.model = DistilBertForSequenceClassification.from_pretrained(
            self.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=3,
            per_device_train_batch_size=16,
            per_device_eval_batch_size=16,
            warmup_steps=100,
            weight_decay=0.01,
            logging_dir=f"{output_dir}/logs",
            logging_steps=10,
            eval_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            save_total_limit=2,
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_dataset,
            eval_dataset=val_dataset,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train model
        logger.info("Starting training...")
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Save label mappings
        with open(f"{output_dir}/label2id.json", 'w', encoding='utf-8') as f:
            json.dump(self.label2id, f, ensure_ascii=False, indent=2)
        
        with open(f"{output_dir}/id2label.json", 'w', encoding='utf-8') as f:
            json.dump(self.id2label, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Model saved to {output_dir}")
        
        # Evaluate
        predictions = trainer.predict(val_dataset)
        y_pred = np.argmax(predictions.predictions, axis=1)
        
        logger.info("Validation Results:")
        logger.info(classification_report(val_labels, y_pred, 
                                        labels=list(range(len(self.label2id))),
                                        target_names=list(self.label2id.keys())))
        
        return trainer

def main():
    """Main function"""
    logger.info("Starting transformer model training...")
    
    trainer = TransformerTrainer()
    trainer.train()
    
    logger.info("Training completed!")

if __name__ == "__main__":
    main()
