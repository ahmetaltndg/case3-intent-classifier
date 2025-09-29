#!/usr/bin/env python3
"""
Cross-lingual Performance Test
Tests model performance on Turkish vs English examples
"""

import json
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.train_baseline import BaselineIntentClassifier
from src.models.transformer_classifier import TransformerIntentClassifier
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np

def load_real_test_data():
    """Load real test examples"""
    with open("data/real_test_examples.json", 'r', encoding='utf-8') as f:
        return json.load(f)

def test_cross_lingual_performance():
    """Test cross-lingual performance"""
    print("Testing Cross-lingual Performance...")
    
    # Load test data
    test_data = load_real_test_data()
    
    # Separate by language
    tr_examples = [ex for ex in test_data if ex['language'] == 'tr']
    en_examples = [ex for ex in test_data if ex['language'] == 'en']
    
    print(f"Turkish examples: {len(tr_examples)}")
    print(f"English examples: {len(en_examples)}")
    
    # Load models
    baseline_model = BaselineIntentClassifier()
    baseline_model.load_model(Path("artifacts/baseline_model"))
    
    transformer_model = TransformerIntentClassifier()
    transformer_model.load_model(Path("artifacts/transformer_model"))
    
    results = {}
    
    for model_name, model in [("baseline", baseline_model), ("transformer", transformer_model)]:
        print(f"\n=== {model_name.upper()} MODEL ===")
        
        model_results = {}
        
        for lang, examples in [("Turkish", tr_examples), ("English", en_examples)]:
            print(f"\n--- {lang} Examples ---")
            
            texts = [ex['text'] for ex in examples]
            true_labels = [ex['intent'] for ex in examples]
            
            # Get predictions
            predictions = []
            for text in texts:
                try:
                    pred = model.predict(text, return_confidence=False)
                    predictions.append(pred['intent'])
                except Exception as e:
                    print(f"Error predicting '{text}': {e}")
                    predictions.append('error')
            
            # Calculate accuracy
            correct = sum(1 for true, pred in zip(true_labels, predictions) if true == pred)
            accuracy = correct / len(predictions)
            
            print(f"Accuracy: {accuracy:.2%} ({correct}/{len(predictions)})")
            
            # Classification report
            print("\nClassification Report:")
            print(classification_report(true_labels, predictions, zero_division=0))
            
            model_results[lang] = {
                'accuracy': accuracy,
                'correct': correct,
                'total': len(predictions),
                'predictions': predictions,
                'true_labels': true_labels
            }
        
        results[model_name] = model_results
    
    # Summary comparison
    print("\n" + "="*50)
    print("CROSS-LINGUAL PERFORMANCE SUMMARY")
    print("="*50)
    
    for model_name in ["baseline", "transformer"]:
        print(f"\n{model_name.upper()} MODEL:")
        tr_acc = results[model_name]["Turkish"]["accuracy"]
        en_acc = results[model_name]["English"]["accuracy"]
        print(f"  Turkish:  {tr_acc:.2%}")
        print(f"  English:  {en_acc:.2%}")
        print(f"  Difference: {abs(tr_acc - en_acc):.2%}")
    
    return results

def main():
    """Main function"""
    try:
        results = test_cross_lingual_performance()
        
        # Save results
        with open("data/cross_lingual_results.json", 'w', encoding='utf-8') as f:
            # Convert numpy types to Python types for JSON serialization
            json_results = {}
            for model_name, model_data in results.items():
                json_results[model_name] = {}
                for lang, lang_data in model_data.items():
                    json_results[model_name][lang] = {
                        'accuracy': float(lang_data['accuracy']),
                        'correct': int(lang_data['correct']),
                        'total': int(lang_data['total'])
                    }
            
            json.dump(json_results, f, ensure_ascii=False, indent=2)
        
        print(f"\nResults saved to data/cross_lingual_results.json")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()
