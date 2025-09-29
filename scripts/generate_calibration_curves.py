#!/usr/bin/env python3
"""
Generate Calibration Curves for Model Evaluation
"""

import json
import sys
import os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.train_baseline import BaselineIntentClassifier
from src.models.transformer_classifier import TransformerIntentClassifier

def load_test_data():
    """Load test data"""
    with open("data/real_test_examples.json", 'r', encoding='utf-8') as f:
        return json.load(f)

def get_model_predictions(model, test_data):
    """Get model predictions with confidence scores"""
    texts = [ex['text'] for ex in test_data]
    true_labels = [ex['intent'] for ex in test_data]
    
    predictions = []
    confidences = []
    
    for text in texts:
        try:
            pred = model.predict(text, return_confidence=True)
            predictions.append(pred['intent'])
            confidences.append(pred['confidence'])
        except Exception as e:
            print(f"Error predicting '{text}': {e}")
            predictions.append('error')
            confidences.append(0.0)
    
    return predictions, confidences, true_labels

def plot_calibration_curve(y_true, y_prob, model_name, save_path):
    """Plot calibration curve"""
    # Convert to binary: correct (1) or incorrect (0)
    y_binary = [1 if pred == true else 0 for pred, true in zip(y_true, y_prob)]
    
    # Get confidence scores (assuming y_prob contains confidence scores)
    confidence_scores = y_prob
    
    # Create calibration curve
    fraction_of_positives, mean_predicted_value = calibration_curve(
        y_binary, confidence_scores, n_bins=10
    )
    
    # Calculate Brier score
    brier_score = brier_score_loss(y_binary, confidence_scores)
    
    # Create plot
    plt.figure(figsize=(10, 8))
    
    # Perfect calibration line
    plt.plot([0, 1], [0, 1], 'k--', label='Perfect Calibration')
    
    # Model calibration curve
    plt.plot(mean_predicted_value, fraction_of_positives, 'b-', 
             label=f'{model_name} (Brier Score: {brier_score:.3f})')
    
    plt.xlabel('Mean Predicted Probability')
    plt.ylabel('Fraction of Positives')
    plt.title(f'Calibration Curve - {model_name}')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Save plot
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    
    return brier_score

def generate_calibration_curves():
    """Generate calibration curves for both models"""
    print("Generating Calibration Curves...")
    
    # Load test data
    test_data = load_test_data()
    print(f"Loaded {len(test_data)} test examples")
    
    # Create output directory
    output_dir = Path("data/calibration_curves")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    results = {}
    
    # Test Baseline Model
    print("\nTesting Baseline Model...")
    try:
        baseline_model = BaselineIntentClassifier()
        baseline_model.load_model(Path("artifacts/baseline_model"))
        
        predictions, confidences, true_labels = get_model_predictions(baseline_model, test_data)
        
        # Calculate accuracy
        correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
        accuracy = correct / len(predictions)
        
        print(f"Baseline Model Accuracy: {accuracy:.2%}")
        
        # Generate calibration curve
        brier_score = plot_calibration_curve(
            predictions, confidences, "Baseline Model",
            output_dir / "baseline_calibration.png"
        )
        
        results['baseline'] = {
            'accuracy': accuracy,
            'brier_score': brier_score,
            'predictions': predictions,
            'confidences': confidences,
            'true_labels': true_labels
        }
        
    except Exception as e:
        print(f"Error testing baseline model: {e}")
        results['baseline'] = {'error': str(e)}
    
    # Test Transformer Model
    print("\nTesting Transformer Model...")
    try:
        transformer_model = TransformerIntentClassifier()
        transformer_model.load_model(Path("artifacts/transformer_model"))
        
        predictions, confidences, true_labels = get_model_predictions(transformer_model, test_data)
        
        # Calculate accuracy
        correct = sum(1 for pred, true in zip(predictions, true_labels) if pred == true)
        accuracy = correct / len(predictions)
        
        print(f"Transformer Model Accuracy: {accuracy:.2%}")
        
        # Generate calibration curve
        brier_score = plot_calibration_curve(
            predictions, confidences, "Transformer Model",
            output_dir / "transformer_calibration.png"
        )
        
        results['transformer'] = {
            'accuracy': accuracy,
            'brier_score': brier_score,
            'predictions': predictions,
            'confidences': confidences,
            'true_labels': true_labels
        }
        
    except Exception as e:
        print(f"Error testing transformer model: {e}")
        results['transformer'] = {'error': str(e)}
    
    # Save results
    with open(output_dir / "calibration_results.json", 'w', encoding='utf-8') as f:
        # Convert numpy types to Python types
        json_results = {}
        for model_name, model_data in results.items():
            if 'error' in model_data:
                json_results[model_name] = model_data
            else:
                json_results[model_name] = {
                    'accuracy': float(model_data['accuracy']),
                    'brier_score': float(model_data['brier_score']),
                    'predictions': model_data['predictions'],
                    'confidences': [float(c) for c in model_data['confidences']],
                    'true_labels': model_data['true_labels']
                }
        
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nCalibration curves saved to {output_dir}")
    print("Files generated:")
    print(f"  - baseline_calibration.png")
    print(f"  - transformer_calibration.png")
    print(f"  - calibration_results.json")
    
    return results

def main():
    """Main function"""
    try:
        results = generate_calibration_curves()
        
        # Print summary
        print("\n" + "="*50)
        print("CALIBRATION SUMMARY")
        print("="*50)
        
        for model_name, model_data in results.items():
            if 'error' in model_data:
                print(f"{model_name.upper()}: ERROR - {model_data['error']}")
            else:
                print(f"{model_name.upper()}:")
                print(f"  Accuracy: {model_data['accuracy']:.2%}")
                print(f"  Brier Score: {model_data['brier_score']:.3f}")
                print(f"  (Lower Brier score = better calibration)")
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()