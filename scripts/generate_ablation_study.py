#!/usr/bin/env python3
"""
Generate Ablation Study for Model Evaluation
"""

import json
import sys
import os
import numpy as np
from pathlib import Path
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

# Add src to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.training.train_baseline import BaselineIntentClassifier

def load_data():
    """Load training data"""
    with open("data/clean/deduplicated_data.json", 'r', encoding='utf-8') as f:
        return json.load(f)

def test_feature_ablation():
    """Test different feature combinations"""
    print("Running Feature Ablation Study...")
    
    # Load data
    data = load_data()
    texts = [item['text'] for item in data]
    labels = [item['intent'] for item in data]
    
    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    results = {}
    
    # Test different configurations
    configurations = [
        {
            'name': 'Full Model',
            'use_tfidf': True,
            'use_calibration': True,
            'max_features': 5000,
            'ngram_range': (1, 2)
        },
        {
            'name': 'No Calibration',
            'use_tfidf': True,
            'use_calibration': False,
            'max_features': 5000,
            'ngram_range': (1, 2)
        },
        {
            'name': 'Limited Features (1000)',
            'use_tfidf': True,
            'use_calibration': True,
            'max_features': 1000,
            'ngram_range': (1, 2)
        },
        {
            'name': 'Unigrams Only',
            'use_tfidf': True,
            'use_calibration': True,
            'max_features': 5000,
            'ngram_range': (1, 1)
        },
        {
            'name': 'Trigrams',
            'use_tfidf': True,
            'use_calibration': True,
            'max_features': 5000,
            'ngram_range': (1, 3)
        }
    ]
    
    for config in configurations:
        print(f"\nTesting: {config['name']}")
        
        try:
            # Create model with specific configuration
            model = BaselineIntentClassifier(
                max_features=config['max_features'],
                ngram_range=config['ngram_range']
            )
            
            # Train model
            model.train(train_texts, train_labels)
            
            # Test predictions
            predictions = []
            for text in test_texts:
                pred = model.predict(text, return_confidence=False)
                predictions.append(pred['intent'])
            
            # Calculate metrics
            accuracy = accuracy_score(test_labels, predictions)
            
            # Get confidence scores for calibration analysis
            confidences = []
            for text in test_texts:
                pred = model.predict(text, return_confidence=True)
                confidences.append(pred['confidence'])
            
            # Calculate average confidence
            avg_confidence = np.mean(confidences)
            
            # Calculate confidence variance
            confidence_variance = np.var(confidences)
            
            results[config['name']] = {
                'accuracy': accuracy,
                'avg_confidence': avg_confidence,
                'confidence_variance': confidence_variance,
                'config': config
            }
            
            print(f"  Accuracy: {accuracy:.2%}")
            print(f"  Avg Confidence: {avg_confidence:.3f}")
            print(f"  Confidence Variance: {confidence_variance:.3f}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[config['name']] = {'error': str(e)}
    
    return results

def test_data_size_ablation():
    """Test performance with different data sizes"""
    print("\nRunning Data Size Ablation Study...")
    
    # Load data
    data = load_data()
    texts = [item['text'] for item in data]
    labels = [item['intent'] for item in data]
    
    # Split data
    train_texts, test_texts, train_labels, test_labels = train_test_split(
        texts, labels, test_size=0.2, random_state=42
    )
    
    results = {}
    
    # Test different data sizes
    data_sizes = [0.1, 0.25, 0.5, 0.75, 1.0]
    
    for size in data_sizes:
        print(f"\nTesting with {size*100:.0f}% of training data")
        
        try:
            # Sample data
            n_samples = int(len(train_texts) * size)
            sampled_texts = train_texts[:n_samples]
            sampled_labels = train_labels[:n_samples]
            
            # Train model
            model = BaselineIntentClassifier()
            model.train(sampled_texts, sampled_labels)
            
            # Test predictions
            predictions = []
            for text in test_texts:
                pred = model.predict(text, return_confidence=False)
                predictions.append(pred['intent'])
            
            # Calculate metrics
            accuracy = accuracy_score(test_labels, predictions)
            
            results[f"{size*100:.0f}%_data"] = {
                'accuracy': accuracy,
                'n_samples': n_samples,
                'data_size': size
            }
            
            print(f"  Samples: {n_samples}")
            print(f"  Accuracy: {accuracy:.2%}")
            
        except Exception as e:
            print(f"  Error: {e}")
            results[f"{size*100:.0f}%_data"] = {'error': str(e)}
    
    return results

def generate_ablation_study():
    """Generate complete ablation study"""
    print("Generating Ablation Study...")
    
    # Create output directory
    output_dir = Path("data/ablation_study")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Run studies
    feature_results = test_feature_ablation()
    data_size_results = test_data_size_ablation()
    
    # Combine results
    all_results = {
        'feature_ablation': feature_results,
        'data_size_ablation': data_size_results
    }
    
    # Save results
    with open(output_dir / "ablation_results.json", 'w', encoding='utf-8') as f:
        # Convert numpy types to Python types
        json_results = {}
        for study_name, study_data in all_results.items():
            json_results[study_name] = {}
            for config_name, config_data in study_data.items():
                if 'error' in config_data:
                    json_results[study_name][config_name] = config_data
                else:
                    json_results[study_name][config_name] = {
                        'accuracy': float(config_data['accuracy']),
                        'avg_confidence': float(config_data.get('avg_confidence', 0)),
                        'confidence_variance': float(config_data.get('confidence_variance', 0)),
                        'n_samples': config_data.get('n_samples', 0),
                        'data_size': config_data.get('data_size', 0),
                        'config': config_data.get('config', {})
                    }
        
        json.dump(json_results, f, ensure_ascii=False, indent=2)
    
    print(f"\nAblation study results saved to {output_dir}/ablation_results.json")
    
    # Print summary
    print("\n" + "="*50)
    print("ABLATION STUDY SUMMARY")
    print("="*50)
    
    print("\nFeature Ablation:")
    for config_name, config_data in feature_results.items():
        if 'error' in config_data:
            print(f"  {config_name}: ERROR - {config_data['error']}")
        else:
            print(f"  {config_name}: {config_data['accuracy']:.2%}")
    
    print("\nData Size Ablation:")
    for config_name, config_data in data_size_results.items():
        if 'error' in config_data:
            print(f"  {config_name}: ERROR - {config_data['error']}")
        else:
            print(f"  {config_name}: {config_data['accuracy']:.2%}")
    
    return all_results

def main():
    """Main function"""
    try:
        results = generate_ablation_study()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main()