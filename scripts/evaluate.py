#!/usr/bin/env python3
"""
Evaluation Script for Intent Classification
Tests with real examples and generates comprehensive metrics
"""

import json
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Any, Tuple
from sklearn.metrics import (
    classification_report, confusion_matrix, f1_score, 
    accuracy_score, precision_recall_fscore_support
)
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter
import time
import numpy as np

class IntentEvaluator:
    def __init__(self, model_type: str = "transformer"):
        """Initialize evaluator with model type"""
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
            print(f"Model {self.model_type} loaded successfully")
            return True
        except Exception as e:
            print(f"Error loading model {self.model_type}: {e}")
            return False
    
    def load_test_data(self, data_file: Path) -> Tuple[List[str], List[str]]:
        """Load test data"""
        with open(data_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        texts = [item['text'] for item in data]
        intents = [item['intent'] for item in data]
        
        print(f"Loaded {len(texts)} test examples")
        return texts, intents
    
    def create_real_test_set(self) -> Tuple[List[str], List[str]]:
        """Create real-world test set with 20 examples"""
        real_examples = [
            # Turkish examples
            ("Kütüphane saat kaçta açılıyor?", "opening_hours"),
            ("Pazar günü açık mı?", "opening_hours"),
            ("Kitabı geç getirdim, ne kadar ceza ödeyeceğim?", "fine_policy"),
            ("Ceza nasıl hesaplanıyor?", "fine_policy"),
            ("Kaç kitap alabilirim?", "borrow_limit"),
            ("Öğrenci olarak limitim ne kadar?", "borrow_limit"),
            ("Grup çalışma odası rezerve edebilir miyim?", "room_booking"),
            ("Sessiz çalışma alanı var mı?", "room_booking"),
            ("E-kitap nasıl indirebilirim?", "ebook_access"),
            ("Online veritabanına nasıl erişirim?", "ebook_access"),
            ("WiFi şifresi nedir?", "wifi"),
            ("İnternet bağlantım çalışmıyor", "wifi"),
            ("Kartımı kaybettim, ne yapmalıyım?", "lost_card"),
            ("Yeni kart nasıl alırım?", "lost_card"),
            ("Üyeliğimi nasıl yenilerim?", "renewal"),
            ("Kartımın süresi dolmuş", "renewal"),
            ("Hastayım, kitabı getiremedim", "ill"),
            ("Sağlık raporu gerekli mi?", "ill"),
            ("Bu hafta hangi etkinlikler var?", "events"),
            ("Yazarlık workshopuna katılmak istiyorum", "events"),
            
            # English examples
            ("What are the library hours?", "opening_hours"),
            ("Is it open on Sundays?", "opening_hours"),
            ("I returned the book late, how much fine do I pay?", "fine_policy"),
            ("How is the fine calculated?", "fine_policy"),
            ("How many books can I borrow?", "borrow_limit"),
            ("What's my limit as a student?", "borrow_limit"),
            ("Can I reserve a group study room?", "room_booking"),
            ("Is there a quiet study area?", "room_booking"),
            ("How do I download e-books?", "ebook_access"),
            ("How do I access online databases?", "ebook_access"),
            ("What's the WiFi password?", "wifi"),
            ("My internet connection isn't working", "wifi"),
            ("I lost my card, what should I do?", "lost_card"),
            ("How do I get a new card?", "lost_card"),
            ("How do I renew my membership?", "renewal"),
            ("My card has expired", "renewal"),
            ("I'm sick, I couldn't return the book", "ill"),
            ("Do I need a medical certificate?", "ill"),
            ("What events are there this week?", "events"),
            ("I want to attend the writing workshop", "events"),
            
            # Mixed language examples
            ("Kütüphane ne zaman açık?", "opening_hours"),
            ("What time does the library open?", "opening_hours"),
            ("Kitabı geç getirdim, fine ne kadar?", "fine_policy"),
            ("How much is the late fee?", "fine_policy"),
            ("Kaç kitap borrow edebilirim?", "borrow_limit"),
            ("How many books can I borrow?", "borrow_limit"),
            ("Room booking yapabilir miyim?", "room_booking"),
            ("Can I book a study room?", "room_booking"),
            ("E-book nasıl download ederim?", "ebook_access"),
            ("How do I download e-books?", "ebook_access"),
            ("WiFi password nedir?", "wifi"),
            ("What's the WiFi password?", "wifi"),
            ("Kartımı kaybettim, what should I do?", "lost_card"),
            ("I lost my card, ne yapmalıyım?", "lost_card"),
            ("Membership nasıl renew ederim?", "renewal"),
            ("How do I renew my üyelik?", "renewal"),
            ("Hastayım, couldn't return the book", "ill"),
            ("I'm sick, kitabı getiremedim", "ill"),
            ("Bu hafta what events are there?", "events"),
            ("This week hangi etkinlikler var?", "events"),
            
            # Ambiguous cases
            ("Merhaba", "other"),
            ("Bu çok güzel", "other"),
            ("Random gibberish text", "other"),
            ("Kütüphane", "other"),
            ("Help", "other"),
            ("I don't know what to ask", "other"),
            ("Can you help me with something?", "other"),
            ("What should I do?", "other"),
            ("I need assistance", "other"),
            ("This is a test", "other")
        ]
        
        texts = [example[0] for example in real_examples]
        intents = [example[1] for example in real_examples]
        
        return texts, intents
    
    def evaluate_model(self, texts: List[str], true_intents: List[str]) -> Dict[str, Any]:
        """Evaluate model performance"""
        if not self.is_loaded:
            if not self.load_model():
                raise RuntimeError(f"Model {self.model_type} not available")
        
        print(f"Evaluating {len(texts)} examples...")
        
        # Make predictions
        start_time = time.time()
        predictions = []
        confidences = []
        
        for text in texts:
            prediction = self.model.predict(text, return_confidence=True)
            predictions.append(prediction['intent'])
            confidences.append(prediction['confidence'])
        
        inference_time = time.time() - start_time
        
        # Calculate metrics
        accuracy = accuracy_score(true_intents, predictions)
        f1_weighted = f1_score(true_intents, predictions, average='weighted')
        f1_macro = f1_score(true_intents, predictions, average='macro')
        
        # Per-class metrics
        precision, recall, f1, support = precision_recall_fscore_support(
            true_intents, predictions, average=None
        )
        
        # Create results
        results = {
            'accuracy': accuracy,
            'f1_weighted': f1_weighted,
            'f1_macro': f1_macro,
            'inference_time': inference_time,
            'avg_inference_time': inference_time / len(texts),
            'predictions': predictions,
            'confidences': confidences,
            'true_intents': true_intents,
            'texts': texts
        }
        
        # Per-class results
        unique_intents = sorted(list(set(true_intents)))
        per_class_results = {}
        
        for i, intent in enumerate(unique_intents):
            per_class_results[intent] = {
                'precision': precision[i],
                'recall': recall[i],
                'f1': f1[i],
                'support': support[i]
            }
        
        results['per_class'] = per_class_results
        
        return results
    
    def analyze_errors(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze prediction errors"""
        predictions = results['predictions']
        true_intents = results['true_intents']
        texts = results['texts']
        confidences = results['confidences']
        
        # Find errors
        errors = []
        for i, (pred, true, text, conf) in enumerate(zip(predictions, true_intents, texts, confidences)):
            if pred != true:
                errors.append({
                    'text': text,
                    'predicted': pred,
                    'true': true,
                    'confidence': conf,
                    'text_length': len(text.split()),
                    'language': self._detect_language(text)
                })
        
        # Error analysis
        error_analysis = {
            'total_errors': len(errors),
            'error_rate': len(errors) / len(predictions),
            'avg_confidence_errors': np.mean([e['confidence'] for e in errors]),
            'avg_confidence_correct': np.mean([conf for i, conf in enumerate(confidences) if predictions[i] == true_intents[i]]),
            'errors_by_language': Counter([e['language'] for e in errors]),
            'errors_by_length': self._analyze_errors_by_length(errors),
            'confusion_pairs': self._analyze_confusion_pairs(errors)
        }
        
        return error_analysis
    
    def _detect_language(self, text: str) -> str:
        """Simple language detection"""
        turkish_chars = set('çğıöşüÇĞIİÖŞÜ')
        english_chars = set('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')
        
        text_chars = set(text.lower())
        
        turkish_ratio = len(text_chars.intersection(turkish_chars)) / len(text_chars) if text_chars else 0
        english_ratio = len(text_chars.intersection(english_chars)) / len(text_chars) if text_chars else 0
        
        if turkish_ratio > english_ratio:
            return "tr"
        elif english_ratio > turkish_ratio:
            return "en"
        else:
            return "mixed"
    
    def _analyze_errors_by_length(self, errors: List[Dict]) -> Dict[str, int]:
        """Analyze errors by text length"""
        length_ranges = {
            '1-5 words': 0,
            '6-10 words': 0,
            '11-15 words': 0,
            '16+ words': 0
        }
        
        for error in errors:
            length = error['text_length']
            if length <= 5:
                length_ranges['1-5 words'] += 1
            elif length <= 10:
                length_ranges['6-10 words'] += 1
            elif length <= 15:
                length_ranges['11-15 words'] += 1
            else:
                length_ranges['16+ words'] += 1
        
        return length_ranges
    
    def _analyze_confusion_pairs(self, errors: List[Dict]) -> List[Tuple[str, str, int]]:
        """Analyze most common confusion pairs"""
        confusion_pairs = Counter()
        for error in errors:
            pair = (error['true'], error['predicted'])
            confusion_pairs[pair] += 1
        
        return confusion_pairs.most_common(10)
    
    def generate_report(self, results: Dict[str, Any], error_analysis: Dict[str, Any]) -> str:
        """Generate evaluation report"""
        report = []
        report.append("# Intent Classification Evaluation Report")
        report.append("")
        
        # Overall metrics
        report.append("## Overall Performance")
        report.append(f"- **Accuracy:** {results['accuracy']:.4f}")
        report.append(f"- **F1-Score (Weighted):** {results['f1_weighted']:.4f}")
        report.append(f"- **F1-Score (Macro):** {results['f1_macro']:.4f}")
        report.append(f"- **Inference Time:** {results['avg_inference_time']:.2f}ms per prediction")
        report.append("")
        
        # Per-class performance
        report.append("## Per-Class Performance")
        report.append("| Intent | Precision | Recall | F1-Score | Support |")
        report.append("|--------|-----------|--------|----------|---------|")
        
        for intent, metrics in results['per_class'].items():
            report.append(f"| {intent} | {metrics['precision']:.3f} | {metrics['recall']:.3f} | {metrics['f1']:.3f} | {metrics['support']} |")
        
        report.append("")
        
        # Error analysis
        report.append("## Error Analysis")
        report.append(f"- **Total Errors:** {error_analysis['total_errors']}")
        report.append(f"- **Error Rate:** {error_analysis['error_rate']:.2%}")
        report.append(f"- **Avg Confidence (Errors):** {error_analysis['avg_confidence_errors']:.3f}")
        report.append(f"- **Avg Confidence (Correct):** {error_analysis['avg_confidence_correct']:.3f}")
        report.append("")
        
        # Errors by language
        report.append("### Errors by Language")
        for lang, count in error_analysis['errors_by_language'].items():
            report.append(f"- **{lang}:** {count}")
        report.append("")
        
        # Errors by length
        report.append("### Errors by Text Length")
        for length_range, count in error_analysis['errors_by_length'].items():
            report.append(f"- **{length_range}:** {count}")
        report.append("")
        
        # Confusion pairs
        report.append("### Most Common Confusion Pairs")
        for (true, pred), count in error_analysis['confusion_pairs']:
            report.append(f"- **{true} → {pred}:** {count}")
        report.append("")
        
        return "\n".join(report)
    
    def save_results(self, results: Dict[str, Any], error_analysis: Dict[str, Any], output_dir: Path):
        """Save evaluation results"""
        output_dir.mkdir(parents=True, exist_ok=True)

        def _sanitize(obj):
            """Recursively convert NumPy types to native Python types for JSON serialization."""
            if isinstance(obj, dict):
                return {k: _sanitize(v) for k, v in obj.items()}
            if isinstance(obj, list):
                return [_sanitize(v) for v in obj]
            if isinstance(obj, tuple):
                return tuple(_sanitize(v) for v in obj)
            if isinstance(obj, (np.integer,)):
                return int(obj)
            if isinstance(obj, (np.floating,)):
                return float(obj)
            if isinstance(obj, (np.ndarray,)):
                return _sanitize(obj.tolist())
            return obj
        
        # Save detailed results
        results_file = output_dir / f"evaluation_results_{self.model_type}.json"
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump({
                'results': _sanitize(results),
                'error_analysis': _sanitize(error_analysis)
            }, f, ensure_ascii=False, indent=2)
        
        # Save report
        report = self.generate_report(results, error_analysis)
        report_file = output_dir / f"evaluation_report_{self.model_type}.md"
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"Results saved to {output_dir}")

def main():
    """Main function"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Evaluate Intent Classification Model")
    parser.add_argument("--model", type=str, default="transformer", choices=["baseline", "transformer"])
    parser.add_argument("--data", type=str, help="Path to test data file")
    parser.add_argument("--output", type=str, default="reports", help="Output directory")
    
    args = parser.parse_args()
    
    # Initialize evaluator
    evaluator = IntentEvaluator(model_type=args.model)
    
    # Load test data
    if args.data:
        texts, intents = evaluator.load_test_data(Path(args.data))
    else:
        # Use real test set
        texts, intents = evaluator.create_real_test_set()
        print(f"Using real test set with {len(texts)} examples")
    
    # Evaluate model
    results = evaluator.evaluate_model(texts, intents)
    
    # Analyze errors
    error_analysis = evaluator.analyze_errors(results)
    
    # Save results
    output_dir = Path(args.output)
    evaluator.save_results(results, error_analysis, output_dir)
    
    # Print summary
    print(f"\nEvaluation Summary for {args.model} model:")
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"F1-Score: {results['f1_weighted']:.4f}")
    print(f"Error Rate: {error_analysis['error_rate']:.2%}")
    print(f"Avg Inference Time: {results['avg_inference_time']:.2f}ms")

if __name__ == "__main__":
    main()
