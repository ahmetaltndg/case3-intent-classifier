#!/usr/bin/env python3
"""
Tests for rejection option (abstain) functionality
"""

import pytest
import json
import numpy as np
from pathlib import Path
from src.training.train_baseline import BaselineIntentClassifier

class TestRejectionOption:
    def setup_method(self):
        """Setup test data"""
        self.test_data = [
            # Her intent için 5 örnek ekleyelim
            {"text": "Kütüphane saat kaçta açılıyor?", "intent": "opening_hours"},
            {"text": "What are the library hours?", "intent": "opening_hours"},
            {"text": "Pazar günü açık mı?", "intent": "opening_hours"},
            {"text": "Akşam kaçta kapanıyor?", "intent": "opening_hours"},
            {"text": "Hafta sonu çalışma saatleri nedir?", "intent": "opening_hours"},
            
            {"text": "Kitabı geç getirdim, ceza var mı?", "intent": "fine_policy"},
            {"text": "How is the late fee calculated?", "intent": "fine_policy"},
            {"text": "Ceza nasıl hesaplanıyor?", "intent": "fine_policy"},
            {"text": "Gecikme ücreti nedir?", "intent": "fine_policy"},
            {"text": "Ne kadar ceza var?", "intent": "fine_policy"},
            
            {"text": "How many books can I borrow?", "intent": "borrow_limit"},
            {"text": "Kaç kitap alabilirim?", "intent": "borrow_limit"},
            {"text": "Öğrenci limiti nedir?", "intent": "borrow_limit"},
            {"text": "Maksimum kaç kitap?", "intent": "borrow_limit"},
            {"text": "Ödünç alma limiti nedir?", "intent": "borrow_limit"},
            
            {"text": "Oda rezervasyonu yapabilir miyim?", "intent": "room_booking"},
            {"text": "Can I reserve a study room?", "intent": "room_booking"},
            {"text": "Sessiz çalışma alanı var mı?", "intent": "room_booking"},
            {"text": "Grup çalışma odası rezerve edebilir miyim?", "intent": "room_booking"},
            {"text": "Çalışma odası rezerve etmek istiyorum", "intent": "room_booking"},
            
            {"text": "E-kitap nasıl indirebilirim?", "intent": "ebook_access"},
            {"text": "How do I access online databases?", "intent": "ebook_access"},
            {"text": "E-kaynaklara erişim nasıl?", "intent": "ebook_access"},
            {"text": "Dijital kaynaklar nerede?", "intent": "ebook_access"},
            {"text": "E-kitap erişimi nasıl?", "intent": "ebook_access"},
            
            {"text": "WiFi şifresi nedir?", "intent": "wifi"},
            {"text": "My internet connection isn't working", "intent": "wifi"},
            {"text": "WiFi çekmiyor", "intent": "wifi"},
            {"text": "İnternet sorunu var", "intent": "wifi"},
            {"text": "Bağlantı problemi", "intent": "wifi"},
            
            {"text": "Kartımı kaybettim", "intent": "lost_card"},
            {"text": "I lost my library card, what should I do?", "intent": "lost_card"},
            {"text": "Yeni kart nasıl alırım?", "intent": "lost_card"},
            {"text": "Kart çıkarmak istiyorum", "intent": "lost_card"},
            {"text": "Kayıp kart raporu", "intent": "lost_card"},
            
            {"text": "Üyeliğimi nasıl yenilerim?", "intent": "renewal"},
            {"text": "How do I renew my membership?", "intent": "renewal"},
            {"text": "Kartımın süresi doldu", "intent": "renewal"},
            {"text": "Yenileme nasıl yapılır?", "intent": "renewal"},
            {"text": "Üyelik yenileme", "intent": "renewal"},
            
            {"text": "Hastayım, kitabı getiremedim", "intent": "ill"},
            {"text": "I'm sick and couldn't return the book", "intent": "ill"},
            {"text": "Sağlık raporu gerekli mi?", "intent": "ill"},
            {"text": "Hasta olduğum için geç kaldım", "intent": "ill"},
            {"text": "Sağlık sorunu", "intent": "ill"},
            
            {"text": "Bu hafta hangi etkinlikler var?", "intent": "events"},
            {"text": "What events are there this week?", "intent": "events"},
            {"text": "Yazarlık workshopuna katılmak istiyorum", "intent": "events"},
            {"text": "Etkinlik takvimi", "intent": "events"},
            {"text": "Seminer programı", "intent": "events"},
            
            {"text": "Çok gürültülü, şikayet etmek istiyorum", "intent": "complaint"},
            {"text": "I want to file a complaint about the noise", "intent": "complaint"},
            {"text": "Hizmet kalitesinden memnun değilim", "intent": "complaint"},
            {"text": "Şikayet formu", "intent": "complaint"},
            {"text": "Memnuniyetsizlik", "intent": "complaint"},
            
            {"text": "Genel bilgi almak istiyorum", "intent": "other"},
            {"text": "I need some general information", "intent": "other"},
            {"text": "Bana yardımcı olur musunuz?", "intent": "other"},
            {"text": "Bilgi almak istiyorum", "intent": "other"},
            {"text": "Yardım istiyorum", "intent": "other"}
        ]
        
        # Ambiguous test cases
        self.ambiguous_cases = [
            "Merhaba",  # Too short
            "Bu çok güzel",  # No clear intent
            "Random gibberish text",  # Nonsensical
            "Kütüphane",  # Too vague
            "Help",  # Too generic
            "I don't know what to ask",  # Meta question
            "Can you help me with something?",  # Too vague
            "What should I do?",  # No specific intent
            "I need assistance",  # Generic request
            "This is a test"  # Test text
        ]
    
    def test_confidence_threshold_rejection(self):
        """Test rejection based on confidence threshold"""
        classifier = BaselineIntentClassifier()
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        # Train model
        classifier.train(texts, intents, test_size=0.3)
        
        # Test with different confidence thresholds
        thresholds = [0.1, 0.3, 0.5, 0.7]
        
        # Test with clear intent
        clear_text = "Kütüphane saat kaçta açılıyor?"
        prediction = classifier.predict(clear_text)
        print(f"Clear text confidence: {prediction['confidence']}")
        
        # Test with ambiguous text
        ambiguous_text = "Merhaba"
        ambiguous_prediction = classifier.predict(ambiguous_text)
        print(f"Ambiguous text confidence: {ambiguous_prediction['confidence']}")
        
        # Ambiguous text should have lower confidence
        assert ambiguous_prediction['confidence'] < prediction['confidence']
        
        # Test that confidence values are reasonable
        assert 0.0 <= prediction['confidence'] <= 1.0
        assert 0.0 <= ambiguous_prediction['confidence'] <= 1.0
        
        # Test that both predictions have valid intents
        assert prediction['intent'] in classifier.intent_labels
        assert ambiguous_prediction['intent'] in classifier.intent_labels
    
    def test_ambiguous_text_rejection(self):
        """Test rejection for ambiguous texts"""
        classifier = BaselineIntentClassifier()
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        # Train model
        classifier.train(texts, intents, test_size=0.3)
        
        # Test ambiguous cases
        for ambiguous_text in self.ambiguous_cases:
            prediction = classifier.predict(ambiguous_text)
            
            # Should have lower confidence
            assert prediction['confidence'] < 0.8  # Adjust threshold as needed
            
            # Check that probabilities are more evenly distributed
            if 'probabilities' in prediction:
                probs = list(prediction['probabilities'].values())
                max_prob = max(probs)
                second_max = sorted(probs, reverse=True)[1]
                
                # For ambiguous cases, max and second max should be closer
                assert max_prob - second_max < 0.3
    
    def test_rejection_rate_analysis(self):
        """Test rejection rate for different confidence thresholds"""
        classifier = BaselineIntentClassifier()
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        # Train model
        classifier.train(texts, intents, test_size=0.3)
        
        # Test with different thresholds
        thresholds = [0.1, 0.3, 0.5, 0.7, 0.9]
        rejection_rates = []
        
        for threshold in thresholds:
            rejected_count = 0
            total_count = len(self.ambiguous_cases)
            
            for text in self.ambiguous_cases:
                prediction = classifier.predict(text)
                if prediction['confidence'] < threshold:
                    rejected_count += 1
            
            rejection_rate = rejected_count / total_count
            rejection_rates.append(rejection_rate)
            
            # Higher thresholds should lead to higher rejection rates
            assert rejection_rate >= 0.0
            assert rejection_rate <= 1.0
        
        # Check that rejection rates increase with threshold
        for i in range(1, len(rejection_rates)):
            assert rejection_rates[i] >= rejection_rates[i-1]
    
    def test_confidence_distribution(self):
        """Test confidence score distribution"""
        classifier = BaselineIntentClassifier()
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        # Train model
        classifier.train(texts, intents, test_size=0.3)
        
        # Test confidence scores for different types of text
        clear_texts = ["Kütüphane saat kaçta açılıyor?", "What are the library hours?"]
        ambiguous_texts = ["Merhaba", "Bu çok güzel"]
        
        clear_confidences = []
        ambiguous_confidences = []
        
        for text in clear_texts:
            prediction = classifier.predict(text)
            clear_confidences.append(prediction['confidence'])
        
        for text in ambiguous_texts:
            prediction = classifier.predict(text)
            ambiguous_confidences.append(prediction['confidence'])
        
        # Clear texts should have higher confidence
        avg_clear_confidence = np.mean(clear_confidences)
        avg_ambiguous_confidence = np.mean(ambiguous_confidences)
        
        assert avg_clear_confidence > avg_ambiguous_confidence
    
    def test_entropy_based_rejection(self):
        """Test entropy-based rejection"""
        classifier = BaselineIntentClassifier()
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        # Train model
        classifier.train(texts, intents, test_size=0.3)
        
        # Calculate entropy for different texts
        def calculate_entropy(probabilities):
            probs = list(probabilities.values())
            entropy = -sum(p * np.log2(p + 1e-10) for p in probs)
            return entropy
        
        clear_text = "Kütüphane saat kaçta açılıyor?"
        ambiguous_text = "Merhaba"
        
        clear_prediction = classifier.predict(clear_text, return_confidence=True)
        ambiguous_prediction = classifier.predict(ambiguous_text, return_confidence=True)
        
        if 'probabilities' in clear_prediction and 'probabilities' in ambiguous_prediction:
            clear_entropy = calculate_entropy(clear_prediction['probabilities'])
            ambiguous_entropy = calculate_entropy(ambiguous_prediction['probabilities'])
            
            # Ambiguous text should have higher entropy (more uncertainty)
            assert ambiguous_entropy > clear_entropy
    
    def test_rejection_consistency(self):
        """Test that rejection is consistent across multiple runs"""
        classifier = BaselineIntentClassifier()
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        # Train model
        classifier.train(texts, intents, test_size=0.3)
        
        # Test same text multiple times
        test_text = "Merhaba"
        predictions = []
        
        for _ in range(5):
            prediction = classifier.predict(test_text)
            predictions.append(prediction)
        
        # All predictions should be consistent
        for i in range(1, len(predictions)):
            assert predictions[i]['intent'] == predictions[0]['intent']
            assert abs(predictions[i]['confidence'] - predictions[0]['confidence']) < 0.01
    
    def test_rejection_with_different_models(self):
        """Test rejection behavior with different model types"""
        # This test would require both baseline and transformer models
        # For now, just test baseline
        classifier = BaselineIntentClassifier()
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        # Train model
        classifier.train(texts, intents, test_size=0.3)
        
        # Test rejection with different confidence thresholds
        thresholds = [0.3, 0.5, 0.7]
        
        for threshold in thresholds:
            rejected_count = 0
            total_count = len(self.ambiguous_cases)
            
            for text in self.ambiguous_cases:
                prediction = classifier.predict(text)
                if prediction['confidence'] < threshold:
                    rejected_count += 1
            
            # Should have some rejections for ambiguous cases
            assert rejected_count > 0
    
    def test_rejection_metrics(self):
        """Test rejection-related metrics"""
        classifier = BaselineIntentClassifier()
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        # Train model
        classifier.train(texts, intents, test_size=0.3)
        
        # Test metrics calculation
        threshold = 0.5
        true_positives = 0
        false_positives = 0
        true_negatives = 0
        false_negatives = 0
        
        # Test with clear cases (should not be rejected)
        for text in ["Kütüphane saat kaçta açılıyor?", "What are the library hours?"]:
            prediction = classifier.predict(text)
            if prediction['confidence'] >= threshold:
                true_positives += 1
            else:
                false_negatives += 1
        
        # Test with ambiguous cases (should be rejected)
        for text in self.ambiguous_cases[:5]:  # Test first 5 ambiguous cases
            prediction = classifier.predict(text)
            if prediction['confidence'] < threshold:
                true_negatives += 1
            else:
                false_positives += 1
        
        # Calculate metrics
        precision = true_positives / (true_positives + false_positives) if (true_positives + false_positives) > 0 else 0
        recall = true_positives / (true_positives + false_negatives) if (true_positives + false_negatives) > 0 else 0
        
        # Metrics should be reasonable
        assert 0.0 <= precision <= 1.0
        assert 0.0 <= recall <= 1.0
    
    def test_rejection_edge_cases(self):
        """Test edge cases for rejection"""
        classifier = BaselineIntentClassifier()
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        # Train model
        classifier.train(texts, intents, test_size=0.3)
        
        # Test edge cases
        edge_cases = [
            "",  # Empty string
            "a",  # Single character
            " " * 100,  # Only spaces
            "123456789",  # Only numbers
            "!@#$%^&*()",  # Only special characters
        ]
        
        for text in edge_cases:
            try:
                prediction = classifier.predict(text)
                # Should handle edge cases gracefully
                assert 'intent' in prediction
                assert 'confidence' in prediction
                assert 0.0 <= prediction['confidence'] <= 1.0
            except Exception as e:
                # Should not crash, but may return abstain
                assert "abstain" in str(e).lower() or "error" in str(e).lower()

if __name__ == "__main__":
    pytest.main([__file__])
