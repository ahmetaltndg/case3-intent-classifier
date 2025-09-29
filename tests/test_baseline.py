#!/usr/bin/env python3
"""
Tests for baseline intent classifier
"""

import pytest
import json
import tempfile
from pathlib import Path
from src.training.train_baseline import BaselineIntentClassifier

class TestBaselineClassifier:
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
    
    def test_classifier_initialization(self):
        """Test classifier initialization"""
        classifier = BaselineIntentClassifier()
        assert classifier.vectorizer is not None
        assert classifier.classifier is not None
        assert not classifier.is_trained
    
    def test_data_preparation(self):
        """Test data preparation"""
        classifier = BaselineIntentClassifier()
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        processed_texts, numeric_intents = classifier.prepare_data(texts, intents)
        
        assert len(processed_texts) == len(texts)
        assert len(numeric_intents) == len(intents)
        assert len(classifier.intent_labels) == len(set(intents))
    
    def test_training(self):
        """Test model training"""
        classifier = BaselineIntentClassifier()
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        # Train model
        results = classifier.train(texts, intents, test_size=0.3)
        
        assert classifier.is_trained
        assert 'f1_score' in results
        assert results['f1_score'] >= 0.0
    
    def test_prediction(self):
        """Test single prediction"""
        classifier = BaselineIntentClassifier()
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        # Train model
        classifier.train(texts, intents, test_size=0.3)
        
        # Test prediction
        test_text = "Kütüphane saat kaçta açılıyor?"
        prediction = classifier.predict(test_text)
        
        assert 'intent' in prediction
        assert 'confidence' in prediction
        assert prediction['intent'] in classifier.intent_labels
        assert 0.0 <= prediction['confidence'] <= 1.0
    
    def test_batch_prediction(self):
        """Test batch prediction"""
        classifier = BaselineIntentClassifier()
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        # Train model
        classifier.train(texts, intents, test_size=0.3)
        
        # Test batch prediction
        test_texts = ["Kütüphane saat kaçta açılıyor?", "What are the library hours?"]
        predictions = classifier.predict_batch(test_texts)
        
        assert len(predictions) == len(test_texts)
        for prediction in predictions:
            assert 'intent' in prediction
            assert 'confidence' in prediction
    
    def test_model_save_load(self):
        """Test model saving and loading"""
        classifier = BaselineIntentClassifier()
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        # Train model
        classifier.train(texts, intents, test_size=0.3)
        
        # Save model
        with tempfile.TemporaryDirectory() as temp_dir:
            model_dir = Path(temp_dir) / "test_model"
            classifier.save_model(model_dir)
            
            # Check files exist
            assert (model_dir / "vectorizer.pkl").exists()
            assert (model_dir / "classifier.pkl").exists()
            assert (model_dir / "labels.json").exists()
            
            # Load model
            new_classifier = BaselineIntentClassifier()
            new_classifier.load_model(model_dir)
            
            assert new_classifier.is_trained
            assert new_classifier.intent_labels == classifier.intent_labels
            
            # Test prediction
            test_text = "Kütüphane saat kaçta açılıyor?"
            prediction1 = classifier.predict(test_text)
            prediction2 = new_classifier.predict(test_text)
            
            assert prediction1['intent'] == prediction2['intent']
            assert abs(prediction1['confidence'] - prediction2['confidence']) < 0.01
    
    def test_preprocessing(self):
        """Test text preprocessing"""
        classifier = BaselineIntentClassifier()
        
        test_cases = [
            ("Kütüphane saat kaçta açılıyor?", "kütüphane saat kaçta açılıyor ?"),
            ("What are the library hours?", "what are the library hours ?"),
            ("  Extra   spaces  ", "extra spaces"),
            ("Mixed case TEXT", "mixed case text")
        ]
        
        for input_text, expected in test_cases:
            result = classifier.preprocess_text(input_text)
            assert result == expected
    
    def test_intent_labels(self):
        """Test intent label mapping"""
        classifier = BaselineIntentClassifier()
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        classifier.train(texts, intents, test_size=0.3)
        
        # Check all intents are mapped
        unique_intents = set(intents)
        assert len(classifier.intent_labels) == len(unique_intents)
        assert set(classifier.intent_labels.keys()) == unique_intents
        
        # Check bidirectional mapping
        for intent, label in classifier.intent_labels.items():
            assert classifier.label_to_intent[label] == intent
    
    def test_confidence_scores(self):
        """Test confidence score validity"""
        classifier = BaselineIntentClassifier()
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        classifier.train(texts, intents, test_size=0.3)
        
        # Test multiple predictions
        test_texts = [
            "Kütüphane saat kaçta açılıyor?",
            "What are the library hours?",
            "Random text that doesn't match any intent"
        ]
        
        for text in test_texts:
            prediction = classifier.predict(text, return_confidence=True)
            assert 0.0 <= prediction['confidence'] <= 1.0
            
            if 'probabilities' in prediction:
                # Check probabilities sum to 1
                prob_sum = sum(prediction['probabilities'].values())
                assert abs(prob_sum - 1.0) < 0.01
                
                # Check all probabilities are valid
                for prob in prediction['probabilities'].values():
                    assert 0.0 <= prob <= 1.0

if __name__ == "__main__":
    pytest.main([__file__])
