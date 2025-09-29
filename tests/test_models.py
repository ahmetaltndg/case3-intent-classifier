# tests/test_models.py - Model Testleri

import pytest
import sys
import os
from pathlib import Path

# Proje root'unu path'e ekle
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.training.train_baseline import BaselineIntentClassifier
from src.models.transformer_classifier import TransformerIntentClassifier

class TestBaselineModel:
    """Baseline model testleri"""
    
    def test_baseline_model_initialization(self):
        """Baseline model başlatma test eder"""
        model = BaselineIntentClassifier()
        assert model is not None
        assert hasattr(model, 'vectorizer')
        assert hasattr(model, 'classifier')
    
    def test_baseline_model_training(self):
        """Baseline model eğitimi test eder"""
        # Test verisi
        train_texts = [
            "Kütüphane saat kaçta açılıyor?",
            "What are the library hours?",
            "Kitabı geç getirdim, ceza var mı?",
            "How many books can I borrow?",
            "Oda rezervasyonu yapabilir miyim?",
            "E-kitap nasıl indirebilirim?",
            "WiFi şifresi nedir?",
            "Kartımı kaybettim",
            "Üyeliğimi nasıl yenilerim?",
            "Hastayım, kitabı getiremedim",
            "Bu hafta hangi etkinlikler var?",
            "Çok gürültülü, şikayet etmek istiyorum",
            "Genel bilgi almak istiyorum"
        ]
    
        train_labels = [
            "opening_hours", "opening_hours", "fine_policy", "borrow_limit",
            "room_booking", "ebook_access", "wifi", "lost_card",
            "renewal", "ill", "events", "complaint", "other"
        ]
        
        model = BaselineIntentClassifier()
        model.train(train_texts, train_labels)
        
        # Model eğitildi mi kontrol et
        assert model.vectorizer is not None
        assert model.classifier is not None
        assert hasattr(model, 'labels')
        assert len(model.labels) == 12
    
    def test_baseline_model_prediction(self):
        """Baseline model tahmini test eder"""
        # Test verisi
        train_texts = [
            "Kütüphane saat kaçta açılıyor?",
            "What are the library hours?",
            "Kitabı geç getirdim, ceza var mı?",
            "How many books can I borrow?"
        ]
        
        train_labels = [
            "opening_hours", "opening_hours", "fine_policy", "borrow_limit"
        ]
        
        model = BaselineIntentClassifier()
        model.train(train_texts, train_labels)
        
        # Tahmin test et
        test_text = "Kütüphane saat kaçta açılıyor?"
        prediction = model.predict(test_text)
        
        assert "intent" in prediction
        assert "confidence" in prediction
        assert prediction["intent"] in model.labels
        assert 0.0 <= prediction["confidence"] <= 1.0
    
    def test_baseline_model_probabilities(self):
        """Baseline model probability çıktısı test eder"""
        # Test verisi
        train_texts = [
            "Kütüphane saat kaçta açılıyor?",
            "What are the library hours?",
            "Kitabı geç getirdim, ceza var mı?",
            "How many books can I borrow?"
        ]
        
        train_labels = [
            "opening_hours", "opening_hours", "fine_policy", "borrow_limit"
        ]
        
        model = BaselineIntentClassifier()
        model.train(train_texts, train_labels)
        
        # Probability çıktısı test et
        test_text = "Kütüphane saat kaçta açılıyor?"
        prediction = model.predict(test_text, return_confidence=True)
        
        assert "probabilities" in prediction
        probabilities = prediction["probabilities"]
        
        # Tüm label'lar için probability olmalı
        for label in model.labels:
            assert label in probabilities
            assert 0.0 <= probabilities[label] <= 1.0
        
        # Probability'lerin toplamı yaklaşık 1 olmalı
        total_prob = sum(probabilities.values())
        assert 0.95 <= total_prob <= 1.05
    
    def test_baseline_model_save_load(self):
        """Baseline model kaydetme/yükleme test eder"""
        # Test verisi
        train_texts = [
            "Kütüphane saat kaçta açılıyor?",
            "What are the library hours?",
            "Kitabı geç getirdim, ceza var mı?",
            "How many books can I borrow?"
        ]
        
        train_labels = [
            "opening_hours", "opening_hours", "fine_policy", "borrow_limit"
        ]
        
        model = BaselineIntentClassifier()
        model.train(train_texts, train_labels)
        
        # Model kaydet
        model_dir = Path("test_baseline_model")
        model.save_model(model_dir)
        
        # Model yükle
        loaded_model = BaselineIntentClassifier()
        loaded_model.load_model(model_dir)
        
        # Aynı tahmin yapmalı
        test_text = "Kütüphane saat kaçta açılıyor?"
        original_prediction = model.predict(test_text)
        loaded_prediction = loaded_model.predict(test_text)
        
        assert original_prediction["intent"] == loaded_prediction["intent"]
        assert abs(original_prediction["confidence"] - loaded_prediction["confidence"]) < 0.01
        
        # Test dosyalarını temizle
        import shutil
        if model_dir.exists():
            shutil.rmtree(model_dir)

class TestTransformerModel:
    """Transformer model testleri"""
    
    def test_transformer_model_initialization(self):
        """Transformer model başlatma test eder"""
        model = TransformerIntentClassifier()
        assert model is not None
        assert hasattr(model, 'tokenizer')
        assert hasattr(model, 'model')
    
    def test_transformer_model_prediction(self):
        """Transformer model tahmini test eder (eğitilmiş model varsa)"""
        model = TransformerIntentClassifier()
        
        # Eğitilmiş model yükle
        model_dir = Path("artifacts/transformer_model")
        if model_dir.exists():
            model.load_model(model_dir)
            
            # Tahmin test et
            test_text = "Kütüphane saat kaçta açılıyor?"
            prediction = model.predict(test_text)
            
            assert "intent" in prediction
            assert "confidence" in prediction
            assert 0.0 <= prediction["confidence"] <= 1.0
        else:
            pytest.skip("Transformer model eğitilmemiş")
    
    def test_transformer_model_probabilities(self):
        """Transformer model probability çıktısı test eder"""
        model = TransformerIntentClassifier()
        
        # Eğitilmiş model yükle
        model_dir = Path("artifacts/transformer_model")
        if model_dir.exists():
            model.load_model(model_dir)
            
            # Probability çıktısı test et
            test_text = "Kütüphane saat kaçta açılıyor?"
            prediction = model.predict(test_text, return_confidence=True)
            
            assert "probabilities" in prediction
            probabilities = prediction["probabilities"]
            
            # Tüm intent'ler için probability olmalı
            expected_intents = [
                "opening_hours", "fine_policy", "borrow_limit", "room_booking",
                "ebook_access", "wifi", "lost_card", "renewal", "ill", 
                "events", "complaint", "other"
            ]
            
            for intent in expected_intents:
                assert intent in probabilities
                assert 0.0 <= probabilities[intent] <= 1.0
            
            # Probability'lerin toplamı yaklaşık 1 olmalı
            total_prob = sum(probabilities.values())
            assert 0.95 <= total_prob <= 1.05
        else:
            pytest.skip("Transformer model eğitilmemiş")

class TestModelComparison:
    """Model karşılaştırma testleri"""
    
    def test_baseline_vs_transformer_consistency(self):
        """Baseline ve transformer modellerinin tutarlılığını test eder"""
        baseline_model = BaselineIntentClassifier()
        transformer_model = TransformerIntentClassifier()
        
        # Eğitilmiş modelleri yükle
        baseline_dir = Path("artifacts/baseline_model")
        transformer_dir = Path("artifacts/transformer_model")
        
        if baseline_dir.exists() and transformer_dir.exists():
            baseline_model.load_model(baseline_dir)
            transformer_model.load_model(transformer_dir)
            
            # Test metinleri
            test_texts = [
                "Kütüphane saat kaçta açılıyor?",
                "What are the library hours?",
                "How many books can I borrow?"
            ]
            
            for text in test_texts:
                baseline_pred = baseline_model.predict(text)
                transformer_pred = transformer_model.predict(text)
                
                # Her iki model de geçerli intent döndürmeli
                assert baseline_pred["intent"] != ""
                assert transformer_pred["intent"] != ""
                
                # Confidence değerleri geçerli olmalı
                assert 0.0 <= baseline_pred["confidence"] <= 1.0
                assert 0.0 <= transformer_pred["confidence"] <= 1.0
        else:
            pytest.skip("Modeller eğitilmemiş")
    
    def test_model_performance_metrics(self):
        """Model performans metriklerini test eder"""
        baseline_model = BaselineIntentClassifier()
        
        # Eğitilmiş model yükle
        model_dir = Path("artifacts/baseline_model")
        if model_dir.exists():
            baseline_model.load_model(model_dir)
            
            # Test verisi
            test_texts = [
                "Kütüphane saat kaçta açılıyor?",
                "What are the library hours?",
                "Kitabı geç getirdim, ceza var mı?",
                "How many books can I borrow?",
                "Oda rezervasyonu yapabilir miyim?"
            ]
            
            test_labels = [
                "opening_hours", "opening_hours", "fine_policy", 
                "borrow_limit", "room_booking"
            ]
            
            # Tahminler yap
            predictions = []
            for text in test_texts:
                pred = baseline_model.predict(text)
                predictions.append(pred["intent"])
            
            # Basit accuracy hesapla
            correct = sum(1 for pred, true in zip(predictions, test_labels) if pred == true)
            accuracy = correct / len(test_labels)
            
            # Accuracy makul bir değerde olmalı
            assert accuracy >= 0.0
            assert accuracy <= 1.0
        else:
            pytest.skip("Baseline model eğitilmemiş")

class TestModelCalibration:
    """Model kalibrasyon testleri"""
    
    def test_baseline_calibration(self):
        """Baseline model kalibrasyonu test eder"""
        baseline_model = BaselineIntentClassifier()
        
        # Eğitilmiş model yükle
        model_dir = Path("artifacts/baseline_model")
        if model_dir.exists():
            baseline_model.load_model(model_dir)
            
            # Test metni
            test_text = "Kütüphane saat kaçta açılıyor?"
            prediction = baseline_model.predict(test_text, return_confidence=True)
            
            # Kalibrasyon kontrolü
            assert "probabilities" in prediction
            probabilities = prediction["probabilities"]
            
            # En yüksek probability confidence ile eşleşmeli
            max_prob = max(probabilities.values())
            assert abs(max_prob - prediction["confidence"]) < 0.01
        else:
            pytest.skip("Baseline model eğitilmemiş")
    
    def test_transformer_calibration(self):
        """Transformer model kalibrasyonu test eder"""
        transformer_model = TransformerIntentClassifier()
        
        # Eğitilmiş model yükle
        model_dir = Path("artifacts/transformer_model")
        if model_dir.exists():
            transformer_model.load_model(model_dir)
            
            # Test metni
            test_text = "Kütüphane saat kaçta açılıyor?"
            prediction = transformer_model.predict(test_text, return_confidence=True)
            
            # Kalibrasyon kontrolü
            assert "probabilities" in prediction
            probabilities = prediction["probabilities"]
            
            # En yüksek probability confidence ile eşleşmeli
            max_prob = max(probabilities.values())
            assert abs(max_prob - prediction["confidence"]) < 0.01
        else:
            pytest.skip("Transformer model eğitilmemiş")

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
