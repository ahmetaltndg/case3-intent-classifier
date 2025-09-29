# tests/test_api.py - FastAPI Servis Testleri

import pytest
import requests
import json
from typing import Dict, Any

# Test konfigürasyonu
BASE_URL = "http://localhost:8000"
TEST_TIMEOUT = 30

class TestFastAPIService:
    """FastAPI servis endpoint'lerini test eder"""
    
    def test_health_endpoint(self):
        """Health check endpoint'ini test eder"""
        response = requests.get(f"{BASE_URL}/health", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert data["status"] == "healthy"
        assert "models" in data
        assert "baseline" in data["models"]
        assert "transformer" in data["models"]
    
    def test_intents_endpoint(self):
        """Intents listesi endpoint'ini test eder"""
        response = requests.get(f"{BASE_URL}/intents", timeout=TEST_TIMEOUT)
        assert response.status_code == 200
        
        data = response.json()
        assert "intents" in data
        assert len(data["intents"]) == 12
        
        expected_intents = [
            "opening_hours", "fine_policy", "borrow_limit", "room_booking",
            "ebook_access", "wifi", "lost_card", "renewal", "ill", 
            "events", "complaint", "other"
        ]
        
        for intent in expected_intents:
            assert intent in data["intents"]
    
    def test_single_intent_prediction_baseline(self):
        """Baseline model ile tek intent tahmini test eder"""
        test_cases = [
            {
                "text": "Kütüphane saat kaçta açılıyor?",
                "expected_intent": "opening_hours",
                "model_type": "baseline"
            },
            {
                "text": "What are the library hours?",
                "expected_intent": "opening_hours", 
                "model_type": "baseline"
            },
            {
                "text": "Kitabı geç getirdim, ceza var mı?",
                "expected_intent": "fine_policy",
                "model_type": "baseline"
            }
        ]
        
        for case in test_cases:
            payload = {
                "text": case["text"],
                "model_type": case["model_type"],
                "return_probabilities": True,
                "confidence_threshold": 0.7
            }
            
            response = requests.post(
                f"{BASE_URL}/intent", 
                json=payload, 
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code == 200
            
            data = response.json()
            assert "intent" in data
            assert "confidence" in data
            assert "model_type" in data
            assert "processing_time" in data
            assert "timestamp" in data
            
            # Model type kontrolü
            assert data["model_type"] == case["model_type"]
            
            # Confidence kontrolü
            assert 0.0 <= data["confidence"] <= 1.0
            
            # Processing time kontrolü
            assert data["processing_time"] > 0
    
    def test_single_intent_prediction_transformer(self):
        """Transformer model ile tek intent tahmini test eder"""
        test_cases = [
            {
                "text": "How many books can I borrow?",
                "expected_intent": "borrow_limit",
                "model_type": "transformer"
            },
            {
                "text": "Oda rezervasyonu yapabilir miyim?",
                "expected_intent": "room_booking",
                "model_type": "transformer"
            }
        ]
        
        for case in test_cases:
            payload = {
                "text": case["text"],
                "model_type": case["model_type"],
                "return_probabilities": True,
                "confidence_threshold": 0.7
            }
            
            response = requests.post(
                f"{BASE_URL}/intent", 
                json=payload, 
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code == 200
            
            data = response.json()
            assert "intent" in data
            assert "confidence" in data
            assert data["model_type"] == case["model_type"]
    
    def test_batch_intent_prediction(self):
        """Batch intent tahmini test eder"""
        test_texts = [
            "Kütüphane saat kaçta açılıyor?",
            "What are the library hours?",
            "Kitabı geç getirdim, ceza var mı?",
            "How many books can I borrow?",
            "Oda rezervasyonu yapabilir miyim?"
        ]
        
        payload = {
            "texts": test_texts,
            "model_type": "baseline",
            "return_probabilities": True,
            "confidence_threshold": 0.7
        }
        
        response = requests.post(
            f"{BASE_URL}/intent/batch", 
            json=payload, 
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert "predictions" in data
        assert len(data["predictions"]) == len(test_texts)
        
        for prediction in data["predictions"]:
            assert "intent" in prediction
            assert "confidence" in prediction
            assert "text" in prediction
            assert 0.0 <= prediction["confidence"] <= 1.0
    
    def test_confidence_threshold_abstain(self):
        """Düşük confidence durumunda abstain döndürme test eder"""
        # Belirsiz bir metin
        payload = {
            "text": "asdf qwerty 123",
            "model_type": "baseline",
            "confidence_threshold": 0.9  # Yüksek threshold
        }
        
        response = requests.post(
            f"{BASE_URL}/intent", 
            json=payload, 
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code == 200
        
        data = response.json()
        # Düşük confidence durumunda abstain döndürülmeli
        if data["confidence"] < payload["confidence_threshold"]:
            assert data["intent"] == "abstain"
    
    def test_invalid_input_validation(self):
        """Geçersiz input validation test eder"""
        # Boş metin
        payload = {
            "text": "",
            "model_type": "baseline"
        }
        
        response = requests.post(
            f"{BASE_URL}/intent", 
            json=payload, 
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code == 422  # Validation error
        
        # Çok uzun metin
        payload = {
            "text": "a" * 1001,  # 1000 karakter limitini aşar
            "model_type": "baseline"
        }
        
        response = requests.post(
            f"{BASE_URL}/intent", 
            json=payload, 
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code == 422  # Validation error
    
    def test_invalid_model_type(self):
        """Geçersiz model type test eder"""
        payload = {
            "text": "Test metni",
            "model_type": "invalid_model"
        }
        
        response = requests.post(
            f"{BASE_URL}/intent", 
            json=payload, 
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code == 500  # Server error
    
    def test_probabilities_output(self):
        """Probability çıktısı test eder"""
        payload = {
            "text": "Kütüphane saat kaçta açılıyor?",
            "model_type": "baseline",
            "return_probabilities": True
        }
        
        response = requests.post(
            f"{BASE_URL}/intent", 
            json=payload, 
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code == 200
        
        data = response.json()
        assert "probabilities" in data
        
        # Tüm intent'ler için probability olmalı
        probabilities = data["probabilities"]
        assert len(probabilities) == 12
        
        # Probability'ler 0-1 arasında olmalı ve toplamı ~1 olmalı
        total_prob = sum(probabilities.values())
        assert 0.95 <= total_prob <= 1.05
        
        for intent, prob in probabilities.items():
            assert 0.0 <= prob <= 1.0
    
    def test_cross_lingual_performance(self):
        """Çok dilli performans test eder"""
        test_cases = [
            {"text": "Kütüphane saat kaçta açılıyor?", "lang": "turkish"},
            {"text": "What are the library hours?", "lang": "english"},
            {"text": "Library saatleri nedir?", "lang": "mixed"}
        ]
        
        for case in test_cases:
            payload = {
                "text": case["text"],
                "model_type": "transformer",
                "confidence_threshold": 0.5
            }
            
            response = requests.post(
                f"{BASE_URL}/intent", 
                json=payload, 
                timeout=TEST_TIMEOUT
            )
            
            assert response.status_code == 200
            
            data = response.json()
            assert "intent" in data
            assert data["intent"] != ""  # Boş intent döndürülmemeli
    
    def test_response_time_performance(self):
        """Yanıt süresi performans test eder"""
        payload = {
            "text": "Kütüphane saat kaçta açılıyor?",
            "model_type": "baseline"
        }
        
        response = requests.post(
            f"{BASE_URL}/intent", 
            json=payload, 
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code == 200
        
        data = response.json()
        processing_time = data["processing_time"]
        
        # Baseline model 100ms'den hızlı olmalı
        assert processing_time < 0.1
        
        # Transformer model test
        payload["model_type"] = "transformer"
        response = requests.post(
            f"{BASE_URL}/intent", 
            json=payload, 
            timeout=TEST_TIMEOUT
        )
        
        assert response.status_code == 200
        
        data = response.json()
        processing_time = data["processing_time"]
        
        # Transformer model 1 saniyeden hızlı olmalı
        assert processing_time < 1.0

if __name__ == "__main__":
    pytest.main([__file__, "-v"])
