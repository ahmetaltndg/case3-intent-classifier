#!/usr/bin/env python3
"""
Tests for serving API
"""

import pytest
import json
import requests
import time
from pathlib import Path
from fastapi.testclient import TestClient
from src.serving.app import app

class TestServingAPI:
    def setup_method(self):
        """Setup test client"""
        self.client = TestClient(app)
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        assert response.status_code == 200
        
        data = response.json()
        assert "status" in data
        assert "models_loaded" in data
        assert "timestamp" in data
    
    def test_root_endpoint(self):
        """Test root endpoint"""
        response = self.client.get("/")
        assert response.status_code == 200
        
        data = response.json()
        assert "message" in data
        assert "version" in data
        assert "docs" in data
        assert "health" in data
    
    def test_intents_endpoint(self):
        """Test intents endpoint"""
        response = self.client.get("/intents")
        assert response.status_code == 200
        
        data = response.json()
        assert "intents" in data
        assert isinstance(data["intents"], dict)
        
        # Check that all expected intents are present
        expected_intents = [
            "opening_hours", "fine_policy", "borrow_limit", "room_booking",
            "ebook_access", "wifi", "lost_card", "renewal", "ill",
            "events", "complaint", "other"
        ]
        
        for intent in expected_intents:
            assert intent in data["intents"]
    
    def test_models_endpoint(self):
        """Test models endpoint"""
        response = self.client.get("/models")
        assert response.status_code == 200
        
        data = response.json()
        assert "available_models" in data
        assert "default_model" in data
        assert isinstance(data["available_models"], list)
    
    def test_intent_classification_single(self):
        """Test single intent classification"""
        # Test data
        test_data = {
            "text": "Kütüphane saat kaçta açılıyor?",
            "model_type": "baseline",
            "return_probabilities": True,
            "confidence_threshold": 0.5
        }
        
        response = self.client.post("/intent", json=test_data)
        
        # Should return 200 or 500 (if model not loaded)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "intent" in data
            assert "confidence" in data
            assert "model_type" in data
            assert "processing_time" in data
            assert "timestamp" in data
            
            # Check data types
            assert isinstance(data["intent"], str)
            assert isinstance(data["confidence"], float)
            assert isinstance(data["model_type"], str)
            assert isinstance(data["processing_time"], float)
            assert isinstance(data["timestamp"], str)
            
            # Check confidence range
            assert 0.0 <= data["confidence"] <= 1.0
    
    def test_intent_classification_batch(self):
        """Test batch intent classification"""
        # Test data
        test_data = {
            "texts": [
                "Kütüphane saat kaçta açılıyor?",
                "What are the library hours?",
                "Kitabı geç getirdim, ceza var mı?"
            ],
            "model_type": "baseline",
            "return_probabilities": True,
            "confidence_threshold": 0.5
        }
        
        response = self.client.post("/intent/batch", json=test_data)
        
        # Should return 200 or 500 (if model not loaded)
        assert response.status_code in [200, 500]
        
        if response.status_code == 200:
            data = response.json()
            assert "predictions" in data
            assert "total_processing_time" in data
            assert "model_type" in data
            
            # Check predictions
            predictions = data["predictions"]
            assert len(predictions) == len(test_data["texts"])
            
            for prediction in predictions:
                assert "intent" in prediction
                assert "confidence" in prediction
                assert "model_type" in prediction
                assert "processing_time" in prediction
                assert "timestamp" in prediction
    
    def test_intent_classification_validation(self):
        """Test input validation"""
        # Test empty text
        test_data = {
            "text": "",
            "model_type": "baseline"
        }
        
        response = self.client.post("/intent", json=test_data)
        assert response.status_code == 422  # Validation error
        
        # Test text too long
        test_data = {
            "text": "a" * 1001,  # Exceeds max_length
            "model_type": "baseline"
        }
        
        response = self.client.post("/intent", json=test_data)
        assert response.status_code == 422  # Validation error
        
        # Test invalid model type
        test_data = {
            "text": "Test text",
            "model_type": "invalid_model"
        }
        
        response = self.client.post("/intent", json=test_data)
        # Should still work (validation allows any string)
        assert response.status_code in [200, 500]
    
    def test_batch_validation(self):
        """Test batch input validation"""
        # Test empty batch
        test_data = {
            "texts": [],
            "model_type": "baseline"
        }
        
        response = self.client.post("/intent/batch", json=test_data)
        assert response.status_code == 422  # Validation error
        
        # Test batch too large
        test_data = {
            "texts": ["text"] * 101,  # Exceeds max_items
            "model_type": "baseline"
        }
        
        response = self.client.post("/intent/batch", json=test_data)
        assert response.status_code == 422  # Validation error
    
    def test_confidence_threshold(self):
        """Test confidence threshold functionality"""
        test_data = {
            "text": "Merhaba",  # Ambiguous text
            "model_type": "baseline",
            "confidence_threshold": 0.9  # High threshold
        }
        
        response = self.client.post("/intent", json=test_data)
        
        if response.status_code == 200:
            data = response.json()
            # With high threshold, should likely return abstain
            if data["confidence"] < test_data["confidence_threshold"]:
                assert data["intent"] == "abstain"
    
    def test_return_probabilities(self):
        """Test return_probabilities parameter"""
        test_data = {
            "text": "Kütüphane saat kaçta açılıyor?",
            "model_type": "baseline",
            "return_probabilities": True
        }
        
        response = self.client.post("/intent", json=test_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "probabilities" in data
            assert data["probabilities"] is not None
            assert isinstance(data["probabilities"], dict)
        
        # Test with return_probabilities=False
        test_data["return_probabilities"] = False
        
        response = self.client.post("/intent", json=test_data)
        
        if response.status_code == 200:
            data = response.json()
            assert "probabilities" in data
            assert data["probabilities"] is None
    
    def test_different_model_types(self):
        """Test different model types"""
        test_text = "Kütüphane saat kaçta açılıyor?"
        
        for model_type in ["baseline", "transformer"]:
            test_data = {
                "text": test_text,
                "model_type": model_type
            }
            
            response = self.client.post("/intent", json=test_data)
            assert response.status_code in [200, 500]
            
            if response.status_code == 200:
                data = response.json()
                assert data["model_type"] == model_type
    
    def test_processing_time(self):
        """Test processing time measurement"""
        test_data = {
            "text": "Kütüphane saat kaçta açılıyor?",
            "model_type": "baseline"
        }
        
        start_time = time.time()
        response = self.client.post("/intent", json=test_data)
        end_time = time.time()
        
        if response.status_code == 200:
            data = response.json()
            assert "processing_time" in data
            assert isinstance(data["processing_time"], float)
            assert data["processing_time"] > 0
            assert data["processing_time"] < (end_time - start_time) + 1.0  # Allow some margin
    
    def test_error_handling(self):
        """Test error handling"""
        # Test with invalid JSON
        response = self.client.post("/intent", data="invalid json")
        assert response.status_code == 422
        
        # Test with missing required fields
        test_data = {
            "model_type": "baseline"
            # Missing "text" field
        }
        
        response = self.client.post("/intent", json=test_data)
        assert response.status_code == 422
    
    def test_cors_headers(self):
        """Test CORS headers"""
        response = self.client.options("/intent")
        assert response.status_code == 200
        
        # Check CORS headers
        headers = response.headers
        assert "access-control-allow-origin" in headers
        assert "access-control-allow-methods" in headers
        assert "access-control-allow-headers" in headers
    
    def test_response_format(self):
        """Test response format consistency"""
        test_cases = [
            "Kütüphane saat kaçta açılıyor?",
            "What are the library hours?",
            "Kitabı geç getirdim, ceza var mı?",
            "How many books can I borrow?",
            "Oda rezervasyonu yapabilir miyim?"
        ]
        
        for text in test_cases:
            test_data = {
                "text": text,
                "model_type": "baseline"
            }
            
            response = self.client.post("/intent", json=test_data)
            
            if response.status_code == 200:
                data = response.json()
                
                # Check required fields
                required_fields = ["intent", "confidence", "model_type", "processing_time", "timestamp"]
                for field in required_fields:
                    assert field in data
                
                # Check data types
                assert isinstance(data["intent"], str)
                assert isinstance(data["confidence"], float)
                assert isinstance(data["model_type"], str)
                assert isinstance(data["processing_time"], float)
                assert isinstance(data["timestamp"], str)
                
                # Check confidence range
                assert 0.0 <= data["confidence"] <= 1.0
                
                # Check timestamp format
                from datetime import datetime
                try:
                    datetime.fromisoformat(data["timestamp"])
                except ValueError:
                    pytest.fail("Invalid timestamp format")

if __name__ == "__main__":
    pytest.main([__file__])
