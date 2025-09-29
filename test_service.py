#!/usr/bin/env python3
"""
FastAPI Servis Test Scripti
"""

import requests
import json

def test_health():
    """Health check test"""
    print("Testing health endpoint...")
    response = requests.get("http://localhost:8000/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {response.json()}")
    return response.status_code == 200

def test_intent_classification():
    """Intent classification test"""
    print("\nTesting intent classification...")
    
    test_cases = [
        {"text": "Kütüphane saat kaçta açılıyor?", "expected": "opening_hours"},
        {"text": "What are the library hours?", "expected": "opening_hours"},
        {"text": "Kitabı geç getirdim, ceza var mı?", "expected": "fine_policy"},
        {"text": "How many books can I borrow?", "expected": "borrow_limit"},
        {"text": "Oda rezervasyonu yapabilir miyim?", "expected": "room_booking"},
        {"text": "E-kitap nasıl indirebilirim?", "expected": "ebook_access"},
        {"text": "WiFi şifresi nedir?", "expected": "wifi"},
        {"text": "Kartımı kaybettim", "expected": "lost_card"},
        {"text": "Üyeliğimi nasıl yenilerim?", "expected": "renewal"},
        {"text": "Hastayım, kitabı getiremedim", "expected": "ill"},
        {"text": "Bu hafta hangi etkinlikler var?", "expected": "events"},
        {"text": "Çok gürültülü, şikayet etmek istiyorum", "expected": "complaint"},
        {"text": "Genel bilgi almak istiyorum", "expected": "other"}
    ]
    
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    for i, test_case in enumerate(test_cases, 1):
        print(f"\nTest {i}: {test_case['text']}")
        
        payload = {
            "text": test_case["text"],
            "model_type": "baseline",
            "return_probabilities": True,
            "confidence_threshold": 0.5
        }
        
        try:
            response = requests.post("http://localhost:8000/intent", json=payload)
            
            if response.status_code == 200:
                result = response.json()
                predicted_intent = result["intent"]
                confidence = result["confidence"]
                
                print(f"Predicted: {predicted_intent} (confidence: {confidence:.3f})")
                print(f"Expected: {test_case['expected']}")
                
                if predicted_intent == test_case["expected"]:
                    print("Correct!")
                    correct_predictions += 1
                else:
                    print("Incorrect!")
                    
                # Show top 3 probabilities
                if "probabilities" in result and result["probabilities"]:
                    sorted_probs = sorted(result["probabilities"].items(), 
                                        key=lambda x: x[1], reverse=True)[:3]
                    print("Top 3 probabilities:")
                    for intent, prob in sorted_probs:
                        print(f"  {intent}: {prob:.3f}")
            else:
                print(f"❌ Error: {response.status_code}")
                print(f"Response: {response.text}")
                
        except Exception as e:
            print(f"Exception: {e}")
    
    accuracy = correct_predictions / total_predictions
    print(f"\nOverall Accuracy: {correct_predictions}/{total_predictions} = {accuracy:.2%}")
    
    return accuracy

def test_batch_classification():
    """Batch classification test"""
    print("\nTesting batch classification...")
    
    texts = [
        "Kütüphane saat kaçta açılıyor?",
        "What are the library hours?",
        "Kitabı geç getirdim, ceza var mı?",
        "How many books can I borrow?",
        "Oda rezervasyonu yapabilir miyim?"
    ]
    
    payload = {
        "texts": texts,
        "model_type": "baseline",
        "return_probabilities": True,
        "confidence_threshold": 0.5
    }
    
    try:
        response = requests.post("http://localhost:8000/intent/batch", json=payload)
        
        if response.status_code == 200:
            result = response.json()
            predictions = result["predictions"]
            total_time = result["total_processing_time"]
            
            print(f"Batch processing time: {total_time:.3f}s")
            print(f"Average time per prediction: {total_time/len(texts):.3f}s")
            
            for i, prediction in enumerate(predictions):
                print(f"{i+1}. {texts[i]} -> {prediction['intent']} ({prediction['confidence']:.3f})")
                
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            print(f"Response: {response.text}")
            return False
            
    except Exception as e:
        print(f"Exception: {e}")
        return False

def test_intents_endpoint():
    """Test intents endpoint"""
    print("\nTesting intents endpoint...")
    
    try:
        response = requests.get("http://localhost:8000/intents")
        
        if response.status_code == 200:
            result = response.json()
            intents = result["intents"]
            
            print(f"Available intents ({len(intents)}):")
            for intent, description in intents.items():
                print(f"  {intent}: {description}")
                
            return True
        else:
            print(f"❌ Error: {response.status_code}")
            return False
            
    except Exception as e:
        print(f"Exception: {e}")
        return False

def main():
    """Main test function"""
    print("Starting FastAPI Service Tests...")
    
    # Test health
    health_ok = test_health()
    
    if not health_ok:
        print("❌ Health check failed. Service may not be running.")
        return
    
    # Test intents endpoint
    intents_ok = test_intents_endpoint()
    
    # Test intent classification
    accuracy = test_intent_classification()
    
    # Test batch classification
    batch_ok = test_batch_classification()
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    print(f"Health Check: {'PASS' if health_ok else 'FAIL'}")
    print(f"Intents Endpoint: {'PASS' if intents_ok else 'FAIL'}")
    print(f"Intent Classification: {'PASS' if accuracy > 0.5 else 'FAIL'} (Accuracy: {accuracy:.2%})")
    print(f"Batch Classification: {'PASS' if batch_ok else 'FAIL'}")
    
    if all([health_ok, intents_ok, accuracy > 0.5, batch_ok]):
        print("\nAll tests passed! Service is working correctly.")
    else:
        print("\nSome tests failed. Check the service configuration.")

if __name__ == "__main__":
    main()
