#!/usr/bin/env python3
"""
Tests for data leakage detection
"""

import pytest
import json
import numpy as np
from pathlib import Path
from collections import Counter
from sklearn.model_selection import train_test_split

class TestDataLeakage:
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
    
    def test_train_test_split(self):
        """Test that train/test split is properly stratified"""
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, intents, test_size=0.3, random_state=42, stratify=intents
        )
        
        # Check that both sets have examples
        assert len(X_train) > 0
        assert len(X_test) > 0
        
        # Check that both sets have all intents (stratified)
        train_intents = set(y_train)
        test_intents = set(y_test)
        all_intents = set(intents)
        
        # All intents should be present in both sets
        assert train_intents == all_intents
        assert test_intents == all_intents
    
    def test_no_duplicate_texts(self):
        """Test that there are no duplicate texts between train and test"""
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, intents, test_size=0.3, random_state=42, stratify=intents
        )
        
        # Check for duplicates
        train_texts = set(X_train)
        test_texts = set(X_test)
        
        # No overlap between train and test
        assert len(train_texts.intersection(test_texts)) == 0
    
    def test_intent_distribution(self):
        """Test that intent distribution is maintained in train/test split"""
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        # Get original distribution
        original_dist = Counter(intents)
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, intents, test_size=0.3, random_state=42, stratify=intents
        )
        
        # Get train and test distributions
        train_dist = Counter(y_train)
        test_dist = Counter(y_test)
        
        # Check that all intents are present in both sets
        assert set(train_dist.keys()) == set(original_dist.keys())
        assert set(test_dist.keys()) == set(original_dist.keys())
        
        # Check that proportions are roughly maintained
        for intent in original_dist.keys():
            train_ratio = train_dist[intent] / len(y_train)
            test_ratio = test_dist[intent] / len(y_test)
            original_ratio = original_dist[intent] / len(intents)
            
            # Ratios should be similar (within 10%)
            assert abs(train_ratio - original_ratio) < 0.1
            assert abs(test_ratio - original_ratio) < 0.1
    
    def test_data_consistency(self):
        """Test that data is consistent across splits"""
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, intents, test_size=0.3, random_state=42, stratify=intents
        )
        
        # Check that text-intent pairs are consistent
        for i, text in enumerate(X_train):
            original_idx = texts.index(text)
            assert y_train[i] == intents[original_idx]
        
        for i, text in enumerate(X_test):
            original_idx = texts.index(text)
            assert y_test[i] == intents[original_idx]
    
    def test_random_state_reproducibility(self):
        """Test that random state ensures reproducibility"""
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        # Split data twice with same random state
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            texts, intents, test_size=0.3, random_state=42, stratify=intents
        )
        
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            texts, intents, test_size=0.3, random_state=42, stratify=intents
        )
        
        # Results should be identical
        assert X_train1 == X_train2
        assert X_test1 == X_test2
        assert y_train1 == y_train2
        assert y_test1 == y_test2
    
    def test_different_random_states(self):
        """Test that different random states produce different splits"""
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        # Split data with different random states
        X_train1, X_test1, y_train1, y_test1 = train_test_split(
            texts, intents, test_size=0.3, random_state=42, stratify=intents
        )
        
        X_train2, X_test2, y_train2, y_test2 = train_test_split(
            texts, intents, test_size=0.3, random_state=123, stratify=intents
        )
        
        # Results should be different (with high probability)
        assert X_train1 != X_train2 or X_test1 != X_test2
    
    def test_test_size_constraints(self):
        """Test that test size constraints are respected"""
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        # Test different test sizes
        for test_size in [0.2, 0.3]:
            X_train, X_test, y_train, y_test = train_test_split(
                texts, intents, test_size=test_size, random_state=42, stratify=intents
            )
            
            # Check test size is approximately correct
            actual_test_size = len(X_test) / len(texts)
            assert abs(actual_test_size - test_size) < 0.1
    
    def test_minimum_samples_per_class(self):
        """Test that each class has minimum samples in both sets"""
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, intents, test_size=0.3, random_state=42, stratify=intents
        )
        
        # Check that each intent has at least 1 sample in both sets
        train_intents = Counter(y_train)
        test_intents = Counter(y_test)
        
        for intent in set(intents):
            assert train_intents[intent] >= 1
            assert test_intents[intent] >= 1
    
    def test_data_integrity(self):
        """Test that original data is not modified during splitting"""
        texts = [item['text'] for item in self.test_data]
        intents = [item['intent'] for item in self.test_data]
        
        # Store original data
        original_texts = texts.copy()
        original_intents = intents.copy()
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            texts, intents, test_size=0.3, random_state=42, stratify=intents
        )
        
        # Check that original data is unchanged
        assert texts == original_texts
        assert intents == original_intents
    
    def test_empty_data_handling(self):
        """Test handling of edge cases"""
        # Test with empty data
        with pytest.raises(ValueError):
            train_test_split([], [], test_size=0.3, random_state=42)
        
        # Test with single sample
        with pytest.raises(ValueError):
            train_test_split(["text"], ["intent"], test_size=0.5, random_state=42, stratify=["intent"])
    
    def test_imbalanced_data(self):
        """Test handling of imbalanced data"""
        # Create imbalanced data
        imbalanced_texts = ["text1", "text2", "text3", "text4", "text5"]
        imbalanced_intents = ["intent1", "intent1", "intent1", "intent2", "intent2"]
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            imbalanced_texts, imbalanced_intents, test_size=0.4, random_state=42, stratify=imbalanced_intents
        )
        
        # Check that both sets have both intents
        assert len(set(y_train)) == 2
        assert len(set(y_test)) == 2
        
        # Check that proportions are maintained
        train_intent1_ratio = y_train.count("intent1") / len(y_train)
        test_intent1_ratio = y_test.count("intent1") / len(y_test)
        original_intent1_ratio = imbalanced_intents.count("intent1") / len(imbalanced_intents)
        
        assert abs(train_intent1_ratio - original_intent1_ratio) < 0.2
        assert abs(test_intent1_ratio - original_intent1_ratio) < 0.2

if __name__ == "__main__":
    pytest.main([__file__])
