#!/usr/bin/env python3
"""
Demo Sentetik Veri Üretimi - OpenAI API olmadan
Intent başına 100+ örnek üretir
"""

import json
import random
from pathlib import Path
from typing import List, Dict, Any
import yaml

class DemoDataGenerator:
    def __init__(self):
        self.intents = self.load_intents()
        self.output_dir = Path("data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Türkçe ve İngilizce örnek şablonları
        self.templates = {
            "opening_hours": {
                "tr": [
                    "Kütüphane saat kaçta açılıyor?",
                    "Pazar günü açık mı?",
                    "Akşam kaçta kapanıyor?",
                    "Hafta sonu çalışma saatleri nedir?",
                    "Bayram günlerinde açık mı?",
                    "Sabah kaçta açılıyor?",
                    "Öğle arası var mı?",
                    "Gece açık mı?",
                    "Çalışma saatleri nedir?",
                    "Ne zaman açık?"
                ],
                "en": [
                    "What are the library hours?",
                    "Is it open on Sundays?",
                    "What time does it close?",
                    "What are the weekend hours?",
                    "Is it open on holidays?",
                    "What time does it open?",
                    "Is there a lunch break?",
                    "Is it open at night?",
                    "What are the operating hours?",
                    "When is it open?"
                ]
            },
            "fine_policy": {
                "tr": [
                    "Kitabı geç getirdim, ne kadar ceza öderim?",
                    "Ceza nasıl hesaplanıyor?",
                    "Gecikme ücreti nedir?",
                    "Ne kadar ceza var?",
                    "Ceza ödemek istemiyorum",
                    "Ceza miktarı nedir?",
                    "Gecikme cezası nasıl?",
                    "Ceza politikası nedir?",
                    "Ne kadar gecikme cezası?",
                    "Ceza hesaplama nasıl?"
                ],
                "en": [
                    "I returned the book late, how much fine do I pay?",
                    "How is the fine calculated?",
                    "What is the late fee?",
                    "How much is the fine?",
                    "I don't want to pay the fine",
                    "What is the fine amount?",
                    "How much is the late fee?",
                    "What is the fine policy?",
                    "How much late fee?",
                    "How is the fine calculated?"
                ]
            },
            "borrow_limit": {
                "tr": [
                    "Kaç kitap alabilirim?",
                    "Öğrenci limiti nedir?",
                    "Maksimum kaç kitap?",
                    "Ödünç alma limiti nedir?",
                    "Kaç kitap ödünç alabilirim?",
                    "Limit nedir?",
                    "Maksimum ödünç alma?",
                    "Kaç kitap alabilirim?",
                    "Ödünç limit nedir?",
                    "Ne kadar kitap alabilirim?"
                ],
                "en": [
                    "How many books can I borrow?",
                    "What is the student limit?",
                    "What is the maximum number of books?",
                    "What is the borrowing limit?",
                    "How many books can I borrow?",
                    "What is the limit?",
                    "Maximum borrowing?",
                    "How many books can I take?",
                    "What is the borrowing limit?",
                    "How many books can I borrow?"
                ]
            },
            "room_booking": {
                "tr": [
                    "Grup çalışma odası rezerve edebilir miyim?",
                    "Sessiz çalışma alanı var mı?",
                    "Oda rezervasyonu yapabilir miyim?",
                    "Çalışma odası rezerve etmek istiyorum",
                    "Grup odası var mı?",
                    "Rezervasyon nasıl yapılır?",
                    "Çalışma alanı rezervasyonu",
                    "Oda kiralamak istiyorum",
                    "Grup çalışma alanı",
                    "Rezervasyon sistemi"
                ],
                "en": [
                    "Can I reserve a group study room?",
                    "Is there a quiet study area?",
                    "Can I book a room?",
                    "I want to reserve a study room",
                    "Is there a group room?",
                    "How do I make a reservation?",
                    "Study area reservation",
                    "I want to rent a room",
                    "Group study area",
                    "Reservation system"
                ]
            },
            "ebook_access": {
                "tr": [
                    "E-kitap nasıl indirebilirim?",
                    "Online veritabanına nasıl erişirim?",
                    "E-kaynaklara erişim nasıl?",
                    "Dijital kaynaklar nerede?",
                    "E-kitap erişimi nasıl?",
                    "Online kütüphane nasıl?",
                    "E-kaynak erişimi",
                    "Dijital erişim nasıl?",
                    "E-kitap indirme",
                    "Online erişim"
                ],
                "en": [
                    "How do I download e-books?",
                    "How do I access online databases?",
                    "How to access e-resources?",
                    "Where are digital resources?",
                    "How to access e-books?",
                    "How to use online library?",
                    "E-resource access",
                    "How to access digital?",
                    "E-book download",
                    "Online access"
                ]
            },
            "wifi": {
                "tr": [
                    "WiFi şifresi nedir?",
                    "İnternet bağlantım çalışmıyor",
                    "WiFi çekmiyor",
                    "İnternet sorunu var",
                    "Bağlantı problemi",
                    "WiFi nasıl bağlanır?",
                    "İnternet erişimi",
                    "Bağlantı sorunu",
                    "WiFi problemi",
                    "İnternet çalışmıyor"
                ],
                "en": [
                    "What's the WiFi password?",
                    "My internet connection isn't working",
                    "WiFi is not working",
                    "There's an internet problem",
                    "Connection problem",
                    "How to connect to WiFi?",
                    "Internet access",
                    "Connection issue",
                    "WiFi problem",
                    "Internet not working"
                ]
            },
            "lost_card": {
                "tr": [
                    "Kartımı kaybettim, ne yapmalıyım?",
                    "Yeni kart nasıl alırım?",
                    "Kart çıkarmak istiyorum",
                    "Kayıp kart raporu",
                    "Kart yenileme",
                    "Yeni kart başvurusu",
                    "Kart kaybı",
                    "Kart değiştirme",
                    "Yeni kart",
                    "Kart problemi"
                ],
                "en": [
                    "I lost my card, what should I do?",
                    "How do I get a new card?",
                    "I want to get a new card",
                    "Lost card report",
                    "Card renewal",
                    "New card application",
                    "Lost card",
                    "Card replacement",
                    "New card",
                    "Card problem"
                ]
            },
            "renewal": {
                "tr": [
                    "Üyeliğimi nasıl yenilerim?",
                    "Kartımın süresi doldu",
                    "Yenileme nasıl yapılır?",
                    "Üyelik yenileme",
                    "Kart yenileme",
                    "Süre uzatma",
                    "Yenileme işlemi",
                    "Üyelik süresi",
                    "Kart süresi",
                    "Yenileme"
                ],
                "en": [
                    "How do I renew my membership?",
                    "My card has expired",
                    "How to renew?",
                    "Membership renewal",
                    "Card renewal",
                    "Extension",
                    "Renewal process",
                    "Membership period",
                    "Card period",
                    "Renewal"
                ]
            },
            "ill": {
                "tr": [
                    "Hastayım, kitabı getiremedim",
                    "Sağlık raporu gerekli mi?",
                    "Hasta olduğum için geç kaldım",
                    "Sağlık sorunu",
                    "Hastalık nedeniyle gecikme",
                    "Doktor raporu",
                    "Sağlık durumu",
                    "Hasta raporu",
                    "Tıbbi durum",
                    "Sağlık problemi"
                ],
                "en": [
                    "I'm sick, I couldn't return the book",
                    "Do I need a medical certificate?",
                    "I'm late because I'm sick",
                    "Health problem",
                    "Late due to illness",
                    "Doctor's report",
                    "Health condition",
                    "Medical report",
                    "Medical condition",
                    "Health issue"
                ]
            },
            "events": {
                "tr": [
                    "Bu hafta hangi etkinlikler var?",
                    "Yazarlık workshopuna katılmak istiyorum",
                    "Etkinlik takvimi",
                    "Seminer programı",
                    "Workshop etkinlikleri",
                    "Kültürel etkinlikler",
                    "Eğitim programları",
                    "Etkinlik listesi",
                    "Program takvimi",
                    "Etkinlikler"
                ],
                "en": [
                    "What events are there this week?",
                    "I want to attend the writing workshop",
                    "Event calendar",
                    "Seminar program",
                    "Workshop events",
                    "Cultural events",
                    "Educational programs",
                    "Event list",
                    "Program schedule",
                    "Events"
                ]
            },
            "complaint": {
                "tr": [
                    "Çok gürültülü, şikayet etmek istiyorum",
                    "Hizmet kalitesinden memnun değilim",
                    "Şikayet formu",
                    "Memnuniyetsizlik",
                    "Sorun bildirimi",
                    "Hizmet sorunu",
                    "Kalite problemi",
                    "Şikayet",
                    "Problem bildirimi",
                    "Memnuniyetsizlik"
                ],
                "en": [
                    "It's too noisy, I want to complain",
                    "I'm not satisfied with the service quality",
                    "Complaint form",
                    "Dissatisfaction",
                    "Problem report",
                    "Service problem",
                    "Quality issue",
                    "Complaint",
                    "Problem report",
                    "Dissatisfaction"
                ]
            },
            "other": {
                "tr": [
                    "Genel bilgi almak istiyorum",
                    "Bana yardımcı olur musunuz?",
                    "Bilgi almak istiyorum",
                    "Yardım istiyorum",
                    "Sorunuz var",
                    "Bilgi",
                    "Yardım",
                    "Destek",
                    "Sorun",
                    "Merhaba"
                ],
                "en": [
                    "I need some general information",
                    "Can you help me?",
                    "I want to get information",
                    "I need help",
                    "I have a question",
                    "Information",
                    "Help",
                    "Support",
                    "Question",
                    "Hello"
                ]
            }
        }
    
    def load_intents(self) -> Dict[str, str]:
        """Intent tanımlarını yükle"""
        intents_file = Path("data/intents.yaml")
        if intents_file.exists():
            with open(intents_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            return {
                "opening_hours": "Kütüphane çalışma saatleri hakkında sorular",
                "fine_policy": "Ceza politikası, gecikme ücretleri hakkında sorular", 
                "borrow_limit": "Ödünç alma limitleri hakkında sorular",
                "room_booking": "Oda rezervasyonu, çalışma alanı rezervasyonu",
                "ebook_access": "E-kitap erişimi, dijital kaynaklar",
                "wifi": "WiFi bağlantı sorunları",
                "lost_card": "Kayıp kart, kart yenileme",
                "renewal": "Üyelik yenileme, kart yenileme",
                "ill": "Hastalık, sağlık durumu",
                "events": "Etkinlikler, seminerler, workshoplar",
                "complaint": "Şikayetler, memnuniyetsizlik",
                "other": "Diğer konular, belirsiz sorular"
            }
    
    def generate_variations(self, base_text: str, language: str) -> List[str]:
        """Temel metinden varyasyonlar üret"""
        variations = [base_text]
        
        # Dil değişiklikleri
        if language == "tr":
            # Türkçe varyasyonlar
            variations.extend([
                base_text.replace("?", ""),
                base_text.replace("?", "?"),
                base_text.lower(),
                base_text.upper(),
                f"Merhaba, {base_text.lower()}",
                f"Selam, {base_text.lower()}",
                f"Lütfen, {base_text.lower()}",
                f"Acaba {base_text.lower()}",
                f"Şu an {base_text.lower()}",
                f"Bugün {base_text.lower()}"
            ])
        else:
            # İngilizce varyasyonlar
            variations.extend([
                base_text.replace("?", ""),
                base_text.replace("?", "?"),
                base_text.lower(),
                base_text.upper(),
                f"Hello, {base_text.lower()}",
                f"Hi, {base_text.lower()}",
                f"Please, {base_text.lower()}",
                f"I wonder {base_text.lower()}",
                f"Right now {base_text.lower()}",
                f"Today {base_text.lower()}"
            ])
        
        return variations
    
    def generate_intent_data(self, intent: str, description: str) -> List[Dict[str, Any]]:
        """Tek intent için veri üret"""
        print(f"Generating data for intent: {intent}")
        
        data = []
        templates = self.templates.get(intent, {"tr": [], "en": []})
        
        # Her dil için 50 örnek üret
        for lang in ["tr", "en"]:
            lang_templates = templates.get(lang, [])
            if not lang_templates:
                continue
                
            for i in range(50):
                # Rastgele template seç
                base_template = random.choice(lang_templates)
                
                # Varyasyonlar üret
                variations = self.generate_variations(base_template, lang)
                selected_text = random.choice(variations)
                
                # Karışık dil örnekleri ekle (her 10 örnekte bir)
                if i % 10 == 0 and lang == "tr":
                    # Türkçe-İngilizce karışık
                    en_template = random.choice(templates.get("en", [""]))
                    mixed_text = f"{selected_text} {en_template}"
                    selected_text = mixed_text
                    lang = "mixed"
                
                data.append({
                    "id": f"{intent}_{len(data)+1}",
                    "text": selected_text,
                    "intent": intent,
                    "language": lang,
                    "length": len(selected_text.split()),
                    "generated_at": 1640995200 + i  # Demo timestamp
                })
        
        print(f"Generated {len(data)} examples for {intent}")
        return data
    
    def generate_all_data(self):
        """Tüm intentler için veri üret"""
        all_data = []
        
        for intent, description in self.intents.items():
            try:
                intent_data = self.generate_intent_data(intent, description)
                all_data.extend(intent_data)
                
                # Her intent için ayrı dosya kaydet
                intent_file = self.output_dir / f"{intent}.json"
                with open(intent_file, 'w', encoding='utf-8') as f:
                    json.dump(intent_data, f, ensure_ascii=False, indent=2)
                
            except Exception as e:
                print(f"Error generating data for {intent}: {e}")
                continue
        
        # Birleşik veri kaydet
        combined_file = self.output_dir / "synthetic_data.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        # Metadata kaydet
        metadata = {
            "total_examples": len(all_data),
            "intents": list(self.intents.keys()),
            "generation_time": 1640995200,
            "examples_per_intent": len(all_data) // len(self.intents),
            "generation_method": "demo_template_based"
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Generated {len(all_data)} total examples")
        print(f"Data saved to {self.output_dir}")
        
        return all_data

def main():
    """Ana fonksiyon"""
    print("Starting demo synthetic data generation...")
    
    generator = DemoDataGenerator()
    data = generator.generate_all_data()
    
    print("Demo synthetic data generation completed!")
    print(f"Total examples: {len(data)}")
    
    # Özet yazdır
    intent_counts = {}
    for item in data:
        intent = item['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print("\nExamples per intent:")
    for intent, count in intent_counts.items():
        print(f"  {intent}: {count}")

if __name__ == "__main__":
    main()
