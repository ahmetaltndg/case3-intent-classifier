#!/usr/bin/env python3
"""
Synthetic Data Generation Script for Intent Classification
Generates 100 examples per intent using LLM prompts
"""

import json
import os
import time
from typing import List, Dict, Any
from pathlib import Path
import yaml
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class SyntheticDataGenerator:
    def __init__(self):
        # Configure Gemini API
        genai.configure(api_key="AIzaSyDnSarqf-iC3SM4n-baAz853eFwnzHZjIo")
        self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
        self.intents = self.load_intents()
        self.output_dir = Path("data/raw")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def load_intents(self) -> Dict[str, str]:
        """Load intent definitions from intents.yaml"""
        intents_file = Path("data/intents.yaml")
        if intents_file.exists():
            with open(intents_file, 'r', encoding='utf-8') as f:
                return yaml.safe_load(f)
        else:
            # Fallback to hardcoded intents
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
    
    def generate_prompt(self, intent: str, description: str) -> str:
        """Generate prompt for synthetic data generation"""
        return f"""Sen bir kütüphane müşteri hizmetleri uzmanısın. Aşağıdaki intent kategorisi için gerçekçi kullanıcı cümleleri üret:

Intent: {intent}
Açıklama: {description}

Gereksinimler:
- Türkçe ve İngilizce karışık cümleler
- Gerçekçi, günlük dil kullanımı
- Çeşitli ifade tarzları (formal, informal, kızgın, nazik)
- 120 farklı örnek üret
- Her cümle farklı olmalı
- Farklı uzunluklarda cümleler (kısa, orta, uzun)
- Farklı duygusal tonlar

Örnek format:
1. "Cümle 1"
2. "Cümle 2"
...
120. "Cümle 120"

Lütfen sadece cümleleri listeleyin, başka açıklama eklemeyin."""

    def call_llm(self, prompt: str, max_retries: int = 3) -> str:
        """Call Gemini LLM with retry logic"""
        for attempt in range(max_retries):
            try:
                # Create full prompt with system message
                full_prompt = f"""Sen bir kütüphane müşteri hizmetleri uzmanısın. Gerçekçi kullanıcı cümleleri üretiyorsun.

{prompt}"""
                
                response = self.model.generate_content(full_prompt)
                return response.text
            except Exception as e:
                print(f"Attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(2 ** attempt)  # Exponential backoff
                else:
                    raise
    
    def parse_examples(self, response: str) -> List[str]:
        """Parse examples from LLM response"""
        examples = []
        lines = response.strip().split('\n')
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # Skip empty lines and headers
            if not line or line.startswith(('Intent:', 'Açıklama:', 'Kurallar:', 'Örnekler:', 'İşte', 'Here')):
                continue
                
            # Remove bullet points
            if line.startswith(('*', '-', '•')):
                line = line[1:].strip()
            
            # Remove numbering (1., 2., etc.)
            if line and line[0].isdigit() and ('.' in line or ')' in line):
                # Find the first quote
                quote_start = line.find('"')
                if quote_start != -1:
                    # Find the last quote
                    quote_end = line.rfind('"')
                    if quote_end > quote_start:
                        example = line[quote_start+1:quote_end]
                        if example.strip() and len(example.strip()) > 5:
                            examples.append(example.strip())
            elif line.startswith('"') and line.endswith('"'):
                # Direct quote format
                example = line[1:-1]
                if example.strip() and len(example.strip()) > 5:
                    examples.append(example.strip())
            elif len(line) > 10 and not line.startswith(('1.', '2.', '3.', '4.', '5.')):
                # Plain text without quotes
                if line and len(line) > 10:
                    examples.append(line)
        
        return examples
    
    def generate_intent_data(self, intent: str, description: str) -> List[Dict[str, Any]]:
        """Generate data for a single intent"""
        print(f"Generating data for intent: {intent}")
        
        prompt = self.generate_prompt(intent, description)
        response = self.call_llm(prompt)
        examples = self.parse_examples(response)
        
        # Create data entries
        data = []
        for i, example in enumerate(examples):
            data.append({
                "id": f"{intent}_{i+1}",
                "text": example,
                "intent": intent,
                "language": self.detect_language(example),
                "length": len(example.split()),
                "generated_at": time.time()
            })
        
        print(f"Generated {len(data)} examples for {intent}")
        return data
    
    def detect_language(self, text: str) -> str:
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
    
    def generate_all_data(self):
        """Generate data for all intents"""
        all_data = []
        
        for intent, description in self.intents.items():
            try:
                intent_data = self.generate_intent_data(intent, description)
                all_data.extend(intent_data)
                
                # Save individual intent data
                intent_file = self.output_dir / f"{intent}.json"
                with open(intent_file, 'w', encoding='utf-8') as f:
                    json.dump(intent_data, f, ensure_ascii=False, indent=2)
                
                # Add delay to avoid rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"Error generating data for {intent}: {e}")
                continue
        
        # Save combined data
        combined_file = self.output_dir / "synthetic_data.json"
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        
        # Save metadata
        metadata = {
            "total_examples": len(all_data),
            "intents": list(self.intents.keys()),
            "generation_time": time.time(),
            "examples_per_intent": len(all_data) // len(self.intents)
        }
        
        metadata_file = self.output_dir / "metadata.json"
        with open(metadata_file, 'w', encoding='utf-8') as f:
            json.dump(metadata, f, ensure_ascii=False, indent=2)
        
        print(f"Generated {len(all_data)} total examples")
        print(f"Data saved to {self.output_dir}")
        
        return all_data

def main():
    """Main function"""
    print("Starting synthetic data generation...")
    
    # Gemini API key is already configured in the class
    
    generator = SyntheticDataGenerator()
    data = generator.generate_all_data()
    
    print("Synthetic data generation completed!")
    print(f"Total examples: {len(data)}")
    
    # Print summary
    intent_counts = {}
    for item in data:
        intent = item['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print("\nExamples per intent:")
    for intent, count in intent_counts.items():
        print(f"  {intent}: {count}")

if __name__ == "__main__":
    main()
