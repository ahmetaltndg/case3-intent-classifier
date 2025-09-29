#!/usr/bin/env python3
"""
Generate 20 Real-World Test Examples for Evaluation
"""

import json
from pathlib import Path

def generate_real_test_examples():
    """Generate 20 real-world test examples"""
    
    real_test_examples = [
        # Turkish examples
        {
            "text": "Kütüphane saat kaçta açılıyor?",
            "intent": "opening_hours",
            "language": "tr"
        },
        {
            "text": "Kitabımı geç getirdim, ne kadar ceza ödeyeceğim?",
            "intent": "fine_policy", 
            "language": "tr"
        },
        {
            "text": "Kaç kitap ödünç alabilirim?",
            "intent": "borrow_limit",
            "language": "tr"
        },
        {
            "text": "Çalışma odası rezervasyonu yapmak istiyorum",
            "intent": "room_booking",
            "language": "tr"
        },
        {
            "text": "E-kitap nasıl indirebilirim?",
            "intent": "ebook_access",
            "language": "tr"
        },
        {
            "text": "WiFi şifresi nedir?",
            "intent": "wifi",
            "language": "tr"
        },
        {
            "text": "Kartımı kaybettim, ne yapmalıyım?",
            "intent": "lost_card",
            "language": "tr"
        },
        {
            "text": "Üyeliğimi nasıl yenilerim?",
            "intent": "renewal",
            "language": "tr"
        },
        {
            "text": "Hastayım, kitabı getiremedim",
            "intent": "ill",
            "language": "tr"
        },
        {
            "text": "Bu hafta hangi etkinlikler var?",
            "intent": "events",
            "language": "tr"
        },
        {
            "text": "Çok gürültülü, şikayet etmek istiyorum",
            "intent": "complaint",
            "language": "tr"
        },
        {
            "text": "Genel bilgi almak istiyorum",
            "intent": "other",
            "language": "tr"
        },
        
        # English examples
        {
            "text": "What are the library hours?",
            "intent": "opening_hours",
            "language": "en"
        },
        {
            "text": "I returned my book late, what's the fine?",
            "intent": "fine_policy",
            "language": "en"
        },
        {
            "text": "How many books can I borrow?",
            "intent": "borrow_limit",
            "language": "en"
        },
        {
            "text": "Can I book a study room?",
            "intent": "room_booking",
            "language": "en"
        },
        {
            "text": "How do I download e-books?",
            "intent": "ebook_access",
            "language": "en"
        },
        {
            "text": "What's the WiFi password?",
            "intent": "wifi",
            "language": "en"
        },
        {
            "text": "I lost my library card",
            "intent": "lost_card",
            "language": "en"
        },
        {
            "text": "How do I renew my membership?",
            "intent": "renewal",
            "language": "en"
        }
    ]
    
    return real_test_examples

def main():
    """Generate and save real test examples"""
    print("Generating 20 real-world test examples...")
    
    examples = generate_real_test_examples()
    
    # Save to file
    output_path = Path("data/real_test_examples.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    
    print(f"Saved {len(examples)} real test examples to {output_path}")
    
    # Print summary
    tr_count = sum(1 for ex in examples if ex['language'] == 'tr')
    en_count = sum(1 for ex in examples if ex['language'] == 'en')
    
    print(f"Turkish examples: {tr_count}")
    print(f"English examples: {en_count}")
    
    # Show intent distribution
    intent_counts = {}
    for ex in examples:
        intent = ex['intent']
        intent_counts[intent] = intent_counts.get(intent, 0) + 1
    
    print("\nIntent distribution:")
    for intent, count in sorted(intent_counts.items()):
        print(f"  {intent}: {count}")

if __name__ == "__main__":
    main()