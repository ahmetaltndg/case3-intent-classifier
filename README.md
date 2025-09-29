# CASE 3 Intent Classifier

Kütüphane müşteri hizmetleri sorgularını Türkçe ve İngilizce olarak 12 intent'e sınıflandıran yapay zeka sistemi.

## Proje Özeti
Bu proje, LLM tabanlı sentetik veriyle eğitilmiş iki model (Baseline ve Transformer) kullanır ve FastAPI ile API servisi sunar.

## Kurulum & Çalıştırma
```bash
git clone https://github.com/ahmetaltndg/case3-intent-classifier.git
cd case3-intent-classifier
pip install -r requirements.txt
python src/serve.py
```

## Temel Komutlar
- Servisi başlat: `python src/serve.py`
- Testleri çalıştır: `python -m pytest tests/ -v`
- Baseline modeli eğit: `python src/training/train_baseline.py`
- Transformer modeli eğit: `python scripts/train_transformer.py`
- Sentetik veri üret: `python scripts/generate_synthetic_data.py`

## API Kullanımı
### Tekli Sınıflandırma
```bash
curl -X POST "http://localhost:8000/intent" -H "Content-Type: application/json" -d '{"text": "Kütüphane saat kaçta açılıyor?"}'
```
### Batch Sınıflandırma
```bash
curl -X POST "http://localhost:8000/intent/batch" -H "Content-Type: application/json" -d '{"texts": ["Kütüphane saat kaçta açılıyor?", "What are the library hours?"]}'
```
### Sağlık Kontrolü
```bash
curl http://localhost:8000/health
```

## Python ile Kullanım
```python
import requests
response = requests.post("http://localhost:8000/intent", json={"text": "Kütüphane saat kaçta açılıyor?"})
print(response.json())
```

## Testler
- Tüm testler: `python -m pytest tests/ -v`
- Manuel test: `python test_service.py`

## Veri & Model Yapısı
- Sentetik ve gerçek veri: `data/`
- Eğitilmiş modeller: `artifacts/`


## Desteklenen Intent'ler
`opening_hours`, `fine_policy`, `borrow_limit`, `room_booking`, `ebook_access`, `wifi`, `lost_card`, `renewal`, `ill`, `events`, `complaint`, `other`

## Kısa Fonksiyon Listesi
- Sentetik veri üretimi
- Model eğitimi
- API ile intent tahmini
- Batch tahmin
- Sağlık kontrolü
- Test ve raporlama


 **Bağımlılıkları Yükleyin**

