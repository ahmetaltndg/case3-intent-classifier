# EXPERIMENTS.md - Deney Günlüğü ve Sonuçları

## 🧪 Deney Genel Bakış

Bu doküman, CASE 3 Intent Classifier'ın geliştirilmesi sırasında yapılan tüm deneyleri takip eder, data generation, model training ve evaluation aşamalarını içerir.

## 📊 Veri Üretimi Deneyleri

### Deney 1: LLM Seçimi
**Tarih**: 2025-09-24  
**Amaç**: Synthetic data generation için optimal LLM seçimi

**Test Edilen Adaylar**:
- OpenAI GPT-4: Yüksek kalite, pahalı
- OpenAI GPT-3.5-turbo: İyi kalite, orta maliyet
- Google Gemini 2.0 Flash: İyi kalite, maliyet-etkin

**Sonuçlar**:
```
Model                | Kalite | Maliyet | Çok Dilli | Karar
---------------------|--------|---------|-----------|----------
GPT-4               | 9/10   | Yüksek  | 9/10      | ❌ Çok pahalı
GPT-3.5-turbo       | 7/10   | Orta    | 7/10      | ❌ Quota aşıldı
Gemini 2.0 Flash    | 8/10   | Düşük   | 8/10      | ✅ Seçildi
```

**Sonuç**: Maliyet-etkinlik ve iyi çok dilli destek için Gemini 2.0 Flash seçildi.

### Deney 2: Veri Hacmi Optimizasyonu
**Tarih**: 2025-09-25  
**Amaç**: Intent başına optimal örnek sayısını belirleme

**Test Edilen Hacimler**:
- 100 örnek/intent (baseline gereksinim)
- 120 örnek/intent (deduplication için hedef)
- 150 örnek/intent (aşırı)

**Sonuçlar**:
```
Hacim | Üretilen | Dedup Sonrası | Kalite | Karar
------|----------|---------------|--------|----------
100   | 1,200    | ~900          | İyi    | ❌ Yetersiz
120   | 1,440    | ~1,100        | İyi    | ✅ Optimal
150   | 1,800    | ~1,400        | İyi    | ❌ Aşırı
```

**Sonuç**: Intent başına 120 örnek optimal denge sağlar.

### Deney 3: Deduplication Stratejisi
**Tarih**: 2025-09-26  
**Amaç**: En iyi deduplication yöntemini seçme

**Test Edilen Yöntemler**:
- MinHash: Hızlı, yaklaşık
- Embedding + Clustering: Doğru, yavaş
- Simple string matching: Hızlı, sınırlı

**Sonuçlar**:
```
Yöntem                | Doğruluk | Hız   | Memory | Karar
----------------------|----------|-------|--------|----------
MinHash              | 7/10     | 9/10  | 8/10   | ❌ Çok yaklaşık
Embedding + DBSCAN   | 9/10     | 6/10  | 6/10   | ✅ Seçildi
String Matching      | 5/10     | 10/10 | 10/10  | ❌ Çok sınırlı
```

**Final Dataset**: 12 intent boyunca 2,713 örnek

## 🤖 Model Eğitimi Deneyleri

### Deney 4: Baseline Model Konfigürasyonu
**Tarih**: 2025-09-27  
**Amaç**: TF-IDF + Logistic Regression optimizasyonu

**Test Edilen Parametreler**:
```python
# Konfigürasyon A: Conservative
max_features = 5,000
ngram_range = (1, 2)
C = 1.0

# Konfigürasyon B: Balanced (Seçilen)
max_features = 10,000
ngram_range = (1, 2)
C = 1.0

# Konfigürasyon C: Aggressive
max_features = 20,000
ngram_range = (1, 3)
C = 10.0
```

**Sonuçlar**:
```
Config | Doğruluk | F1-Score | Hız   | Memory | Karar
-------|----------|----------|-------|--------|----------
A      | 82.1%    | 0.815    | 9/10  | 9/10   | ❌ Çok conservative
B      | 86.5%    | 0.861    | 8/10  | 7/10   | ✅ Seçildi
C      | 87.2%    | 0.868    | 6/10  | 5/10   | ❌ Çok kaynak yoğun
```

### Deney 5: Transformer Model Fine-tuning
**Tarih**: 2025-09-28  
**Amaç**: DistilBERT fine-tuning optimizasyonu

**Test Edilen Hyperparametreler**:
```python
# Konfigürasyon A: Conservative
learning_rate = 2e-5
epochs = 2
batch_size = 16

# Konfigürasyon B: Balanced (Seçilen)
learning_rate = 5e-5
epochs = 3
batch_size = 8

# Konfigürasyon C: Aggressive
learning_rate = 1e-4
epochs = 5
batch_size = 4
```

**Eğitim İlerlemesi**:
```
Epoch | Train Loss | Eval Loss | Doğruluk
------|------------|-----------|----------
1     | 2.48       | 1.19      | 74.0%
2     | 0.78       | 0.42      | 87.0%
3     | 0.15       | 0.27      | 93.0%
```

**Sonuçlar**:
```
Config | Doğruluk | F1-Score | Eğitim Süresi | Karar
-------|----------|----------|---------------|----------
A      | 89.1%    | 0.891    | 15 dk        | ❌ Underfitted
B      | 93.0%    | 0.930    | 42 dk        | ✅ Seçildi
C      | 92.8%    | 0.928    | 90 dk        | ❌ Overfitted
```

## 📈 Kalibrasyon Deneyleri

### Deney 6: Kalibrasyon Yöntemi Karşılaştırması
**Tarih**: 2025-09-28  
**Amaç**: Kalibrasyon tekniklerini karşılaştırma

**Test Edilen Yöntemler**:
- No Calibration (baseline)
- Platt Scaling (logistic regression)
- Temperature Scaling (neural networks)
- Isotonic Regression

**Sonuçlar**:
```
Model       | Yöntem           | Brier Score | ECE    | Karar
------------|------------------|-------------|--------|----------
Baseline    | No Calibration   | 0.623       | 0.089  | ❌ Kötü
Baseline    | Platt Scaling    | 0.577       | 0.045  | ✅ Seçildi
Transformer | No Calibration   | 0.945       | 0.156  | ❌ Kötü
Transformer | Temperature      | 0.915       | 0.078  | ✅ Seçildi
```

### Deney 7: Confidence Threshold Optimizasyonu
**Tarih**: 2025-09-28  
**Amaç**: Optimal rejection threshold bulma

**Test Edilen Threshold'lar**: 0.5, 0.6, 0.7, 0.8, 0.9

**Sonuçlar**:
```
Threshold | Precision | Recall | Abstain Oranı | Karar
----------|-----------|--------|---------------|----------
0.5       | 0.78      | 0.95   | 5%            | ❌ Çok permissive
0.6       | 0.82      | 0.92   | 8%            | ❌ Hala düşük precision
0.7       | 0.87      | 0.88   | 12%           | ✅ Seçildi
0.8       | 0.92      | 0.81   | 19%           | ❌ Çok restrictive
0.9       | 0.96      | 0.72   | 28%           | ❌ Çok fazla abstention
```

## 🌍 Cross-lingual Deneyleri

### Deney 8: Dil Performans Analizi
**Tarih**: 2025-09-28  
**Amaç**: TR vs EN performansını değerlendirme

**Test Seti**: 20 örnek (12 Türkçe, 8 İngilizce)

**Sonuçlar**:
```
Model       | Dil      | Doğruluk | F1-Score | Notlar
------------|----------|----------|----------|-------
Baseline    | Türkçe   | 75.0%    | 0.67     | Düşük performans
Baseline    | İngilizce| 100.0%   | 1.00     | Mükemmel
Transformer | Türkçe   | 83.3%    | 0.78     | Baseline'dan daha iyi
Transformer | İngilizce| 100.0%   | 1.00     | Mükemmel
```

**Analiz**: İngilizce performans tutarlı olarak daha iyi çünkü:
1. DistilBERT'te daha fazla İngilizce training data
2. TF-IDF İngilizce word boundary'lerle daha iyi çalışır
3. Türkçe agglutinative doğası her iki modeli de zorlar

## 🔬 Ablation Studies

### Deney 9: Feature Ablation
**Tarih**: 2025-09-28  
**Amaç**: Feature önemini anlama

**Test Edilen Konfigürasyonlar**:
```
Konfigürasyon           | Doğruluk | F1-Score | Notlar
------------------------|----------|----------|-------
Full Model              | 86.56%   | 0.861    | Baseline
No Calibration          | 86.56%   | 0.861    | Fark yok
Limited Features (1K)   | 86.19%   | 0.849    | Hafif düşüş
Unigrams Only           | 85.45%   | 0.859    | Bigrams yardım eder
Trigrams                | 86.56%   | 0.872    | En iyi performans
```

**Sonuç**: Bigrams ve trigrams önemli değer sağlar.

### Deney 10: Veri Boyutu Ablation
**Tarih**: 2025-09-28  
**Amaç**: Veri gereksinimlerini anlama

**Sonuçlar**:
```
Veri Boyutu | Doğruluk | F1-Score | İyileşme
------------|----------|----------|----------
10%         | 65.01%   | 0.601    | Baseline
25%         | 75.69%   | 0.708    | +10.68%
50%         | 81.22%   | 0.842    | +5.53%
75%         | 84.90%   | 0.862    | +3.68%
100%        | 86.37%   | 0.854    | +1.47%
```

**Analiz**: %75'ten sonra diminishing returns, ama %100 hala faydalı.

## 🎯 Final Model Performansı

### Production Metrikleri
```
Model       | Doğruluk | F1-Score | Latency | Memory
------------|----------|----------|---------|--------
Baseline    | 86.5%    | 0.861    | 3ms     | 50MB
Transformer | 93.0%    | 0.930    | 75ms    | 500MB
```

### Hata Analizi
**Yaygın Yanlış Sınıflandırmalar**:
1. **Türkçe → İngilizce**: "Kütüphane saat kaçta açılıyor?" bazen İngilizce olarak sınıflandırılır
2. **Intent Karışıklığı**: `complaint` vs `other` (belirsiz sorgular)
3. **Düşük Confidence**: Karmaşık sorgular abstain'i doğru tetikler



## 🔄 Deney Reproducibility

Tüm deneyler şunlarla reproduce edilebilir:
- **Veri**: `scripts/generate_synthetic_data.py`
- **Eğitim**: `scripts/train_baseline.py`, `scripts/train_transformer.py`
- **Değerlendirme**: `scripts/generate_calibration_curves.py`, `scripts/generate_ablation_study.py`
- **Konfigürasyon**: Tüm hyperparametreler ilgili script'lerde dokümante edilmiş

**Random Seeds**: Reproducibility için sabit (seed=42)
**Environment**: Python 3.12, requirements.txt sağlanmış
