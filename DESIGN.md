# DESIGN.md - CASE 3 Intent Classifier Mimarisi

## Sistem Mimarisi

### Genel Bakış
Bu proje, kütüphane müşteri hizmetleri sorguları için çok dilli (Türkçe/İngilizce) intent classification sistemi implement eder. Sistem, synthetic data generation, dual model architecture ve comprehensive evaluation kullanarak CASE 3 gereksinimlerini karşılar.

### Temel Bileşenler

#### 1. Veri Pipeline'ı
```
LLM (Gemini) → Synthetic Data → Deduplication → Clean Dataset
```

**Tasarım Kararları:**
- **LLM Seçimi**: Google Gemini 2.0 Flash Experimental (2025-09-24)
  - **Trade-off**: Maliyet vs Kalite - Gemini makul maliyetle iyi çok dilli destek sağlar
  - **Alternatif**: OpenAI GPT-4 (daha yüksek maliyet, benzer kalite)
- **Veri Hacmi**: Intent başına 120 örnek (deduplication sonrası 100+ hedefi) (2025-09-25)
  - **Trade-off**: Daha fazla veri model performansını artırır ama generation maliyetini artırır
- **Deduplication**: Embedding-based similarity + DBSCAN clustering (2025-09-26)
  - **Trade-off**: Hesaplama maliyeti vs veri kalitesi - temiz training data sağlar

#### 2. Model Mimarisi

**Dual Model Yaklaşımı:** (2025-09-27 - 2025-09-28)
```
Baseline Model: TF-IDF + Logistic Regression + Platt Scaling
Transformer Model: DistilBERT + Fine-tuning + Temperature Scaling
```

**Tasarım Gerekçesi:**
- **Baseline Model**: 
  - **Artıları**: Hızlı inference, interpretable, düşük kaynak gereksinimi
  - **Eksileri**: Sınırlı semantic understanding, TF-IDF limitasyonları
  - **Kullanım**: Production serving, hızlı yanıtlar
- **Transformer Model**:
  - **Artıları**: Üstün semantic understanding, daha iyi çok dilli performans
  - **Eksileri**: Daha yüksek hesaplama maliyeti, daha büyük memory footprint
  - **Kullanım**: Yüksek doğruluk senaryoları, karmaşık sorgular

#### 3. Kalibrasyon Stratejisi

**Platt Scaling (Baseline)**:
- **Neden**: Logistic regression çıktıları doğal olarak iyi kalibre edilmiş, Platt scaling fine-tuning sağlar
- **Implementasyon**: Cross-validation on held-out set

**Temperature Scaling (Transformer)**:
- **Neden**: Neural network'ler overconfident olma eğilimindedir, temperature scaling transformer'lar için etkilidir
- **Implementasyon**: Single parameter optimization on validation set

#### 4. Rejection Mekanizması

**Confidence Threshold**: 0.7
- **Tasarım Kararı**: Precision ve recall arasında denge
- **Trade-off**: Daha yüksek threshold = daha fazla abstention ama daha yüksek precision
- **Alternatif**: Intent class'a göre dynamic threshold

#### 5. Serving Mimarisi

**FastAPI Servisi**:
```
Client → FastAPI → Model Selection → Prediction → Response
```

**API Tasarımı**:
- **Single Prediction**: `/intent` endpoint
- **Batch Prediction**: `/intent/batch` endpoint
- **Health Check**: `/health` endpoint
- **Model Info**: `/intents` endpoint

**Trade-off'lar**:
- **Synchronous**: Basit implementasyon, ama batch request'lerde bloklar
- **Alternatif**: Async processing with job queues (daha karmaşık, daha iyi scalability)

## Teknik Kararlar

### 1. Model Seçim Kriterleri

**Baseline Model**:
- **TF-IDF**: Text classification için endüstri standardı
- **Logistic Regression**: Hızlı, interpretable, iyi baseline
- **Max Features**: 10,000 (performans ve memory arasında denge)

**Transformer Model**:
- **DistilBERT**: BERT'ten daha küçük, daha hızlı inference, iyi performans
- **Multilingual**: `distilbert-base-multilingual-cased`
- **Fine-tuning**: 3 epochs, learning rate 5e-5

### 2. Veri İşleme Pipeline'ı

**Tokenization**:
- **Baseline**: 1-2 n-grams ile TF-IDF
- **Transformer**: DistilBERT tokenizer, max length 128

**Preprocessing**:
- **Text Cleaning**: Minimal (orijinal dil pattern'lerini korur)
- **Encoding**: Pipeline boyunca UTF-8

### 3. Değerlendirme Stratejisi

**Metrikler**:
- **Primary**: F1-Score (macro average)
- **Secondary**: Accuracy, Precision, Recall
- **Calibration**: Brier Score, Calibration Curves

**Test Setleri**:
- **Synthetic**: Generated data'nın %20'si
- **Real-world**: 20 manuel oluşturulmuş örnek
- **Cross-lingual**: Ayrı Türkçe/İngilizce değerlendirme

## Performans Karakteristikleri

### Latency
- **Baseline Model**: ~2-5ms per prediction
- **Transformer Model**: ~50-100ms per prediction

### Memory Usage
- **Baseline Model**: ~50MB
- **Transformer Model**: ~500MB

### Accuracy
- **Baseline Model**: 85-86% accuracy
- **Transformer Model**: 90-93% accuracy

## Scalability Considerations

### Horizontal Scaling
- **Stateless Design**: Modeller instance'lar arasında replicate edilebilir
- **Load Balancing**: FastAPI multiple worker'ları destekler

### Vertical Scaling
- **Memory**: Transformer model yeterli RAM gerektirir
- **CPU**: Her iki model de CPU-optimized (GPU dependency yok)

### Caching Strategy
- **Model Loading**: Modeller startup'ta bir kez yüklenir
- **Prediction Caching**: Implement edilmedi (stateless design)

## Güvenlik & Güvenilirlik

### Input Validation
- **Text Length**: 1-1000 karakter
- **Encoding**: UTF-8 validation
- **Rate Limiting**: Implement edilmedi (eklenebilir)

### Error Handling
- **Graceful Degradation**: Transformer fail olursa baseline'a fallback
- **Logging**: Debug için comprehensive logging

### Data Privacy
- **No Persistence**: Prediction'lar saklanmaz
- **API Keys**: Environment variable management

## Monitoring & Observability

### Logging
- **Structured Logging**: Kolay parsing için JSON format
- **Log Levels**: INFO, WARNING, ERROR
- **Metrics**: Prediction latency, accuracy tracking

### Health Checks
- **Model Status**: Modellerin yüklendiğini doğrula
- **Memory Usage**: Kaynak tüketimini izle
- **API Endpoints**: Temel bağlantı testleri

