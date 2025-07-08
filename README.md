# 🛡️ KomentarBersih

Aplikasi Streamlit untuk mengidentifikasi komentar judi online menggunakan model IndoBERT dari Hugging Face.

## 🚀 Fitur

- **Input Single Komentar**: Analisis satu komentar secara real-time
- **Input File CSV**: Analisis batch komentar dari file CSV
- **Input URL YouTube**: Scraping dan analisis komentar dari video YouTube
- **Model IndoBERT**: Menggunakan model pre-trained untuk klasifikasi komentar judi
- **Preprocessing**: Cleaning teks sebelum analisis
- **Threshold Filtering**: Filter hasil berdasarkan confidence score
- **Export Results**: Download hasil analisis dalam format CSV
- **Visualisasi**: Pie chart & bar chart hasil prediksi
- **Logging**: Semua aktivitas tercatat di `app.log`

## 📋 Prerequisites

- Python 3.10+
- YouTube Data API v3 (untuk fitur YouTube scraping)
- Koneksi internet (untuk download model Hugging Face)

## 🛠️ Installation

1. **Clone repository**

```bash
git clone https://github.com/fhru/komentarbersih-dashboard.git
cd komentarbersih-dashboard
```

2. **Install dependencies**

```bash
pip install -r requirements.txt
```

3. **Setup .env untuk API Key**

Buat file `.env` di root folder:

```
YOUTUBE_API_KEY=your_api_key_here
```

Atau set environment variable secara manual:

```bash
# Windows
set YOUTUBE_API_KEY=your_api_key_here

# Linux/Mac
export YOUTUBE_API_KEY=your_api_key_here
```

## 🎯 Cara Menjalankan

1. **Jalankan aplikasi**

```bash
streamlit run app.py
```

2. **Buka browser**

```
http://localhost:8501
```

## 📊 Penggunaan

### 1. Input Single Komentar

- Masukkan komentar yang ingin dianalisis
- Klik "Analisis Komentar"
- Lihat hasil prediksi dengan confidence score

### 2. Input File CSV

- Upload file CSV dengan kolom 'komentar' atau 'comment'
- Set jumlah komentar yang dianalisis
- Download hasil dalam format CSV

### 3. Input URL YouTube

- Masukkan URL video YouTube
- Set jumlah komentar maksimal dan threshold
- Analisis komentar dari video tersebut

## 🔧 Model Loading

Model IndoBERT akan **otomatis dimuat saat aplikasi pertama kali dijalankan** menggunakan caching Streamlit. Ini memastikan:

- Model hanya dimuat sekali per sesi
- Performa yang lebih cepat untuk prediksi berikutnya
- Penggunaan memori yang efisien

## 📁 Struktur Proyek

```
streamlit_v4/
├── app.py                 # Aplikasi utama Streamlit
├── requirements.txt       # Dependencies
├── README.md              # Dokumentasi
├── .gitignore             # Git ignore file
├── app.log                # Log file aplikasi
└── utils/
    ├── cleaning.py        # Preprocessing teks
    ├── predictor.py       # Model prediction
    └── scraper.py         # YouTube scraping
```

## 🧪 Testing

Setiap modul dapat diuji secara independen:

```bash
# Test cleaning
python utils/cleaning.py

# Test predictor
python utils/predictor.py

# Test scraper
python utils/scraper.py
```

## 📝 Logging

Aplikasi menggunakan logging untuk tracking:

- File: `app.log`
- Level: INFO
- Format: timestamp - level - message

## 🔍 Model Details

- **Model**: `fhru/indobert-komentarbersih`
- **Task**: Text Classification
- **Labels**:
  - 0: Komentar Normal
  - 1: Komentar Judi

## ⚙️ Configuration

### Environment Variables

- `YOUTUBE_API_KEY`: API key untuk YouTube Data API v3

### Model Settings

- Threshold confidence dapat diatur di UI
- Batch size dapat disesuaikan di kode
- Model caching menggunakan `@st.cache_resource`

## 📈 Performance Tips

1. **GPU Usage**: Gunakan GPU untuk akselerasi model
2. **Batch Processing**: Analisis batch untuk efisiensi
3. **Caching**: Model di-cache untuk performa optimal
4. **Threshold**: Gunakan threshold untuk filter hasil
