# 🛡️ PhisInc — Phishing URL Detection System

**PhisInc** adalah aplikasi web interaktif berbasis Streamlit yang dirancang untuk mendeteksi situs web penipuan (phishing) menggunakan Machine Learning. Aplikasi ini menganalisis URL yang dimasukkan pengguna secara real-time berdasarkan 30 indikator fitur keamanan, memberikan klasifikasi instan (Legitimate/Phishing), dan memvisualisasikan tingkat kepercayaan prediksi.

Aplikasi ini hadir dengan antarmuka pengguna **Dark Glassmorphic UI** yang premium, modern, dan sepenuhnya responsif.

---

## 🚀 Fitur Utama

- **Analisis URL Real-Time**: Cukup masukkan URL dan aplikasi akan langsung mengekstrak indikator keamanan serta memprosesnya ke model machine learning.
- **Model Machine Learning Akurat**: Didukung oleh algoritma **Gradient Boosting Classifier** untuk prediksi dengan akurasi dan presisi tinggi.
- **Ekstraksi 30 Fitur URL**: Memeriksa aspek-aspek krusial dari URL yang dibagi dalam 4 kategori (Address Bar, Fitur Abnormal, HTML & JavaScript, dan Reputasi Domain).
- **Desain Premium & Modern**: Tampilan bertema gelap (*dark mode*) yang memukau dengan efek glassmorphism, animasi interaktif, indikator probabilitas visual, dan detail fitur yang dikelompokkan dengan rapi.
- **Bilingual Interface**: Antarmuka ramah pengguna disajikan dalam Bahasa Indonesia dengan istilah industri standar.

---

## 🛠️ Tech Stack & Dependensi

Proyek ini dibangun menggunakan pustaka Python berikut:

- **Frontend & Web Framework**: [Streamlit](https://streamlit.io/) (v1.28.0)
- **Machine Learning**: [Scikit-Learn](https://scikit-learn.org/) (v1.3.2), [Joblib](https://joblib.readthedocs.io/)
- **Pemrosesan Data**: [NumPy](https://numpy.org/), [Pandas](https://pandas.pydata.org/)
- **Ekstraksi & Scraping URL**: 
  - [BeautifulSoup4](https://www.crummy.com/software/BeautifulSoup/) (Analisis HTML)
  - [Requests](https://requests.readthedocs.io/) (HTTP Client)
  - [python-whois](https://pypi.org/project/python-whois/) (Informasi pendaftaran domain)
  - [lxml](https://lxml.de/) (Parser HTML)

---

## 📁 Struktur Proyek

```directory
PhisInc/
├── app.py                      # Berisi antarmuka Streamlit (UI) dan logika klasifikasi utama
├── featureextraction.py        # Kelas ekstraktor untuk mengambil 30 fitur keamanan dari sebuah URL
├── gradient_boosting_model.pkl # Model Gradient Boosting Classifier yang telah dilatih
├── phishing.csv                # Dataset latih yang digunakan untuk model
├── requirements.txt            # Daftar pustaka Python yang diperlukan
├── runtime.txt                 # Konfigurasi runtime Python untuk deploy
└── README.md                   # Dokumentasi proyek (file ini)
```

---

## 🔬 30 Fitur URL yang Dianalisis

Aplikasi mengekstrak dan menganalisis 30 parameter keamanan dari URL, diklasifikasikan ke dalam:

### 1. Fitur Berbasis Address Bar (Bilah Alamat)
- **Using IP Address**: Mendeteksi jika domain menggunakan alamat IP secara langsung.
- **Long URL**: Panjang URL (semakin panjang berisiko tinggi).
- **Short URL**: Penggunaan layanan pemendek URL (seperti bit.ly, goo.gl).
- **Symbol @**: Penggunaan simbol `@` yang dapat mengalihkan fokus pengguna.
- **Redirecting //**: Adanya redirect di dalam URL.
- **Prefix Suffix**: Penggunaan tanda hubung (`-`) pada nama domain.
- **Sub Domains**: Jumlah subdomain (semakin banyak berisiko).
- **HTTPS**: Kehadiran protokol SSL/TLS yang valid.
- **Domain Registration Length**: Masa berlaku pendaftaran domain.
- **Favicon**: Apakah favicon dimuat dari domain luar.

### 2. Fitur Abnormal URL
- **Non-Standard Port**: Penggunaan port tidak umum (selain 80/443).
- **HTTPS in Domain**: Kata "HTTPS" disematkan di dalam nama domain untuk mengelabui.
- **Request URL**: Persentase resource eksternal yang diminta.
- **Anchor URL**: Persentase link `<a href>` yang merujuk ke luar domain.
- **Links in Script Tags**: Tautan eksternal di dalam tag `<script>`, `<link>`, `<meta>`.
- **Server Form Handler (SFH)**: Alamat aksi form (`action`) yang tidak sah/kosong.
- **Info Email**: Adanya fungsi pengiriman email input formulir langsung dari client.
- **Abnormal URL**: Kesesuaian host WHOIS dengan URL.
- **Website Forwarding**: Jumlah pengalihan URL.
- **Status Bar Customization**: Manipulasi status bar menggunakan JavaScript.

### 3. Fitur HTML & JavaScript
- **Disable Right Click**: Pemblokiran klik kanan untuk menyembunyikan source code.
- **Using Popup Window**: Penggunaan window pop-up untuk meminta data sensitif.
- **Iframe Redirection**: Penggunaan tag `<iframe>` tersembunyi.

### 4. Fitur Reputasi Domain
- **Age of Domain**: Usia domain sejak pertama kali didaftarkan.
- **DNS Recording**: Ketersediaan catatan DNS yang sah.
- **Website Traffic**: Peringkat lalu lintas situs (Alex/SimilarWeb).
- **PageRank**: Skor otoritas halaman web.
- **Google Index**: Indeksasi halaman di mesin pencari Google.
- **Links Pointing to Page**: Jumlah tautan balik yang merujuk ke situs.
- **Stats Report**: Kehadiran domain di basis data ancaman/phishing umum.

---

## ⚙️ Cara Menjalankan Secara Lokal

Ikuti langkah-langkah berikut untuk menjalankan aplikasi di komputer Anda:

### 1. Prasyarat
Pastikan Anda telah menginstal **Python 3.10** ke atas di sistem Anda.

### 2. Kloning Repositori
```bash
git clone <url-repositori-anda>
cd PhisInc
```

### 3. Buat dan Aktifkan Virtual Environment
Sangat disarankan menggunakan virtual environment bawaan agar pustaka tidak konflik dengan sistem global Anda:

**Windows (PowerShell/CMD):**
```powershell
python -m venv .venv
.venv\Scripts\activate
```

**macOS / Linux:**
```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 4. Instal Dependensi
Instal semua pustaka yang tertera di dalam `requirements.txt`:
```bash
pip install -r requirements.txt
```

### 5. Jalankan Aplikasi Streamlit
Jalankan server Streamlit lokal:
```bash
streamlit run app.py
```
Setelah dijalankan, buka browser Anda di alamat default: **`http://localhost:8501`**.

---

## 🛡️ Panduan Penggunaan Aman
*Sistem ini dirancang untuk membantu mengidentifikasi URL yang mencurigakan menggunakan karakteristik struktural dan reputasi domain. Harap selalu waspada dan jangan pernah memasukkan kredensial, kata sandi, atau data kartu kredit pada situs web yang tingkat kepercayaan Legitimasinya meragukan.*
