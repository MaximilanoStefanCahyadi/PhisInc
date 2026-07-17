"""
explainability.py — Lapisan penjelasan (XAI) untuk PhisInc.

Teknik utama: ATRIBUSI KONTRAFAKTUAL PER-FITUR ("flip attribution").
Untuk setiap fitur i (nilai di {-1, 0, 1}), nilai dipaksa menjadi +1
(indikasi aman) dan -1 (indikasi phishing), lalu selisih LOG-ODDS phishing
(decision_function, dinegasikan) diukur terhadap prediksi asli. Seluruh
60 varian + input asli dihitung dalam satu batch 61 baris (satu panggilan
decision_function, plus satu predict_proba untuk baris asli) -> ~1 ms.

Tidak ada dependensi baru: hanya numpy + pandas + scikit-learn yang sudah
terpasang.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# ------------------------------------------------------------------
# Kontrak model
# model.classes_ == array([-1, 1])  (label phishing.csv: -1 phishing, 1 legit)
#   kolom 0 predict_proba = P(phishing), kolom 1 = P(legitimate)
# ------------------------------------------------------------------
PHISH_COL = 0
LEGIT_COL = 1
N_FEATURES = 30

# Nama fitur persis seperti saat model dilatih (model.feature_names_in_),
# urutan sama dengan FeatureExtraction.getFeaturesList().
ENGLISH_NAMES = [
    "UsingIP", "LongURL", "ShortURL", "Symbol@", "Redirecting//",
    "PrefixSuffix-", "SubDomains", "HTTPS", "DomainRegLen", "Favicon",
    "NonStdPort", "HTTPSDomainURL", "RequestURL", "AnchorURL",
    "LinksInScriptTags", "ServerFormHandler", "InfoEmail", "AbnormalURL",
    "WebsiteForwarding", "StatusBarCust", "DisableRightClick",
    "UsingPopupWindow", "IframeRedirection", "AgeofDomain", "DNSRecording",
    "WebsiteTraffic", "PageRank", "GoogleIndex", "LinksPointingToPage",
    "StatsReport",
]


def check_model_contract(model) -> None:
    """Fail-fast bila urutan kelas / jumlah fitur tidak sesuai asumsi app."""
    classes = list(getattr(model, "classes_", []))
    if classes != [-1, 1]:
        raise ValueError(
            f"Urutan kelas model tidak sesuai asumsi [-1, 1]: {classes}. "
            "Indeks predict_proba di app.py dan explainability.py harus disesuaikan."
        )
    n_in = getattr(model, "n_features_in_", N_FEATURES)
    if n_in != N_FEATURES:
        raise ValueError(f"Model mengharapkan {n_in} fitur, bukan {N_FEATURES}.")


def as_model_frame(model, X: np.ndarray) -> "pd.DataFrame | np.ndarray":
    """Bungkus array menjadi DataFrame bernama kolom sesuai pelatihan model
    (menghilangkan peringatan sklearn 'X does not have valid feature names')."""
    names = getattr(model, "feature_names_in_", None)
    if names is not None:
        return pd.DataFrame(X, columns=list(names))
    return X


# ------------------------------------------------------------------
# Kelompok fitur (indeks 0-29, urutan persis getFeaturesList())
# ------------------------------------------------------------------
# (Indeks 0-7, 10, 11 murni dari string URL; 25-27, 29 dari layanan pihak
# ketiga — keduanya tidak butuh flag reliabilitas terpisah.)
PAGE_FEATURES = {9, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 28}    # butuh fetch HTML
WHOIS_FEATURES = {8, 17, 23, 24}                                   # butuh lookup WHOIS

# Fitur yang implementasi ekstraksinya diketahui rusak / konstan / terbalik.
# Selalu diturunkan prioritasnya dan diberi tanda "tidak terverifikasi".
BROKEN_FEATURES = {
    0,   # UsingIP: ipaddress.ip_address() dipanggil pada URL utuh -> selalu 1
    16,  # InfoEmail: regex character-class -> cocok di hampir semua HTML
    17,  # AbnormalURL: membandingkan response.text == whois_response
    19,  # StatusBarCust: logika terbalik terhadap dataset pelatihan
    20,  # DisableRightClick: logika terbalik
    21,  # UsingPopupWindow: logika terbalik
    22,  # IframeRedirection: regex character-class + logika terbalik
    25,  # WebsiteTraffic: data.alexa.com sudah dimatikan (2022) -> selalu -1
    26,  # PageRank: scraping checkpagerank.net hampir selalu gagal
    27,  # GoogleIndex: modul `googlesearch` tidak terpasang -> selalu 1
}

BROKEN_NOTES = {
    0: "Pemeriksaan IP pada implementasi saat ini tidak pernah aktif (bug ekstraksi).",
    16: "Deteksi fitur ini terlalu sensitif (hampir selalu menyala) — abaikan bila sendirian.",
    17: "Perbandingan WHOIS pada implementasi saat ini hampir selalu gagal cocok.",
    19: "Logika deteksi fitur ini terbalik pada implementasi saat ini.",
    20: "Logika deteksi fitur ini terbalik pada implementasi saat ini.",
    21: "Logika deteksi fitur ini terbalik pada implementasi saat ini.",
    22: "Logika deteksi fitur ini terbalik pada implementasi saat ini.",
    25: "Layanan Alexa sudah dihentikan — fitur ini selalu bernilai buruk untuk semua URL.",
    26: "Layanan pemeringkat eksternal hampir selalu gagal diakses.",
    27: "Modul pencarian Google tidak terpasang — fitur ini selalu bernilai aman.",
}


# ------------------------------------------------------------------
# Tabel pemetaan 30 fitur -> penjelasan bahasa Indonesia
# ------------------------------------------------------------------
FEATURE_INFO = [
    {  # 0
        "nama": "Alamat IP di URL",
        "arti": "Apakah URL memakai alamat IP mentah, bukan nama domain.",
        "alasan_phishing": "URL memakai alamat IP mentah — situs resmi hampir selalu memakai nama domain; IP dipakai untuk menghindari pelacakan.",
        "alasan_legit": "URL memakai nama domain, bukan alamat IP mentah.",
    },
    {  # 1
        "nama": "Panjang URL",
        "arti": "Panjang total URL (>= 75 karakter dianggap berbahaya, 54-75 mencurigakan).",
        "alasan_phishing": "URL sangat panjang (>= 75 karakter) — sering dipakai untuk menyembunyikan bagian mencurigakan dari pandangan pengguna.",
        "alasan_suspicious": "URL cukup panjang (54-75 karakter) — patut dicermati, meski belum tentu berbahaya.",
        "alasan_legit": "Panjang URL normal (< 54 karakter).",
    },
    {  # 2
        "nama": "Layanan Pemendek URL",
        "arti": "Apakah URL memakai layanan pemendek (bit.ly, tinyurl, dsb.).",
        "alasan_phishing": "URL memakai layanan pemendek yang menyembunyikan alamat tujuan sebenarnya.",
        "alasan_legit": "URL tidak memakai layanan pemendek.",
    },
    {  # 3
        "nama": "Simbol @",
        "arti": "Ada karakter '@' di dalam URL.",
        "alasan_phishing": "URL mengandung '@' — browser mengabaikan semua teks sebelum '@', sehingga alamat asli dapat disamarkan.",
        "alasan_legit": "Tidak ada karakter '@' di URL.",
    },
    {  # 4
        "nama": "Pengalihan '//'",
        "arti": "'//' muncul lagi di luar posisi protokol.",
        "alasan_phishing": "Ditemukan '//' di tengah URL — indikasi trik pengalihan (redirect) ke situs lain yang disisipkan dalam path.",
        "alasan_legit": "Tidak ada pola pengalihan '//' yang mencurigakan.",
    },
    {  # 5
        "nama": "Tanda '-' pada Domain",
        "arti": "Nama domain mengandung tanda hubung.",
        "alasan_phishing": "Domain mengandung tanda '-' — pola umum peniruan merek resmi (contoh: paypal-login.com).",
        "alasan_legit": "Domain tidak mengandung tanda hubung.",
    },
    {  # 6
        "nama": "Jumlah Subdomain",
        "arti": "Banyaknya titik/subdomain pada URL.",
        "alasan_phishing": "URL memiliki banyak subdomain berlapis — trik agar terlihat seperti domain resmi (contoh: bank.com.verifikasi.xyz).",
        "alasan_suspicious": "URL memiliki satu subdomain ekstra — perlu dicermati.",
        "alasan_legit": "Struktur domain sederhana tanpa subdomain berlebihan.",
    },
    {  # 7
        "nama": "HTTPS",
        "arti": "Apakah koneksi memakai skema https.",
        "alasan_phishing": "Situs tidak memakai HTTPS — data yang dikirim tidak terenkripsi.",
        "alasan_legit": "Situs memakai HTTPS. Catatan: banyak situs phishing kini juga ber-HTTPS, jadi ini bukan jaminan aman.",
    },
    {  # 8
        "nama": "Masa Registrasi Domain",
        "arti": "Sisa masa registrasi domain >= 1 tahun (dari WHOIS).",
        "alasan_phishing": "Domain terdaftar untuk jangka pendek — pelaku phishing biasanya mendaftar domain semurah/sesingkat mungkin.",
        "alasan_legit": "Domain terdaftar untuk jangka panjang (>= 1 tahun) — ciri pengelolaan situs yang serius.",
    },
    {  # 9
        "nama": "Favicon",
        "arti": "Favicon dimuat dari domain sendiri atau dari domain lain.",
        "alasan_phishing": "Favicon dimuat dari domain lain — indikasi halaman meniru identitas situs lain.",
        "alasan_legit": "Favicon dimuat dari domain sendiri.",
    },
    {  # 10
        "nama": "Port Non-Standar",
        "arti": "Ada nomor port selain standar pada bagian domain.",
        "alasan_phishing": "URL memakai port non-standar — sering dipakai server phishing untuk menghindari pemindaian.",
        "alasan_legit": "URL memakai port standar.",
    },
    {  # 11
        "nama": "'https' dalam Nama Domain",
        "arti": "Kata 'https' muncul di dalam nama domain itu sendiri.",
        "alasan_phishing": "Kata 'https' disisipkan dalam nama domain — trik visual agar URL terlihat aman (contoh: https-secure-login.com).",
        "alasan_legit": "Tidak ada penyalahgunaan kata 'https' dalam nama domain.",
    },
    {  # 12
        "nama": "Sumber Objek Eksternal",
        "arti": "Persentase gambar/media halaman yang dimuat dari domain lain.",
        "alasan_phishing": "Sebagian besar gambar/media dimuat dari domain lain — halaman phishing sering 'mencuri' aset dari situs asli yang ditirunya.",
        "alasan_suspicious": "Cukup banyak gambar/media dimuat dari domain lain.",
        "alasan_legit": "Mayoritas gambar/media dimuat dari domain sendiri.",
    },
    {  # 13
        "nama": "Tautan Anchor",
        "arti": "Persentase tautan <a> yang kosong, javascript:, atau keluar domain.",
        "alasan_phishing": "Mayoritas tautan pada halaman kosong atau mengarah keluar domain — ciri halaman tiruan.",
        "alasan_suspicious": "Cukup banyak tautan kosong atau keluar domain.",
        "alasan_legit": "Mayoritas tautan halaman mengarah ke domain sendiri.",
    },
    {  # 14
        "nama": "Tautan Script/Link",
        "arti": "Persentase tag <script>/<link> yang dimuat dari domain lain.",
        "alasan_phishing": "Mayoritas script/stylesheet dimuat dari domain lain — pola halaman tiruan.",
        "alasan_suspicious": "Sebagian script/stylesheet dimuat dari domain lain.",
        "alasan_legit": "Script/stylesheet umumnya dimuat dari domain sendiri.",
    },
    {  # 15
        "nama": "Penanganan Form (SFH)",
        "arti": "Ke mana data form dikirim (kosong, about:blank, atau domain lain).",
        "alasan_phishing": "Form mengirim data ke alamat kosong/about:blank — pola klasik pencurian kredensial.",
        "alasan_suspicious": "Form mengirim data ke domain lain — data Anda bisa jatuh ke pihak ketiga.",
        "alasan_legit": "Form (jika ada) mengirim data ke domain sendiri.",
    },
    {  # 16
        "nama": "Pengiriman via Email",
        "arti": "Halaman memakai mail()/mailto: untuk mengirim data.",
        "alasan_phishing": "Terindikasi pengiriman data via email (mailto:/mail()) — data korban dikirim langsung ke pelaku.",
        "alasan_legit": "Tidak terindikasi pengiriman data via email.",
    },
    {  # 17
        "nama": "URL Abnormal",
        "arti": "Kecocokan identitas URL dengan catatan WHOIS-nya.",
        "alasan_phishing": "Identitas URL tidak cocok dengan catatan WHOIS domain.",
        "alasan_legit": "Identitas URL konsisten dengan catatan WHOIS.",
    },
    {  # 18
        "nama": "Pengalihan Berantai",
        "arti": "Berapa kali permintaan dialihkan (redirect) sebelum halaman final.",
        "alasan_phishing": "URL melakukan banyak pengalihan berantai (> 4) — menyembunyikan tujuan akhir.",
        "alasan_suspicious": "URL melakukan beberapa pengalihan (2-4) — perlu dicermati.",
        "alasan_legit": "URL tidak melakukan pengalihan berlebihan.",
    },
    {  # 19
        "nama": "Manipulasi Status Bar",
        "arti": "Skrip onMouseOver yang mengubah alamat di status bar browser.",
        "alasan_phishing": "Terindikasi skrip yang memanipulasi status bar agar alamat asli tidak terlihat.",
        "alasan_legit": "Tidak terindikasi manipulasi status bar.",
    },
    {  # 20
        "nama": "Klik Kanan Dinonaktifkan",
        "arti": "Halaman memblokir klik kanan agar kode tidak bisa diperiksa.",
        "alasan_phishing": "Klik kanan dinonaktifkan — mencegah pengguna memeriksa kode sumber halaman.",
        "alasan_legit": "Klik kanan tidak diblokir.",
    },
    {  # 21
        "nama": "Jendela Popup",
        "arti": "Halaman memakai jendela popup (umumnya untuk meminta data).",
        "alasan_phishing": "Halaman memunculkan popup — teknik umum meminta data pribadi secara agresif.",
        "alasan_legit": "Tidak terindikasi popup yang meminta data.",
    },
    {  # 22
        "nama": "Iframe Tersembunyi",
        "arti": "Halaman lain disematkan lewat iframe tanpa terlihat.",
        "alasan_phishing": "Terindikasi iframe tersembunyi — halaman lain disematkan tanpa sepengetahuan pengguna.",
        "alasan_legit": "Tidak terindikasi iframe tersembunyi.",
    },
    {  # 23
        "nama": "Umur Domain",
        "arti": "Domain sudah berumur >= 6 bulan (dari WHOIS).",
        "alasan_phishing": "Domain masih sangat baru (< 6 bulan) — domain phishing biasanya baru dibuat dan berumur pendek.",
        "alasan_legit": "Domain sudah berumur cukup lama (>= 6 bulan).",
    },
    {  # 24
        "nama": "Catatan DNS/WHOIS",
        "arti": "Ada catatan DNS/WHOIS valid untuk domain ini.",
        "alasan_phishing": "Catatan DNS/WHOIS domain tidak ditemukan atau sangat baru — jejak digital minim.",
        "alasan_legit": "Catatan DNS/WHOIS domain ditemukan dan cukup lama.",
    },
    {  # 25
        "nama": "Trafik Situs",
        "arti": "Peringkat trafik situs (basis data Alexa).",
        "alasan_phishing": "Situs tidak memiliki catatan trafik yang berarti — situs phishing berumur pendek sehingga trafiknya tak tercatat.",
        "alasan_suspicious": "Trafik situs tercatat namun peringkatnya rendah.",
        "alasan_legit": "Situs memiliki peringkat trafik tinggi.",
    },
    {  # 26
        "nama": "PageRank",
        "arti": "Peringkat global (reputasi) domain.",
        "alasan_phishing": "Reputasi/peringkat global domain rendah atau tidak ditemukan.",
        "alasan_legit": "Domain memiliki reputasi/peringkat global yang baik.",
    },
    {  # 27
        "nama": "Terindeks Google",
        "arti": "Apakah situs muncul di indeks pencarian Google.",
        "alasan_phishing": "Situs tidak ditemukan di indeks Google — situs phishing sering terlalu baru/tersembunyi untuk terindeks.",
        "alasan_legit": "Situs terindeks oleh Google.",
    },
    {  # 28
        "nama": "Tautan Masuk",
        "arti": "Banyaknya tautan yang menunjuk ke halaman ini.",
        "alasan_phishing": "Hampir tidak ada tautan yang menunjuk ke halaman ini — situs resmi yang mapan biasanya banyak dirujuk.",
        "alasan_suspicious": "Hanya sedikit tautan yang menunjuk ke halaman ini.",
        "alasan_legit": "Halaman memiliki cukup banyak tautan masuk.",
    },
    {  # 29
        "nama": "Laporan Statistik (Daftar Hitam)",
        "arti": "Domain/alamat IP cocok dengan daftar host phishing yang dikenal.",
        "alasan_phishing": "Domain atau alamat IP situs ini cocok dengan daftar hitam phishing yang dikenal.",
        "alasan_legit": "Domain/IP tidak ada dalam daftar hitam phishing yang dikenal.",
    },
]


# ------------------------------------------------------------------
# 1) TEKNIK UTAMA: atribusi kontrafaktual (flip attribution)
# ------------------------------------------------------------------
def flip_attribution(model, features, phish_col: int = PHISH_COL):
    """
    Hitung kontribusi lokal per fitur dengan kontrafaktual satu-fitur.

    Efek diukur pada skala LOG-ODDS (decision_function), bukan probabilitas:
    pada prediksi ekstrem (P mendekati 0 atau 1) probabilitas sudah jenuh
    sehingga semua selisih ~0, sedangkan log-odds tetap membedakan fitur
    mana yang paling berpengaruh.

    Returns:
        p0      : float           P(phishing) untuk input asli.
        risk    : ndarray (30,)   penurunan log-odds phishing bila fitur i
                                  dipaksa bersih (+1). Positif = fitur ini
                                  sedang menaikkan skor phishing.
        protect : ndarray (30,)   kenaikan log-odds phishing bila fitur i
                                  dipaksa buruk (-1). Positif = fitur ini
                                  sedang menahan skor phishing.
    """
    x = np.asarray(list(features), dtype=np.float64).reshape(1, -1)
    n = x.shape[1]
    X = np.repeat(x, 2 * n + 1, axis=0)
    idx = np.arange(n)
    X[1 + idx, idx] = 1.0        # baris 1..n     : fitur i dipaksa bersih (+1)
    X[1 + n + idx, idx] = -1.0   # baris n+1..2n  : fitur i dipaksa buruk (-1)

    Xf = as_model_frame(model, X)
    p0 = float(model.predict_proba(Xf[:1])[0, phish_col])

    # decision_function = log-odds kelas legit (+1); negasikan agar menjadi
    # log-odds phishing.
    margin = -np.asarray(model.decision_function(Xf), dtype=np.float64)
    m0 = margin[0]
    risk = m0 - margin[1:1 + n]
    protect = margin[1 + n:2 * n + 1] - m0
    return p0, risk, protect


# ------------------------------------------------------------------
# 2) TEKNIK SEKUNDER (cross-check offline): dekomposisi Saabas
# ------------------------------------------------------------------
def saabas_contributions(model, features):
    """
    Dekomposisi aditif eksak dari decision_function(x) per fitur untuk
    GradientBoostingClassifier biner: baseline + contrib.sum() ==
    decision_function(x). contrib[i] < 0 -> mendorong ke PHISHING.
    Nilai akar tiap pohon (bias per-stage) dimasukkan ke baseline.
    Untuk validasi offline flip_attribution, bukan untuk tampilan.
    """
    x = np.asarray(list(features), dtype=np.float32).reshape(1, -1)
    contrib = np.zeros(x.shape[1], dtype=np.float64)

    prior = float(model.init_.class_prior_[LEGIT_COL])
    baseline = np.log(prior / (1.0 - prior))

    lr = model.learning_rate
    for stage in model.estimators_[:, 0]:
        t = stage.tree_
        baseline += lr * float(t.value[0, 0, 0])
        node = 0
        while t.children_left[node] != -1:
            f = t.feature[node]
            nxt = (t.children_left[node]
                   if x[0, f] <= t.threshold[node]
                   else t.children_right[node])
            contrib[f] += lr * (t.value[nxt, 0, 0] - t.value[node, 0, 0])
            node = nxt
    return baseline, contrib


# ------------------------------------------------------------------
# Reliabilitas fitur (deteksi kegagalan fetch/WHOIS)
# ------------------------------------------------------------------
def extraction_status(fe):
    """fe.response / fe.whois_response tetap string "" bila pengambilan gagal.
    Fetch juga dianggap gagal bila server merespons dengan status error
    (>= 400): fitur konten dari halaman 404/500 bukan bukti nyata.
    fe=None (pemanggil tidak menyertakan objek ekstraksi) dianggap berhasil —
    selalu teruskan `fe` dari app agar demosi reliabilitas berfungsi."""
    if fe is None:
        return True, True
    response = getattr(fe, "response", "")
    fetch_ok = (not isinstance(response, str)) and bool(getattr(response, "ok", False))
    w = getattr(fe, "whois_response", "")
    whois_ok = not isinstance(w, str)
    if whois_ok:
        try:
            whois_ok = w.creation_date is not None
        except Exception:
            whois_ok = False
    return fetch_ok, whois_ok


def feature_reliable(i: int, fetch_ok: bool, whois_ok: bool) -> bool:
    if i in BROKEN_FEATURES:
        return False
    if i in PAGE_FEATURES and not fetch_ok:
        return False
    if i in WHOIS_FEATURES and not whois_ok:
        return False
    return True


def reliability_note(i: int, fetch_ok: bool, whois_ok: bool) -> str:
    if i in BROKEN_FEATURES:
        return BROKEN_NOTES.get(i, "Fitur ini tidak dapat diverifikasi.")
    if i in PAGE_FEATURES and not fetch_ok:
        return "Halaman gagal diambil — nilai fitur ini adalah nilai default, bukan hasil pemeriksaan nyata."
    if i in WHOIS_FEATURES and not whois_ok:
        return "Data WHOIS tidak tersedia — nilai fitur ini adalah nilai default, bukan hasil pemeriksaan nyata."
    return ""


# ------------------------------------------------------------------
# 3) Vonis berjenjang (verdict banding)
# ------------------------------------------------------------------
def verdict_band(p_phish: float) -> dict:
    """Band berdasarkan P(phishing). Ambang konservatif; kalibrasi ulang
    dengan holdout set sebelum mengubah ambang."""
    if p_phish >= 0.85:
        return {
            "kode": "high-risk", "label": "TERINDIKASI PHISHING",
            "pesan": "Jangan buka tautan ini. Jangan masukkan kata sandi, OTP, atau data pribadi apa pun.",
        }
    if p_phish >= 0.60:
        return {
            "kode": "suspicious", "label": "MENCURIGAKAN",
            "pesan": "Terdapat beberapa indikasi phishing. Hindari memasukkan kredensial atau data pribadi.",
        }
    if p_phish > 0.40:
        return {
            "kode": "uncertain", "label": "TIDAK PASTI",
            "pesan": "Model tidak dapat memutuskan dengan yakin. Periksa ejaan domain dan akses situs resmi "
                     "lewat bookmark atau pencarian, bukan lewat tautan ini.",
        }
    if p_phish > 0.15:
        return {
            "kode": "likely-safe", "label": "KEMUNGKINAN AMAN",
            "pesan": "Mayoritas indikator terlihat wajar, namun tetap periksa alamat domain sebelum login.",
        }
    return {
        "kode": "safe", "label": "SANGAT MUNGKIN AMAN",
        "pesan": "Hampir semua indikator terlihat wajar. Tidak ada deteksi yang 100% pasti — tetap waspada.",
    }


# ------------------------------------------------------------------
# 4) Perakit penjelasan
# ------------------------------------------------------------------
def build_explanation(model, features, fe=None, top_n: int = 3,
                      min_effect: float = 0.05) -> dict:
    """
    Rakit penjelasan "MENGAPA" untuk satu prediksi.

    Returns dict berisi p_phish, p_legit, band, risk_factors, trust_factors
    (terurut: fitur andal dulu, lalu efek terbesar), fetch_ok, whois_ok,
    n_unreliable_bad. `min_effect` dalam satuan log-odds (bukan probabilitas).
    """
    features = list(features)
    p0, risk, protect = flip_attribution(model, features)
    fetch_ok, whois_ok = extraction_status(fe)

    risk_factors, trust_factors = [], []
    for i, v in enumerate(features):
        info = FEATURE_INFO[i]
        andal = feature_reliable(i, fetch_ok, whois_ok)
        catatan = reliability_note(i, fetch_ok, whois_ok)

        if v <= 0:  # kandidat faktor risiko (nilai -1 atau 0)
            eff = float(risk[i])
            if eff >= min_effect:
                alasan = (info.get("alasan_suspicious", info["alasan_phishing"])
                          if v == 0 else info["alasan_phishing"])
                risk_factors.append({
                    "index": i, "nama": info["nama"], "nilai": int(v),
                    "efek": round(eff, 2),  # satuan log-odds
                    "alasan": alasan, "andal": andal, "catatan": catatan,
                })
        else:  # v == 1 -> kandidat faktor kepercayaan
            eff = float(protect[i])
            if eff >= min_effect:
                trust_factors.append({
                    "index": i, "nama": info["nama"], "nilai": 1,
                    "efek": round(eff, 2),  # satuan log-odds
                    "alasan": info["alasan_legit"], "andal": andal, "catatan": catatan,
                })

    # Fitur andal didahulukan; dalam kelompok yang sama urutkan efek menurun.
    sort_key = lambda d: (not d["andal"], -d["efek"])
    risk_factors.sort(key=sort_key)
    trust_factors.sort(key=sort_key)

    n_unreliable_bad = sum(
        1 for i, v in enumerate(features)
        if v == -1 and not feature_reliable(i, fetch_ok, whois_ok)
    )

    return {
        "p_phish": p0,
        "p_legit": 1.0 - p0,
        "band": verdict_band(p0),
        "risk_factors": risk_factors[:top_n],
        "trust_factors": trust_factors[:top_n],
        "fetch_ok": fetch_ok,
        "whois_ok": whois_ok,
        "n_unreliable_bad": n_unreliable_bad,
    }
