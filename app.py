import streamlit as st
import pickle
import numpy as np
from featureextraction import FeatureExtraction
import time
import os
import joblib

# ───────────────────────────────────────────────────────────────
# Page Configuration
# ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="PhisInc — Phishing URL Detection",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ───────────────────────────────────────────────────────────────
# Load Model (defined before CSS to avoid tokenizer issues)
# ───────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    try:
        model = joblib.load("gradient_boosting_model.pkl")
        return model
    except FileNotFoundError:
        st.error("❌ Model file 'gradient_boosting_model.pkl' tidak ditemukan!")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        return None


# ───────────────────────────────────────────────────────────────
# Design System — Global CSS
# ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ── Google Fonts ── */
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root Variables ── */
:root {
    --primary: #6C63FF;
    --primary-light: #8B83FF;
    --primary-dark: #4F46E5;
    --accent-safe: #00D97E;
    --accent-safe-bg: rgba(0,217,126,0.08);
    --accent-danger: #FF4D6A;
    --accent-danger-bg: rgba(255,77,106,0.08);
    --accent-warning: #FFB347;
    --accent-warning-bg: rgba(255,179,71,0.08);
    --bg-primary: #0E1117;
    --bg-secondary: #1A1D23;
    --surface: rgba(255,255,255,0.03);
    --surface-hover: rgba(255,255,255,0.06);
    --text-primary: #F0F2F6;
    --text-secondary: #9CA3AF;
    --text-muted: #6B7280;
    --border: rgba(108,99,255,0.15);
    --border-light: rgba(255,255,255,0.06);
    --radius-lg: 16px;
    --radius-md: 12px;
    --radius-sm: 8px;
    --radius-xs: 6px;
    --shadow-glow: 0 0 30px rgba(108,99,255,0.15);
    --shadow-card: 0 4px 24px rgba(0,0,0,0.3);
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

/* ── Global Reset ── */
html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

/* ── Hide Streamlit Defaults ── */
#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}

/* ── Custom Scrollbar ── */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--primary-dark); border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: var(--primary); }

/* ── Main Content Area ── */
.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1100px !important;
}

/* ── Hero Section ── */
.hero-container {
    text-align: center;
    padding: 3rem 1rem 2rem;
    margin-bottom: 1.5rem;
}

.hero-icon {
    font-size: 3.5rem;
    display: inline-block;
    animation: pulse-shield 2.5s ease-in-out infinite;
    margin-bottom: 0.5rem;
}

@keyframes pulse-shield {
    0%, 100% { transform: scale(1); filter: drop-shadow(0 0 8px rgba(108,99,255,0.3)); }
    50% { transform: scale(1.08); filter: drop-shadow(0 0 20px rgba(108,99,255,0.6)); }
}

.hero-title {
    font-size: 2.8rem;
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #6C63FF 0%, #8B83FF 40%, #00D97E 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0.3rem 0;
    line-height: 1.2;
}

.hero-subtitle {
    font-size: 1.1rem;
    color: var(--text-secondary);
    font-weight: 400;
    margin-top: 0.5rem;
    line-height: 1.6;
}

.hero-subtitle strong {
    color: var(--primary-light);
    font-weight: 600;
}

/* ── Glass Card ── */
.glass-card {
    background: var(--surface);
    backdrop-filter: blur(20px);
    -webkit-backdrop-filter: blur(20px);
    border: 1px solid var(--border);
    border-radius: var(--radius-lg);
    padding: 2rem;
    margin-bottom: 1.5rem;
    box-shadow: var(--shadow-card);
    transition: var(--transition);
}

.glass-card:hover {
    border-color: rgba(108,99,255,0.3);
    box-shadow: var(--shadow-glow), var(--shadow-card);
}

.glass-card-compact {
    background: var(--surface);
    backdrop-filter: blur(20px);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-md);
    padding: 1.25rem;
    margin-bottom: 0.75rem;
    transition: var(--transition);
}

.glass-card-compact:hover {
    background: var(--surface-hover);
    border-color: var(--border);
}

/* ── Section Label ── */
.section-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--primary-light);
    margin-bottom: 1rem;
}

.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

/* ── Input Styling ── */
.stTextInput > div > div > input,
.stTextInput input,
input[type="text"] {
    background: rgba(255,255,255,0.08) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    caret-color: #FFFFFF !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 1rem !important;
    padding: 0.75rem 1rem !important;
    transition: var(--transition) !important;
}

.stTextInput > div > div > input:focus,
.stTextInput input:focus,
input[type="text"]:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,0.2) !important;
    background: rgba(255,255,255,0.1) !important;
}

.stTextInput > div > div > input::placeholder,
.stTextInput input::placeholder,
input[type="text"]::placeholder {
    color: #9CA3AF !important;
    -webkit-text-fill-color: #9CA3AF !important;
    font-style: italic;
    opacity: 1 !important;
}

/* ── Button Styling ── */
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(135deg, var(--primary) 0%, var(--primary-dark) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 2rem !important;
    letter-spacing: 0.02em;
    transition: var(--transition) !important;
    width: 100%;
    margin-top: 0px;
    box-shadow: 0 4px 15px rgba(108,99,255,0.3) !important;
}

.stButton > button:hover, .stFormSubmitButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(108,99,255,0.45) !important;
}

.stButton > button:active, .stFormSubmitButton > button:active {
    transform: translateY(0) !important;
}

/* ── Result Cards ── */
.result-card {
    background: var(--surface);
    backdrop-filter: blur(20px);
    border-radius: var(--radius-lg);
    padding: 2rem;
    text-align: center;
    border: 1px solid var(--border-light);
    transition: var(--transition);
    animation: fadeInUp 0.6s ease-out;
}

.result-card-safe {
    border-color: rgba(0,217,126,0.25);
    box-shadow: 0 0 40px rgba(0,217,126,0.08), var(--shadow-card);
}

.result-card-danger {
    border-color: rgba(255,77,106,0.25);
    box-shadow: 0 0 40px rgba(255,77,106,0.08), var(--shadow-card);
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(20px); }
    to { opacity: 1; transform: translateY(0); }
}

.result-icon {
    font-size: 3.5rem;
    margin-bottom: 0.75rem;
    display: inline-block;
}

.result-icon-safe { animation: bounce-safe 1s ease-in-out; }
.result-icon-danger { animation: shake-danger 0.6s ease-in-out; }

@keyframes bounce-safe {
    0%, 100% { transform: scale(1); }
    30% { transform: scale(1.3); }
    60% { transform: scale(0.95); }
    80% { transform: scale(1.05); }
}

@keyframes shake-danger {
    0%, 100% { transform: rotate(0); }
    20% { transform: rotate(-8deg); }
    40% { transform: rotate(8deg); }
    60% { transform: rotate(-5deg); }
    80% { transform: rotate(5deg); }
}

.result-status {
    font-size: 1.5rem;
    font-weight: 700;
    margin-bottom: 0.25rem;
}

.result-status-safe { color: var(--accent-safe); }
.result-status-danger { color: var(--accent-danger); }

.result-subtitle {
    font-size: 0.85rem;
    color: var(--text-secondary);
}

/* ── Confidence Display ── */
.confidence-container {
    margin-top: 1.5rem;
}

.confidence-label {
    font-size: 0.8rem;
    font-weight: 500;
    color: var(--text-secondary);
    margin-bottom: 0.5rem;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

.confidence-value {
    font-size: 2.2rem;
    font-weight: 800;
    letter-spacing: -0.02em;
}

.confidence-value-safe { color: var(--accent-safe); }
.confidence-value-danger { color: var(--accent-danger); }

/* ── Probability Bars ── */
.prob-bar-container {
    margin-top: 1rem;
}

.prob-bar-row {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 0.6rem;
}

.prob-bar-label {
    font-size: 0.78rem;
    font-weight: 500;
    color: var(--text-secondary);
    min-width: 90px;
    text-align: right;
}

.prob-bar-track {
    flex: 1;
    height: 8px;
    background: rgba(255,255,255,0.06);
    border-radius: 4px;
    overflow: hidden;
}

.prob-bar-fill-safe {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent-safe), #34D399);
    animation: barGrow 1s ease-out;
    transition: width 0.5s ease;
}

.prob-bar-fill-danger {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, var(--accent-danger), #FB7185);
    animation: barGrow 1s ease-out;
    transition: width 0.5s ease;
}

.prob-bar-value {
    font-size: 0.8rem;
    font-weight: 600;
    color: var(--text-primary);
    min-width: 55px;
}

@keyframes barGrow {
    from { width: 0 !important; }
}

/* ── Feature Grid ── */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.6rem;
    margin-top: 1rem;
}

@media (max-width: 768px) {
    .feature-grid { grid-template-columns: repeat(2, 1fr); }
}

@media (max-width: 480px) {
    .feature-grid { grid-template-columns: 1fr; }
}

.feature-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.65rem 0.85rem;
    background: var(--surface);
    border-radius: var(--radius-sm);
    border-left: 3px solid transparent;
    transition: var(--transition);
    font-size: 0.82rem;
}

.feature-item:hover {
    background: var(--surface-hover);
    transform: translateX(3px);
}

.feature-safe {
    border-left-color: var(--accent-safe);
}

.feature-suspicious {
    border-left-color: var(--accent-warning);
}

.feature-dangerous {
    border-left-color: var(--accent-danger);
}

.feature-icon {
    font-size: 0.9rem;
    flex-shrink: 0;
}

.feature-name {
    color: var(--text-primary);
    font-weight: 500;
    flex: 1;
    line-height: 1.3;
}

.feature-badge {
    font-size: 0.65rem;
    font-weight: 600;
    padding: 0.15rem 0.5rem;
    border-radius: 10px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
    flex-shrink: 0;
}

.badge-safe {
    background: var(--accent-safe-bg);
    color: var(--accent-safe);
}

.badge-suspicious {
    background: var(--accent-warning-bg);
    color: var(--accent-warning);
}

.badge-dangerous {
    background: var(--accent-danger-bg);
    color: var(--accent-danger);
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12141A 0%, #0E1117 100%) !important;
    border-right: 1px solid var(--border-light) !important;
}

[data-testid="stSidebar"] .block-container {
    padding-top: 1.5rem !important;
}

.sidebar-brand {
    text-align: center;
    padding: 1rem 0 1.5rem;
    border-bottom: 1px solid var(--border-light);
    margin-bottom: 1.5rem;
}

.sidebar-brand-icon {
    font-size: 2.2rem;
    display: block;
    margin-bottom: 0.3rem;
}

.sidebar-brand-name {
    font-size: 1.3rem;
    font-weight: 700;
    background: linear-gradient(135deg, var(--primary), var(--primary-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.sidebar-brand-tag {
    font-size: 0.7rem;
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-top: 0.2rem;
}

.sidebar-section-title {
    font-size: 0.72rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.12em;
    color: var(--text-muted);
    margin-bottom: 0.75rem;
    padding-left: 0.25rem;
}

.sidebar-panel {
    background: var(--surface);
    border: 1px solid var(--border-light);
    border-radius: var(--radius-md);
    padding: 1rem;
    margin-bottom: 1rem;
}

.sidebar-step {
    display: flex;
    align-items: flex-start;
    gap: 0.65rem;
    margin-bottom: 0.75rem;
}

.sidebar-step:last-child { margin-bottom: 0; }

.step-number {
    width: 22px;
    height: 22px;
    min-width: 22px;
    border-radius: 50%;
    background: linear-gradient(135deg, var(--primary), var(--primary-dark));
    color: white;
    font-size: 0.65rem;
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 1px;
}

.step-text {
    font-size: 0.82rem;
    color: var(--text-secondary);
    line-height: 1.45;
}

.step-text strong {
    color: var(--text-primary);
    font-weight: 600;
}

.feature-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: 0.75rem;
    font-weight: 500;
    padding: 0.3rem 0.65rem;
    border-radius: 20px;
    background: var(--surface);
    border: 1px solid var(--border-light);
    color: var(--text-secondary);
    margin: 0.2rem;
    transition: var(--transition);
}

.feature-tag:hover {
    border-color: var(--primary);
    color: var(--primary-light);
}

.tech-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    font-size: 0.72rem;
    font-weight: 600;
    padding: 0.4rem 0.8rem;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(108,99,255,0.12), rgba(108,99,255,0.04));
    border: 1px solid rgba(108,99,255,0.2);
    color: var(--primary-light);
}

/* ── Metric Override ── */
[data-testid="stMetricValue"] {
    font-family: 'Inter', sans-serif !important;
    font-weight: 700 !important;
}

/* ── Progress Bar ── */
.stProgress > div > div > div > div {
    background: linear-gradient(90deg, var(--primary), var(--primary-light)) !important;
    border-radius: 4px;
}

/* ── Expander Styling ── */
.streamlit-expanderHeader {
    font-family: 'Inter', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.9rem !important;
    background: var(--surface) !important;
    border-radius: var(--radius-sm) !important;
    border: 1px solid var(--border-light) !important;
}

/* ── Footer ── */
.custom-footer {
    text-align: center;
    padding: 2rem 1rem 1rem;
    margin-top: 2rem;
    border-top: 1px solid var(--border-light);
}

.footer-brand {
    font-size: 0.85rem;
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 0.3rem;
}

.footer-brand strong {
    background: linear-gradient(135deg, var(--primary), var(--primary-light));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.footer-sub {
    font-size: 0.72rem;
    color: var(--text-muted);
    letter-spacing: 0.03em;
}

.footer-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.65rem;
    font-weight: 600;
    padding: 0.25rem 0.7rem;
    border-radius: 20px;
    background: linear-gradient(135deg, rgba(108,99,255,0.1), rgba(0,217,126,0.05));
    border: 1px solid rgba(108,99,255,0.15);
    color: var(--primary-light);
    margin-top: 0.75rem;
}

/* ── Warning/Info/Error/Success Overrides ── */
.stAlert {
    border-radius: var(--radius-md) !important;
    font-family: 'Inter', sans-serif !important;
}

/* ── Divider ── */
hr {
    border-color: var(--border-light) !important;
    margin: 1.5rem 0 !important;
}
</style>
""", unsafe_allow_html=True)




# ───────────────────────────────────────────────────────────────
# Sidebar
# ───────────────────────────────────────────────────────────────
with st.sidebar:
    # Brand
    st.markdown("""
    <div class="sidebar-brand">
        <span class="sidebar-brand-icon">🛡️</span>
        <div class="sidebar-brand-name">PhisInc</div>
        <div class="sidebar-brand-tag">Phishing Detection Engine</div>
    </div>
    """, unsafe_allow_html=True)

    # How to Use
    st.markdown('<div class="sidebar-section-title">📋 Cara Penggunaan</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-panel">
        <div class="sidebar-step">
            <span class="step-number">1</span>
            <span class="step-text">Masukkan URL lengkap dengan <strong>http://</strong> atau <strong>https://</strong></span>
        </div>
        <div class="sidebar-step">
            <span class="step-number">2</span>
            <span class="step-text">Tekan <strong>Enter</strong> atau klik tombol <strong>Analisis URL</strong></span>
        </div>
        <div class="sidebar-step">
            <span class="step-number">3</span>
            <span class="step-text">Lihat hasil prediksi dan detail fitur yang diekstrak</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature Categories
    st.markdown('<div class="sidebar-section-title">🔬 Fitur yang Dianalisis</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="sidebar-panel">
        <div style="display:flex; flex-wrap:wrap; gap:0.25rem;">
            <span class="feature-tag">🔗 Fitur URL</span>
            <span class="feature-tag">🌐 Fitur Domain</span>
            <span class="feature-tag">📄 Fitur HTML</span>
            <span class="feature-tag">📊 Fitur Eksternal</span>
        </div>
        <div style="margin-top:0.75rem; font-size:0.78rem; color:var(--text-muted);">
            Total <strong style="color:var(--primary-light);">30 fitur</strong> diekstrak dan dianalisis dari setiap URL.
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Tech Stack
    st.markdown('<div class="sidebar-section-title">⚙️ Teknologi</div>', unsafe_allow_html=True)
    st.markdown("""
    <div style="text-align:center; padding:0.5rem 0;">
        <span class="tech-badge">🤖 Gradient Boosting Classifier</span>
    </div>
    """, unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# Hero Section
# ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <div class="hero-icon">🛡️</div>
    <h1 class="hero-title">Phishing URL Detection</h1>
    <p class="hero-subtitle">
        Lindungi diri Anda dari ancaman phishing.
        Sistem kami menggunakan <strong>Machine Learning</strong> untuk menganalisis URL secara real-time.
    </p>
</div>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# Input Section
# ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="section-label">🔍 Masukkan URL untuk dianalisis</div>
""", unsafe_allow_html=True)

with st.form(key='url_form', clear_on_submit=False):
    col1, col2 = st.columns([3, 1])

    with col1:
        url_input = st.text_input(
            "URL Target",
            placeholder="https://example.com",
            help="Masukkan URL lengkap yang ingin dianalisis (harus dimulai dengan http:// atau https://)",
            label_visibility="collapsed"
        )

    with col2:
        analyze_button = st.form_submit_button(
            "🔍  Analisis URL",
            type="primary",
            use_container_width=True
        )


# ───────────────────────────────────────────────────────────────
# Load Model
# ───────────────────────────────────────────────────────────────
model = load_model()


# ───────────────────────────────────────────────────────────────
# Analysis & Results
# ───────────────────────────────────────────────────────────────
if analyze_button and url_input:
    if model is None:
        st.error("Model tidak dapat dimuat. Pastikan file 'gradient_boosting_model.pkl' ada di direktori yang sama.")
    else:
        # Validasi input
        if not url_input.startswith(('http://', 'https://')):
            st.warning("⚠️ URL harus dimulai dengan http:// atau https://")
        else:
            # Progress
            progress_bar = st.progress(0)
            status_text = st.empty()

            try:
                # Extract features
                status_text.text("🔄 Mengekstrak fitur dari URL...")
                progress_bar.progress(30)

                obj = FeatureExtraction(url_input)
                features = obj.getFeaturesList()

                progress_bar.progress(60)
                status_text.text("🤖 Melakukan prediksi...")

                # Reshape features
                features_array = np.array(features).reshape(1, -1)

                # Predict
                prediction = model.predict(features_array)[0]
                probability = model.predict_proba(features_array)[0]

                progress_bar.progress(100)
                status_text.text("✅ Analisis selesai!")
                time.sleep(0.5)
                progress_bar.empty()
                status_text.empty()

                # Compute values
                confidence = max(probability) * 100
                prob_legit = probability[1] * 100
                prob_phish = probability[0] * 100
                is_safe = prediction == 1

                # ── Results Section ──
                st.markdown("""
                <div class="section-label">📊 Hasil Analisis</div>
                """, unsafe_allow_html=True)

                # Main result card
                if is_safe:
                    st.markdown(f"""
                    <div class="result-card result-card-safe">
                        <div class="result-icon result-icon-safe">✅</div>
                        <div class="result-status result-status-safe">LEGITIMATE</div>
                        <div class="result-subtitle">URL ini kemungkinan aman untuk diakses</div>
                        <div class="confidence-container">
                            <div class="confidence-label">Tingkat Kepercayaan</div>
                            <div class="confidence-value confidence-value-safe">{confidence:.1f}%</div>
                        </div>
                        <div class="prob-bar-container">
                            <div class="prob-bar-row">
                                <span class="prob-bar-label">Legitimate</span>
                                <div class="prob-bar-track">
                                    <div class="prob-bar-fill-safe" style="width:{prob_legit:.1f}%"></div>
                                </div>
                                <span class="prob-bar-value">{prob_legit:.1f}%</span>
                            </div>
                            <div class="prob-bar-row">
                                <span class="prob-bar-label">Phishing</span>
                                <div class="prob-bar-track">
                                    <div class="prob-bar-fill-danger" style="width:{prob_phish:.1f}%"></div>
                                </div>
                                <span class="prob-bar-value">{prob_phish:.1f}%</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)
                else:
                    st.markdown(f"""
                    <div class="result-card result-card-danger">
                        <div class="result-icon result-icon-danger">🚨</div>
                        <div class="result-status result-status-danger">PHISHING DETECTED</div>
                        <div class="result-subtitle">URL ini terdeteksi berbahaya — jangan masukkan data sensitif!</div>
                        <div class="confidence-container">
                            <div class="confidence-label">Tingkat Kepercayaan</div>
                            <div class="confidence-value confidence-value-danger">{confidence:.1f}%</div>
                        </div>
                        <div class="prob-bar-container">
                            <div class="prob-bar-row">
                                <span class="prob-bar-label">Phishing</span>
                                <div class="prob-bar-track">
                                    <div class="prob-bar-fill-danger" style="width:{prob_phish:.1f}%"></div>
                                </div>
                                <span class="prob-bar-value">{prob_phish:.1f}%</span>
                            </div>
                            <div class="prob-bar-row">
                                <span class="prob-bar-label">Legitimate</span>
                                <div class="prob-bar-track">
                                    <div class="prob-bar-fill-safe" style="width:{prob_legit:.1f}%"></div>
                                </div>
                                <span class="prob-bar-value">{prob_legit:.1f}%</span>
                            </div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

                # ── Feature Details ──
                feature_names = [
                    "Using IP Address", "Long URL", "Short URL", "Symbol @",
                    "Redirecting //", "Prefix Suffix", "Sub Domains", "HTTPS",
                    "Domain Registration Length", "Favicon", "Non-Standard Port",
                    "HTTPS in Domain", "Request URL", "Anchor URL",
                    "Links in Script Tags", "Server Form Handler", "Info Email",
                    "Abnormal URL", "Website Forwarding", "Status Bar Customization",
                    "Disable Right Click", "Using Popup Window", "Iframe Redirection",
                    "Age of Domain", "DNS Recording", "Website Traffic",
                    "PageRank", "Google Index", "Links Pointing to Page", "Stats Report"
                ]

                with st.expander("🔬 Detail Fitur yang Diekstrak (30 Fitur)", expanded=False):
                    # Summary counts at the top
                    safe_count = sum(1 for v in features if v == 1)
                    suspicious_count = sum(1 for v in features if v == 0)
                    danger_count = sum(1 for v in features if v == -1)

                    sum_cols = st.columns(3)
                    with sum_cols[0]:
                        st.success(f"✅ Safe: **{safe_count}**")
                    with sum_cols[1]:
                        st.warning(f"⚠️ Suspicious: **{suspicious_count}**")
                    with sum_cols[2]:
                        st.error(f"❌ Danger: **{danger_count}**")

                    st.markdown("---")

                    # Display features one by one in 3 columns
                    for i in range(0, len(feature_names), 3):
                        cols = st.columns(3)
                        for j in range(3):
                            idx = i + j
                            if idx < len(feature_names):
                                name = feature_names[idx]
                                value = features[idx]
                                with cols[j]:
                                    if value == 1:
                                        st.markdown(f"✅ **{name}**")
                                        st.caption("Status: Safe")
                                    elif value == 0:
                                        st.markdown(f"⚠️ **{name}**")
                                        st.caption("Status: Suspicious")
                                    else:
                                        st.markdown(f"❌ **{name}**")
                                        st.caption("Status: Dangerous")
                                    st.markdown("")

            except Exception as e:
                progress_bar.empty()
                status_text.empty()
                st.error(f"❌ Terjadi kesalahan saat menganalisis URL: {str(e)}")
                st.info("Pastikan URL valid dan dapat diakses.")

elif analyze_button and not url_input:
    st.warning("⚠️ Silakan masukkan URL terlebih dahulu!")


# ───────────────────────────────────────────────────────────────
# Footer
# ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="custom-footer">
    <div class="footer-brand">🛡️ <strong>PhisInc</strong> — Phishing URL Detection System</div>
    <div class="footer-sub">Selalu verifikasi URL sebelum memasukkan data sensitif</div>
    <div style="margin-top:0.75rem;">
        <span class="footer-badge">⚡ Powered by Machine Learning</span>
    </div>
</div>
""", unsafe_allow_html=True)