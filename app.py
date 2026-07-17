import html
from pathlib import Path

import joblib
import streamlit as st

from explainability import (
    ENGLISH_NAMES,
    FEATURE_INFO,
    build_explanation,
    check_model_contract,
)
from featureextraction import FeatureExtraction

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
# Load Model
# ───────────────────────────────────────────────────────────────
MODEL_PATH = Path(__file__).parent / "gradient_boosting_model.pkl"


@st.cache_resource
def load_model():
    # Biarkan exception menjalar: st.cache_resource tidak menyimpan hasil bila
    # fungsi gagal, sehingga kegagalan sementara (file terkunci) bisa dicoba ulang.
    model = joblib.load(MODEL_PATH)
    check_model_contract(model)
    return model


def try_load_model():
    try:
        return load_model()
    except FileNotFoundError:
        st.error("Model 'gradient_boosting_model.pkl' tidak ditemukan!")
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
    return None


def html_block(markup: str) -> str:
    """Buang indentasi dan baris kosong agar parser Markdown (CommonMark) tidak
    membaca HTML ber-indentasi >=4 spasi sebagai code block, dan baris kosong
    tidak memutus blok HTML di tengah."""
    return "\n".join(
        line.strip() for line in markup.splitlines() if line.strip()
    )


# ───────────────────────────────────────────────────────────────
# Ikon SVG (mewarisi warna via currentColor; dekoratif → aria-hidden)
# ───────────────────────────────────────────────────────────────
def _svg(paths: str, size: int = 20) -> str:
    return (
        f'<svg viewBox="0 0 24 24" width="{size}" height="{size}" fill="none" '
        f'stroke="currentColor" stroke-width="2" stroke-linecap="round" '
        f'stroke-linejoin="round" aria-hidden="true" focusable="false">{paths}</svg>'
    )


ICON_ALERT = _svg('<path d="M10.29 3.86L1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z"/><line x1="12" y1="9" x2="12" y2="13"/><line x1="12" y1="17" x2="12.01" y2="17"/>')
ICON_SHIELD = _svg('<path d="M12 22s8-4 8-10V5l-8-3-8 3v7c0 6 8 10 8 10z"/><polyline points="9 12 11 14 15 10"/>')
ICON_DANGER = _svg('<polygon points="7.86 2 16.14 2 22 7.86 22 16.14 16.14 22 7.86 22 2 16.14 2 7.86 7.86 2"/><line x1="15" y1="9" x2="9" y2="15"/><line x1="9" y1="9" x2="15" y2="15"/>')
ICON_HELP = _svg('<circle cx="12" cy="12" r="10"/><path d="M9.09 9a3 3 0 0 1 5.83 1c0 2-3 3-3 3"/><line x1="12" y1="17" x2="12.01" y2="17"/>')

BAND_ICONS = {
    "high-risk": ICON_DANGER,
    "suspicious": ICON_ALERT,
    "uncertain": ICON_HELP,
    "likely-safe": ICON_SHIELD,
    "safe": ICON_SHIELD,
}


def truncate_middle(text: str, max_len: int = 60) -> str:
    if len(text) <= max_len:
        return text
    half = (max_len - 1) // 2
    return text[:half] + "…" + text[-half:]


# ───────────────────────────────────────────────────────────────
# Design System — Global CSS
# ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ── Root Variables ── */
:root {
    --primary: #6C63FF;
    --primary-light: #8B83FF;
    --primary-dark: #5A51F0;
    --accent-safe: #00D97E;
    --accent-safe-soft: #7BD389;
    --accent-danger: #FF4D6A;
    --accent-warning: #FFB347;
    --accent-suspicious: #FF7A59;
    --bg-primary: #0E1117;
    --surface: rgba(255,255,255,0.03);
    --surface-hover: rgba(255,255,255,0.06);
    --text-primary: #F0F2F6;
    --text-secondary: #9CA3AF;
    --text-muted: #8A93A2;
    --border: rgba(108,99,255,0.15);
    --border-light: rgba(255,255,255,0.06);
    --radius-lg: 16px;
    --radius-md: 12px;
    --radius-sm: 8px;
    --shadow-card: 0 4px 24px rgba(0,0,0,0.3);
    --transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    /* Skala tipografi — 6 langkah, minimum 12px */
    --fs-xs: 0.75rem;
    --fs-sm: 0.875rem;
    --fs-base: 1rem;
    --fs-lg: 1.25rem;
    --fs-xl: 1.75rem;
    --fs-2xl: 2.1rem;
}

html, body, [class*="css"] {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif !important;
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
header {visibility: hidden;}
.stDeployButton {display: none;}

::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: var(--bg-primary); }
::-webkit-scrollbar-thumb { background: var(--primary-dark); border-radius: 3px; }
* { scrollbar-width: thin; scrollbar-color: var(--primary-dark) var(--bg-primary); }

.block-container {
    padding-top: 2rem !important;
    padding-bottom: 2rem !important;
    max-width: 1100px !important;
}

/* ── Fokus terlihat untuk semua elemen interaktif ── */
:is(button, summary, a, input):focus-visible {
    outline: 2px solid var(--primary-light) !important;
    outline-offset: 2px;
    box-shadow: 0 0 0 4px rgba(108,99,255,0.25) !important;
}

/* ── Hormati preferensi reduksi gerakan ── */
@media (prefers-reduced-motion: reduce) {
    *, *::before, *::after {
        animation-duration: 0.01ms !important;
        animation-iteration-count: 1 !important;
        transition-duration: 0.01ms !important;
    }
}

/* ── Hero (kompak, rata kiri) ── */
.hero-container {
    display: flex;
    align-items: center;
    gap: 1rem;
    text-align: left;
    padding: 1.25rem 0 1rem;
    margin-bottom: 0.5rem;
}

.hero-icon { font-size: 2rem; }

.hero-title {
    font-size: clamp(1.6rem, 4vw, var(--fs-2xl));
    font-weight: 800;
    letter-spacing: -0.02em;
    background: linear-gradient(135deg, #8B83FF 0%, #6C63FF 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    margin: 0;
    line-height: 1.2;
}

.hero-subtitle {
    font-size: 0.95rem;
    color: var(--text-secondary);
    margin: 0.15rem 0 0;
    max-width: 56ch;
    line-height: 1.6;
}

.hero-subtitle strong { color: var(--primary-light); font-weight: 600; }

/* ── Section Label (eyebrow — satu-satunya gaya uppercase) ── */
.section-label {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: var(--fs-xs);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--primary-light);
    margin: 1rem 0;
}

.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, var(--border), transparent);
}

/* ── Input ── */
.stTextInput > div > div > input,
.stTextInput input,
input[type="text"] {
    background: rgba(255,255,255,0.08) !important;
    border: 1.5px solid var(--border) !important;
    border-radius: var(--radius-md) !important;
    color: #FFFFFF !important;
    -webkit-text-fill-color: #FFFFFF !important;
    caret-color: #FFFFFF !important;
    font-size: var(--fs-base) !important;
    padding: 0.75rem 1rem !important;
    transition: var(--transition) !important;
}

.stTextInput input:focus,
input[type="text"]:focus {
    border-color: var(--primary) !important;
    box-shadow: 0 0 0 3px rgba(108,99,255,0.2) !important;
    background: rgba(255,255,255,0.1) !important;
}

.stTextInput input::placeholder,
input[type="text"]::placeholder {
    color: #9CA3AF !important;
    -webkit-text-fill-color: #9CA3AF !important;
    opacity: 1 !important;
}

.input-hint {
    font-size: var(--fs-xs);
    color: var(--text-muted);
    margin-top: 0.35rem;
}

/* ── Button ── */
.stButton > button, .stFormSubmitButton > button {
    background: linear-gradient(135deg, var(--primary-dark) 0%, var(--primary) 100%) !important;
    color: white !important;
    border: none !important;
    border-radius: var(--radius-md) !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    padding: 0.75rem 2rem !important;
    letter-spacing: 0.02em;
    transition: var(--transition) !important;
    width: 100%;
    box-shadow: 0 4px 15px rgba(108,99,255,0.3) !important;
}

.stButton > button:hover, .stFormSubmitButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 25px rgba(108,99,255,0.45) !important;
}

.stButton > button:active, .stFormSubmitButton > button:active {
    transform: translateY(0) !important;
}

/* ── Verdict Band ── */
.verdict-band {
    display: flex;
    align-items: center;
    gap: 1rem;
    padding: 1.1rem 1.4rem;
    border-radius: var(--radius-lg);
    border: 1px solid;
    margin-bottom: 1rem;
    animation: fadeInUp 0.4s ease-out;
}

.verdict-high-risk   { border-color: rgba(255,77,106,0.45);  background: rgba(255,77,106,0.10);  color: var(--accent-danger); }
.verdict-suspicious  { border-color: rgba(255,122,89,0.45);  background: rgba(255,122,89,0.10);  color: var(--accent-suspicious); }
.verdict-uncertain   { border-color: rgba(255,179,71,0.45);  background: rgba(255,179,71,0.10);  color: var(--accent-warning); }
.verdict-likely-safe { border-color: rgba(123,211,137,0.45); background: rgba(123,211,137,0.10); color: var(--accent-safe-soft); }
.verdict-safe        { border-color: rgba(0,217,126,0.45);   background: rgba(0,217,126,0.10);   color: var(--accent-safe); }

.verdict-icon { display: flex; flex-shrink: 0; }
.verdict-icon svg { width: 28px; height: 28px; }

.verdict-label {
    font-size: var(--fs-lg);
    font-weight: 800;
    letter-spacing: 0.02em;
}

.verdict-sub {
    font-size: var(--fs-sm);
    color: var(--text-primary);
    margin-top: 0.15rem;
    line-height: 1.5;
}

.verdict-url {
    margin-left: auto;
    font-size: var(--fs-xs);
    color: var(--text-secondary);
    max-width: 28ch;
    overflow: hidden;
    text-overflow: ellipsis;
    white-space: nowrap;
    flex-shrink: 0;
}

@keyframes fadeInUp {
    from { opacity: 0; transform: translateY(12px); }
    to { opacity: 1; transform: translateY(0); }
}

/* ── Stacked Probability Bar ── */
.stack-bar-wrap {
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin: 0.5rem 0 1.25rem;
}

.stack-label {
    font-size: var(--fs-xs);
    font-weight: 600;
    white-space: nowrap;
}

.stack-label-danger { color: var(--accent-danger); }
.stack-label-safe { color: var(--accent-safe); }

.stack-bar {
    flex: 1;
    display: flex;
    height: 10px;
    border-radius: 5px;
    overflow: hidden;
    background: rgba(255,255,255,0.06);
}

.stack-fill-phish { background: linear-gradient(90deg, var(--accent-danger), #FB7185); }
.stack-fill-legit { background: linear-gradient(90deg, #34D399, var(--accent-safe)); }

/* ── Reason Cards (penjelasan XAI) ── */
.reason-group { margin-bottom: 1.25rem; }

.reason-group-title {
    display: flex;
    align-items: center;
    gap: 0.5rem;
    font-size: var(--fs-sm);
    font-weight: 700;
    margin: 0 0 0.6rem;
}

.reason-group-risk { color: var(--accent-danger); }
.reason-group-trust { color: var(--accent-safe); }

.reason-card {
    display: flex;
    align-items: flex-start;
    gap: 0.9rem;
    padding: 0.9rem 1.1rem;
    margin-bottom: 0.6rem;
    background: var(--surface);
    border: 1px solid var(--border-light);
    border-left-width: 4px;
    border-radius: var(--radius-md);
    flex-wrap: wrap;
}

.reason-risk { border-left-color: var(--accent-danger); }
.reason-warn { border-left-color: var(--accent-warning); }
.reason-trust { border-left-color: var(--accent-safe); }
.reason-unverified { border-left-color: var(--text-muted); }

.reason-icon { display: flex; flex-shrink: 0; margin-top: 2px; }
.reason-risk .reason-icon { color: var(--accent-danger); }
.reason-warn .reason-icon { color: var(--accent-warning); }
.reason-trust .reason-icon { color: var(--accent-safe); }
.reason-unverified .reason-icon { color: var(--text-muted); }

.reason-body { flex: 1; min-width: 220px; }

.reason-title {
    font-size: var(--fs-sm);
    font-weight: 600;
    color: var(--text-primary);
}

.reason-en {
    font-size: var(--fs-xs);
    font-weight: 400;
    color: var(--text-muted);
    margin-left: 0.4rem;
}

.reason-text {
    font-size: var(--fs-sm);
    color: var(--text-secondary);
    line-height: 1.5;
    margin-top: 0.15rem;
}

.reason-note {
    font-size: var(--fs-xs);
    color: var(--text-muted);
    font-style: italic;
    margin-top: 0.25rem;
}

.reason-weight {
    display: flex;
    align-items: center;
    gap: 3px;
    margin-left: auto;
    flex-shrink: 0;
}

.reason-weight .dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: rgba(255,255,255,0.12);
}

.reason-risk .dot.on, .reason-warn .dot.on { background: var(--accent-danger); }
.reason-trust .dot.on { background: var(--accent-safe); }
.reason-unverified .dot.on { background: var(--text-muted); }

.reason-weight-label {
    font-size: var(--fs-xs);
    color: var(--text-secondary);
    margin-left: 0.35rem;
}

.reason-empty {
    font-size: var(--fs-sm);
    color: var(--text-secondary);
    padding: 0.6rem 0.2rem;
}

/* ── Summary Chips ── */
.chip-row {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 1rem;
}

.chip {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: var(--fs-xs);
    font-weight: 600;
    padding: 0.3rem 0.75rem;
    border-radius: 20px;
    border: 1px solid;
}

.chip-danger { color: var(--accent-danger); border-color: rgba(255,77,106,0.35); background: rgba(255,77,106,0.08); }
.chip-warn   { color: var(--accent-warning); border-color: rgba(255,179,71,0.35); background: rgba(255,179,71,0.08); }
.chip-safe   { color: var(--accent-safe); border-color: rgba(0,217,126,0.35); background: rgba(0,217,126,0.08); }

/* ── Feature Grid (dalam expander) ── */
.feature-grid {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 0.6rem;
    margin-top: 0.5rem;
}

@media (max-width: 768px) {
    .feature-grid { grid-template-columns: repeat(2, 1fr); }
}

@media (max-width: 480px) {
    .feature-grid { grid-template-columns: 1fr; }
    .verdict-band { flex-wrap: wrap; }
    .verdict-url { margin-left: 0; flex-basis: 100%; max-width: 100%; }
}

.feature-item {
    display: flex;
    align-items: center;
    gap: 0.6rem;
    padding: 0.65rem 0.85rem;
    background: var(--surface);
    border-radius: var(--radius-sm);
    border-left: 3px solid transparent;
    font-size: var(--fs-xs);
}

.feature-safe { border-left-color: var(--accent-safe); }
.feature-suspicious { border-left-color: var(--accent-warning); }
.feature-dangerous { border-left-color: var(--accent-danger); }

.feature-name {
    color: var(--text-primary);
    font-weight: 500;
    flex: 1;
    line-height: 1.3;
}

.feature-sub {
    font-size: 0.7rem;
    font-weight: 400;
    color: var(--text-muted);
}

.feature-badge {
    font-size: var(--fs-xs);
    font-weight: 600;
    padding: 0.15rem 0.5rem;
    border-radius: 10px;
    flex-shrink: 0;
}

.badge-safe { background: rgba(0,217,126,0.08); color: var(--accent-safe); }
.badge-suspicious { background: rgba(255,179,71,0.08); color: var(--accent-warning); }
.badge-dangerous { background: rgba(255,77,106,0.08); color: var(--accent-danger); }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #12141A 0%, #0E1117 100%) !important;
    border-right: 1px solid var(--border-light) !important;
}

[data-testid="stSidebar"] .block-container { padding-top: 1.5rem !important; }

.sidebar-brand {
    text-align: center;
    padding: 1rem 0 1.5rem;
    border-bottom: 1px solid var(--border-light);
    margin-bottom: 1.5rem;
}

.sidebar-brand-icon { font-size: 2.2rem; display: block; margin-bottom: 0.3rem; }

.sidebar-brand-name {
    font-size: var(--fs-lg);
    font-weight: 700;
    background: linear-gradient(135deg, var(--primary-light), var(--primary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.sidebar-brand-tag {
    font-size: var(--fs-xs);
    color: var(--text-muted);
    text-transform: uppercase;
    letter-spacing: 0.15em;
    margin-top: 0.2rem;
}

.sidebar-section-title {
    font-size: var(--fs-xs);
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
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
    font-size: var(--fs-xs);
    font-weight: 700;
    display: flex;
    align-items: center;
    justify-content: center;
    margin-top: 1px;
}

.step-text {
    font-size: var(--fs-sm);
    color: var(--text-secondary);
    line-height: 1.45;
}

.step-text strong { color: var(--text-primary); font-weight: 600; }

.feature-tag {
    display: inline-flex;
    align-items: center;
    gap: 0.35rem;
    font-size: var(--fs-xs);
    font-weight: 500;
    padding: 0.3rem 0.65rem;
    border-radius: 20px;
    background: var(--surface);
    border: 1px solid var(--border-light);
    color: var(--text-secondary);
    margin: 0.2rem;
}

.sidebar-disclaimer {
    font-size: var(--fs-xs);
    color: var(--text-muted);
    line-height: 1.5;
    border-left: 3px solid var(--accent-warning);
    padding-left: 0.6rem;
}

/* ── Expander ── */
[data-testid="stExpander"] summary {
    font-weight: 600 !important;
    font-size: var(--fs-sm) !important;
}

[data-testid="stExpander"] {
    background: var(--surface);
    border: 1px solid var(--border-light) !important;
    border-radius: var(--radius-md) !important;
}

/* ── Footer ── */
.custom-footer {
    text-align: center;
    padding: 2rem 1rem 1rem;
    margin-top: 2rem;
    border-top: 1px solid var(--border-light);
}

.footer-brand {
    font-size: var(--fs-sm);
    font-weight: 600;
    color: var(--text-secondary);
    margin-bottom: 0.3rem;
}

.footer-brand strong {
    background: linear-gradient(135deg, var(--primary-light), var(--primary));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}

.footer-sub {
    font-size: var(--fs-xs);
    color: var(--text-muted);
    letter-spacing: 0.03em;
}

.stAlert { border-radius: var(--radius-md) !important; }

hr { border-color: var(--border-light) !important; margin: 1.5rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# Sidebar
# ───────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(html_block("""
    <div class="sidebar-brand">
        <span class="sidebar-brand-icon" aria-hidden="true">🛡️</span>
        <div class="sidebar-brand-name">PhisInc</div>
        <div class="sidebar-brand-tag">Phishing Detection Engine</div>
    </div>
    """), unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Cara Penggunaan</div>', unsafe_allow_html=True)
    st.markdown(html_block("""
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
            <span class="step-text">Baca vonis, <strong>alasan di baliknya</strong>, dan detail 30 indikator</span>
        </div>
    </div>
    """), unsafe_allow_html=True)

    st.markdown('<div class="sidebar-section-title">Fitur yang Dianalisis</div>', unsafe_allow_html=True)
    st.markdown(html_block("""
    <div class="sidebar-panel">
        <div style="display:flex; flex-wrap:wrap; gap:0.25rem;">
            <span class="feature-tag">Fitur URL</span>
            <span class="feature-tag">Fitur Domain</span>
            <span class="feature-tag">Fitur HTML</span>
            <span class="feature-tag">Fitur Eksternal</span>
        </div>
        <div style="margin-top:0.75rem; font-size:var(--fs-xs); color:var(--text-muted);">
            Total <strong style="color:var(--primary-light);">30 indikator</strong> diekstrak dan dianalisis dari setiap URL.
        </div>
    </div>
    """), unsafe_allow_html=True)

    st.markdown(html_block("""
    <div class="sidebar-panel">
        <div class="sidebar-disclaimer">
            Hasil analisis bersifat <strong>prediksi, bukan jaminan</strong>.
            Tetap waspada dan jangan pernah memasukkan kata sandi, OTP, atau
            data kartu pada situs yang meragukan.
        </div>
    </div>
    """), unsafe_allow_html=True)

    with st.expander("Tentang model"):
        st.caption(
            "Klasifikasi menggunakan **Gradient Boosting Classifier** yang dilatih "
            "pada 11.000+ URL berlabel. Penjelasan per-URL dihitung dengan atribusi "
            "kontrafaktual: setiap indikator diuji ulang untuk mengukur pengaruhnya "
            "terhadap skor phishing."
        )


# ───────────────────────────────────────────────────────────────
# Hero
# ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-container">
    <span class="hero-icon" aria-hidden="true">🛡️</span>
    <div>
        <h1 class="hero-title">Phishing URL Detection</h1>
        <p class="hero-subtitle">
            Analisis URL secara real-time dengan <strong>Machine Learning</strong> —
            lengkap dengan penjelasan mengapa sebuah tautan dinilai berbahaya atau aman.
        </p>
    </div>
</div>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────
# Input
# ───────────────────────────────────────────────────────────────
st.markdown('<h2 class="section-label">Masukkan URL untuk dianalisis</h2>', unsafe_allow_html=True)

with st.form(key='url_form', clear_on_submit=False):
    col1, col2 = st.columns([3, 1])

    with col1:
        url_input = st.text_input(
            "URL Target",
            placeholder="https://example.com",
            label_visibility="collapsed"
        )

    with col2:
        analyze_button = st.form_submit_button(
            "Analisis URL",
            type="primary",
            use_container_width=True
        )

st.markdown(
    '<div class="input-hint">Contoh: https://nama-situs.com — sertakan http:// atau https://</div>',
    unsafe_allow_html=True
)

model = try_load_model()


# ───────────────────────────────────────────────────────────────
# Renderer hasil analisis
# ───────────────────────────────────────────────────────────────
def render_weight(efek: float, max_efek: float) -> str:
    """Meter 3 titik + label teks, berdasarkan besaran efek relatif terhadap
    efek terbesar dalam kelompoknya (bukan posisi urut, yang bisa menyesatkan
    saat faktor tak terverifikasi diturunkan prioritasnya)."""
    ratio = efek / max_efek if max_efek > 0 else 0.0
    if ratio >= 2 / 3:
        n_on, label = 3, "Pengaruh besar"
    elif ratio >= 1 / 3:
        n_on, label = 2, "Pengaruh sedang"
    else:
        n_on, label = 1, "Pengaruh kecil"
    dots = "".join(
        f'<span class="dot{" on" if d < n_on else ""}"></span>' for d in range(3)
    )
    return (
        f'<div class="reason-weight" role="img" aria-label="{label}">'
        f'{dots}<span class="reason-weight-label">{label}</span></div>'
    )


def render_reason_card(factor: dict, max_efek: float, kind: str) -> str:
    if not factor["andal"]:
        css, icon = "reason-unverified", ICON_HELP
    elif kind == "risk":
        css = "reason-warn" if factor["nilai"] == 0 else "reason-risk"
        icon = ICON_ALERT
    else:
        css, icon = "reason-trust", ICON_SHIELD

    en_name = ENGLISH_NAMES[factor["index"]]
    note = (
        f'<div class="reason-note">{html.escape(factor["catatan"])}</div>'
        if factor["catatan"] else ""
    )
    return f"""
    <div class="reason-card {css}">
        <span class="reason-icon">{icon}</span>
        <div class="reason-body">
            <div class="reason-title">{html.escape(factor["nama"])}
                <span class="reason-en">{html.escape(en_name)}</span>
            </div>
            <div class="reason-text">{html.escape(factor["alasan"])}</div>
            {note}
        </div>
        {render_weight(factor["efek"], max_efek)}
    </div>
    """


def render_reason_group(title: str, factors: list, kind: str, empty_text: str) -> str:
    group_css = "reason-group-risk" if kind == "risk" else "reason-group-trust"
    icon = ICON_ALERT if kind == "risk" else ICON_SHIELD
    max_efek = max((f["efek"] for f in factors), default=0.0)
    cards = (
        "".join(render_reason_card(f, max_efek, kind) for f in factors)
        if factors else f'<div class="reason-empty">{empty_text}</div>'
    )
    return f"""
    <div class="reason-group">
        <h3 class="reason-group-title {group_css}">
            <span aria-hidden="true">{icon}</span>{title}
        </h3>
        {cards}
    </div>
    """


def render_results(url: str, features: list, expl: dict) -> None:
    band = expl["band"]
    p_phish = expl["p_phish"] * 100
    p_legit = expl["p_legit"] * 100

    st.markdown('<h2 class="section-label">Hasil Analisis</h2>', unsafe_allow_html=True)

    # ── Verdict band ──
    url_esc = html.escape(url)
    st.markdown(html_block(f"""
    <div class="verdict-band verdict-{band["kode"]}" role="status">
        <span class="verdict-icon">{BAND_ICONS[band["kode"]]}</span>
        <div class="verdict-text">
            <div class="verdict-label">{band["label"]}</div>
            <div class="verdict-sub">{band["pesan"]}</div>
        </div>
        <div class="verdict-url" title="{url_esc}">{html.escape(truncate_middle(url))}</div>
    </div>
    """), unsafe_allow_html=True)

    # ── Stacked probability bar ──
    st.markdown(html_block(f"""
    <div class="stack-bar-wrap" role="img"
         aria-label="Skor model: phishing {p_phish:.1f} persen, aman {p_legit:.1f} persen">
        <span class="stack-label stack-label-danger">Phishing {p_phish:.1f}%</span>
        <div class="stack-bar">
            <div class="stack-fill-phish" style="width:{p_phish:.1f}%"></div>
            <div class="stack-fill-legit" style="width:{p_legit:.1f}%"></div>
        </div>
        <span class="stack-label stack-label-safe">{p_legit:.1f}% Aman</span>
    </div>
    """), unsafe_allow_html=True)

    # ── Peringatan kejujuran data ──
    if not expl["fetch_ok"]:
        st.warning(
            "Halaman tidak dapat diambil (situs tidak merespons). Sebagian indikator "
            "konten otomatis bernilai terburuk, sehingga skor bisa condong ke phishing "
            "bukan karena bukti nyata. Faktor terdampak diberi tanda tanya."
        )
    if not expl["whois_ok"]:
        st.info(
            "Data WHOIS domain tidak tersedia. Indikator umur/registrasi domain memakai "
            "nilai default dan diberi tanda tanya."
        )

    # ── Penjelasan: mengapa? ──
    st.markdown('<h2 class="section-label">Mengapa hasilnya seperti ini?</h2>', unsafe_allow_html=True)

    risk_html = render_reason_group(
        "Faktor risiko utama", expl["risk_factors"], "risk",
        "Tidak ada faktor risiko yang berarti ditemukan untuk URL ini."
    )
    trust_html = render_reason_group(
        "Faktor yang meyakinkan", expl["trust_factors"], "trust",
        "Tidak ada faktor yang meyakinkan ditemukan untuk URL ini."
    )

    # Kelompok yang searah dengan vonis ditampilkan lebih dulu
    # (> 0.40 mengikuti ambang band "uncertain" di verdict_band).
    if expl["p_phish"] > 0.40:
        st.markdown(html_block(risk_html + trust_html), unsafe_allow_html=True)
    else:
        st.markdown(html_block(trust_html + risk_html), unsafe_allow_html=True)

    if expl["n_unreliable_bad"] > 0:
        st.caption(
            f"{expl['n_unreliable_bad']} indikator bernilai 'buruk' berasal dari sinyal yang "
            "tidak dapat diverifikasi (layanan eksternal mati / ekstraksi gagal) dan "
            "diberi prioritas rendah dalam penjelasan."
        )

    st.caption(
        "**Cara membaca:** \"Pengaruh\" diukur dengan menguji ulang model — satu indikator "
        "diubah pada satu waktu — lalu melihat seberapa besar skor phishing berubah. "
        "Skor model belum dikalibrasi penuh; perlakukan sebagai peringkat risiko, bukan "
        "probabilitas kejadian yang presisi."
    )

    # ── Detail 30 indikator ──
    with st.expander("Lihat semua 30 indikator teknis"):
        safe_count = sum(1 for v in features if v == 1)
        suspicious_count = sum(1 for v in features if v == 0)
        danger_count = sum(1 for v in features if v == -1)

        st.markdown(html_block(f"""
        <div class="chip-row">
            <span class="chip chip-danger">✕ {danger_count} indikasi buruk</span>
            <span class="chip chip-warn">! {suspicious_count} mencurigakan</span>
            <span class="chip chip-safe">✓ {safe_count} indikasi aman</span>
        </div>
        """), unsafe_allow_html=True)

        # Urutkan: buruk dulu, lalu mencurigakan, lalu aman.
        order = sorted(range(len(features)), key=lambda i: features[i])
        items = []
        for i in order:
            v = features[i]
            if v == -1:
                css, badge_css, badge = "feature-dangerous", "badge-dangerous", "BAHAYA"
            elif v == 0:
                css, badge_css, badge = "feature-suspicious", "badge-suspicious", "WASPADA"
            else:
                css, badge_css, badge = "feature-safe", "badge-safe", "AMAN"
            items.append(f"""
            <div class="feature-item {css}">
                <div class="feature-name">{html.escape(FEATURE_INFO[i]["nama"])}
                    <div class="feature-sub">{html.escape(ENGLISH_NAMES[i])}</div>
                </div>
                <span class="feature-badge {badge_css}">{badge}</span>
            </div>
            """)

        st.markdown(
            html_block(f'<div class="feature-grid">{"".join(items)}</div>'),
            unsafe_allow_html=True,
        )


# ───────────────────────────────────────────────────────────────
# Analysis
# ───────────────────────────────────────────────────────────────
url_clean = url_input.strip() if url_input else ""

if analyze_button and url_clean:
    if model is None:
        st.error("Model tidak dapat dimuat. Pastikan file 'gradient_boosting_model.pkl' ada di direktori yang sama.")
    elif not url_clean.lower().startswith(('http://', 'https://')):
        st.warning("URL harus dimulai dengan http:// atau https://")
    else:
        try:
            with st.spinner("Mengekstrak indikator dan menganalisis URL…"):
                obj = FeatureExtraction(url_clean)
                features = obj.getFeaturesList()
                expl = build_explanation(model, features, fe=obj)

            render_results(url_clean, features, expl)

        except Exception as e:
            st.error(f"Terjadi kesalahan saat menganalisis URL: {e}")
            st.info("Pastikan URL valid dan dapat diakses.")

elif analyze_button and not url_clean:
    st.warning("Silakan masukkan URL terlebih dahulu!")


# ───────────────────────────────────────────────────────────────
# Footer
# ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="custom-footer">
    <div class="footer-brand"><strong>PhisInc</strong> — Phishing URL Detection System</div>
    <div class="footer-sub">Selalu verifikasi URL sebelum memasukkan data sensitif</div>
</div>
""", unsafe_allow_html=True)
