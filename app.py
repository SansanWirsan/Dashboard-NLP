"""
═══════════════════════════════════════════════════════════════════════════
  📰 NLP DASHBOARD — Reuters-21578 Text Classification
  Algoritma : Decision Tree (CART)
  Eksperimen: BoW | N-gram | TF-IDF
  Author    : Mini Project NLP
═══════════════════════════════════════════════════════════════════════════
Cara menjalankan:
    pip install -r requirements.txt
    streamlit run app.py
"""
import base64
import ast
import re
import time
import ast
import re
import time
import warnings
from collections import Counter
from io import BytesIO

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    precision_recall_fscore_support
)
from sklearn.feature_selection import chi2, SelectKBest
from sklearn.svm import LinearSVC

import matplotlib.pyplot as plt
import seaborn as sns

warnings.filterwarnings("ignore")

# ───────────────────────────────────────────────────────────────────────────
# KONFIGURASI HALAMAN
# ───────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="NLP Dashboard — Reuters DT",
    page_icon="favicon.svg",
    layout="wide",
    initial_sidebar_state="expanded",
)
# --- FUNGSI HELPER UNTUK FOTO PROFIL ---
def get_base64_image(image_path):
    """Mengonversi gambar lokal ke format base64 agar bisa ditampilkan di HTML."""
    try:
        with open(image_path, "rb") as img_file:
            return base64.b64encode(img_file.read()).decode()
    except Exception:
        return None

# ───────────────────────────────────────────────────────────────────────────
# SIDEBAR — NAVIGASI & PROFIL
# ───────────────────────────────────────────────────────────────────────────
with st.sidebar:
    # --- BAGIAN FOTO PROFIL ---
    # Ganti "foto_profil.jpg" dengan nama file foto asli kamu yang ada di folder project
    img_b64 = get_base64_image("foto_profil.jpg")
    
    if img_b64:
        st.markdown(f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <img src="data:image/jpeg;base64,{img_b64}" width="100" 
                     style="border-radius: 50%; border: 3px solid var(--brand-1); 
                     box-shadow: var(--shadow-md); object-fit: cover; height: 100px;">
                <h4 style="margin: 10px 0 0 0; color: #f8fafc !important;">Wirsan Wijoyo</h4>
                <p style="color: #cbd5e1 !important; font-size: 13px; margin: 0;">202310370311193 | UMM</p>
            </div>
        """, unsafe_allow_html=True)
    else:
        # Jika foto tidak ditemukan, gunakan avatar inisial sebagai cadangan
        st.markdown(f"""
            <div style="text-align: center; margin-bottom: 20px;">
                <img src="https://ui-avatars.com/api/?name=Wirsan+Wijoyo&background=2E75B6&color=fff&rounded=true&size=100" 
                     width="100" style="border-radius: 50%;">
                <h4 style="margin: 10px 0 0 0; color: #f8fafc !important;">Wirsan Wijoyo</h4>
                <p style="color: #cbd5e1 !important; font-size: 13px; margin: 0;">202310370311193 | UMM</p>
            </div>
        """, unsafe_allow_html=True)

    # (judul & navigasi ada di blok sidebar berikutnya)
# ───────────────────────────────────────────────────────────────────────────
# CUSTOM CSS — UI/UX PROFESIONAL
# ───────────────────────────────────────────────────────────────────────────
# Deteksi tema aktif Streamlit (light / dark) — bukan dari OS,
# supaya warna teks tidak "hilang" saat user memaksa tema terang.
try:
    _THEME_BASE = st.get_option("theme.base") or "light"
except Exception:
    _THEME_BASE = "light"
_IS_DARK = str(_THEME_BASE).lower() == "dark"

if _IS_DARK:
    _TOKENS = """
        --surface:        #1e293b;
        --surface-2:      #0f172a;
        --surface-hover:  #334155;
        --text:           #f1f5f9;
        --text-muted:     #cbd5e1;
        --border:         rgba(255, 255, 255, 0.10);
        --shadow-sm: 0 2px 8px rgba(0, 0, 0, 0.30);
        --shadow-md: 0 8px 24px rgba(0, 0, 0, 0.40);
    """
else:
    _TOKENS = """
        --surface:        #ffffff;
        --surface-2:      #f8fafc;
        --surface-hover:  #f1f5f9;
        --text:           #0f172a;
        --text-muted:     #334155;
        --border:         rgba(15, 23, 42, 0.10);
        --shadow-sm: 0 2px 8px rgba(15, 23, 42, 0.06);
        --shadow-md: 0 8px 24px rgba(15, 23, 42, 0.10);
    """

st.markdown(f"""
<style>
    :root {{
        --brand-1: #2E75B6;
        --brand-2: #7030A0;
        --brand-3: #4ECDC4;
        --accent:  #ED7D31;
        --radius-lg: 16px;
        --radius-md: 12px;
        --shadow-lg: 0 18px 40px rgba(46, 117, 182, 0.22);
        --transition: all .25s cubic-bezier(.4,0,.2,1);
        --hero-text:      #ffffff;
        --hero-sub:       #e8ecf4;
        {_TOKENS}
    }}

    html {{ scroll-behavior: smooth; }}
    .stApp, .block-container {{
        font-family: 'Inter', 'Segoe UI', system-ui, -apple-system, sans-serif;
        color: var(--text);
    }}
    .block-container {{ padding-top: 2rem; padding-bottom: 3rem; max-width: 1400px; }}

    /* Headings — paksa kontras tinggi pada kedua tema */
    h1, h2, h3, h4, h5, h6,
    .stMarkdown h1, .stMarkdown h2, .stMarkdown h3,
    .stMarkdown h4, .stMarkdown h5, .stMarkdown h6 {{
        color: var(--text) !important;
        letter-spacing: -0.3px;
        opacity: 1 !important;
        text-shadow: none !important;
        -webkit-text-fill-color: var(--text) !important;
        background: none !important;
        -webkit-background-clip: initial !important;
        background-clip: initial !important;
    }}
    h1 {{ font-weight: 800; }}
    h2 {{ font-weight: 700; }}

    /* Paragraf & teks markdown umum */
    .stMarkdown p, .stMarkdown li, .stMarkdown span, .stMarkdown label,
    .stCaption, p, label {{
        color: var(--text) !important;
    }}
    .stMarkdown small, .stCaption {{ color: var(--text-muted) !important; }}

    /* HERO */
    .hero {{
        position: relative; overflow: hidden;
        background: linear-gradient(120deg, var(--brand-1) 0%, var(--brand-2) 100%);
        padding: 2.2rem 2.5rem; border-radius: var(--radius-lg);
        box-shadow: var(--shadow-lg);
        margin-bottom: 1.8rem;
        transition: var(--transition);
    }}
    .hero:hover {{ transform: translateY(-2px); box-shadow: 0 22px 50px rgba(112,48,160,0.30); }}
    .hero::before {{
        content: ""; position: absolute; top: -50%; left: -50%;
        width: 200%; height: 200%;
        background: radial-gradient(circle, rgba(255,255,255,0.15) 0%, transparent 60%);
        animation: shimmer 8s ease-in-out infinite;
    }}
    .hero h1, .hero p {{ position: relative; z-index: 1; }}
    .hero h1 {{ color: var(--hero-text) !important; -webkit-text-fill-color: var(--hero-text) !important; margin: 0 0 .5rem 0; font-size: 2.3rem; }}
    .hero p {{ color: var(--hero-sub) !important; -webkit-text-fill-color: var(--hero-sub) !important; margin: 0; font-size: 1.05rem; }}
    @keyframes shimmer {{
        0%,100% {{ transform: translate(0,0); }}
        50%     {{ transform: translate(10%, 10%); }}
    }}

    /* METRIC CARDS */
    [data-testid="stMetric"] {{
        background: var(--surface);
        padding: 1.2rem 1.4rem;
        border-radius: var(--radius-md);
        border: 1px solid var(--border);
        border-left: 4px solid var(--brand-1);
        box-shadow: var(--shadow-sm);
        transition: var(--transition);
    }}
    [data-testid="stMetric"]:hover {{
        transform: translateY(-4px) scale(1.01);
        box-shadow: var(--shadow-md);
        border-left-color: var(--accent);
    }}
    [data-testid="stMetricLabel"] {{ font-weight: 600; color: var(--text-muted) !important; }}
    [data-testid="stMetricValue"] {{ color: var(--text) !important; font-weight: 800; }}

    /* SIDEBAR — tetap gelap di kedua tema */
    [data-testid="stSidebar"] {{
        background: linear-gradient(180deg, #1e293b 0%, #0f172a 100%);
        border-right: 1px solid rgba(255,255,255,0.05);
    }}
    [data-testid="stSidebar"] * {{
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-weight: 500;
        text-shadow: 0 1px 2px rgba(0,0,0,0.4);
        opacity: 1 !important;
    }}
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] h4 {{
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-weight: 800 !important;
        letter-spacing: -0.2px;
        text-shadow: 0 2px 4px rgba(0,0,0,0.55);
    }}
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] .stCaption,
    [data-testid="stSidebar"] small,
    [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {{
        color: #f1f5f9 !important;
        -webkit-text-fill-color: #f1f5f9 !important;
        font-weight: 600 !important;
        font-size: 0.95rem !important;
        opacity: 1 !important;
    }}
    [data-testid="stSidebar"] .stRadio label,
    [data-testid="stSidebar"] .stCheckbox label,
    [data-testid="stSidebar"] .stSlider label,
    [data-testid="stSidebar"] .stSelectbox label,
    [data-testid="stSidebar"] .stFileUploader label {{
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-weight: 700 !important;
    }}
    [data-testid="stSidebar"] .stButton button {{ transition: var(--transition); }}
    [data-testid="stSidebar"] .stButton button:hover {{
        transform: translateY(-1px);
        box-shadow: 0 6px 14px rgba(46,117,182,0.35);
    }}

    /* TABS */
    .stTabs [data-baseweb="tab-list"] {{
        gap: 8px;
        border-bottom: 1px solid var(--border);
    }}
    .stTabs [data-baseweb="tab"] {{
        background: var(--surface);
        color: var(--text) !important;
        border-radius: 10px 10px 0 0;
        padding: 10px 20px; font-weight: 600;
        border: 1px solid var(--border);
        border-bottom: none;
        transition: var(--transition);
    }}
    .stTabs [data-baseweb="tab"]:hover {{
        background: var(--surface-hover);
        color: var(--text) !important;
        transform: translateY(-2px);
    }}
    .stTabs [aria-selected="true"] {{
        background: linear-gradient(135deg, var(--brand-1), var(--brand-2)) !important;
        color: #ffffff !important;
        box-shadow: var(--shadow-md);
    }}
    .stTabs [aria-selected="true"] * {{ color: #ffffff !important; -webkit-text-fill-color: #ffffff !important; }}

    /* BUTTONS */
    .stButton > button, .stDownloadButton > button {{
        border-radius: 10px;
        font-weight: 700;
        transition: var(--transition);
        border: 1px solid var(--border);
        color: var(--text) !important;
        -webkit-text-fill-color: var(--text) !important;
        background: var(--surface);
    }}
    .stButton > button *, .stDownloadButton > button * {{
        color: inherit !important;
        -webkit-text-fill-color: inherit !important;
        font-weight: inherit !important;
    }}
    .stButton > button:hover, .stDownloadButton > button:hover {{
        transform: translateY(-2px);
        box-shadow: var(--shadow-md);
        border-color: var(--brand-1);
        color: var(--brand-1) !important;
        -webkit-text-fill-color: var(--brand-1) !important;
    }}
    .stButton > button[kind="primary"] {{
        background: linear-gradient(135deg, #2563eb 0%, #6d28d9 100%) !important;
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        border: none;
        font-weight: 800 !important;
        text-shadow: 0 1px 3px rgba(0,0,0,0.65);
        box-shadow: 0 10px 24px rgba(46,117,182,0.30);
    }}
    .stButton > button[kind="primary"] *,
    .stButton > button[kind="primary"] p,
    .stButton > button[kind="primary"] span {{
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
        font-weight: 800 !important;
        text-shadow: 0 1px 3px rgba(0,0,0,0.65);
    }}
    .stButton > button[kind="primary"]:hover {{
        filter: brightness(1.08);
        color: #ffffff !important;
        -webkit-text-fill-color: #ffffff !important;
    }}

    /* CARD */
    .card {{
        background: var(--surface);
        color: var(--text);
        padding: 1.5rem;
        border-radius: var(--radius-md);
        border: 1px solid var(--border);
        box-shadow: var(--shadow-sm);
        margin-bottom: 1rem;
        transition: var(--transition);
    }}
    .card * {{ color: var(--text); }}
    .card:hover {{
        transform: translateY(-3px);
        box-shadow: var(--shadow-md);
        border-color: var(--brand-1);
    }}

    /* INPUTS */
    .stTextInput input, .stTextArea textarea, .stNumberInput input,
    .stSelectbox div[data-baseweb="select"] > div {{
        border-radius: 10px !important;
        color: var(--text) !important;
        transition: var(--transition);
    }}
    .stTextInput input:focus, .stTextArea textarea:focus {{
        border-color: var(--brand-1) !important;
        box-shadow: 0 0 0 3px rgba(46,117,182,0.15) !important;
    }}
    [data-testid="stExpander"] details summary {{
        border-radius: 10px;
        color: var(--text) !important;
        transition: var(--transition);
    }}
    [data-testid="stExpander"] details summary:hover {{ background: var(--surface-hover); }}

    [data-testid="stDataFrame"] {{
        border-radius: var(--radius-md);
        overflow: hidden;
        box-shadow: var(--shadow-sm);
    }}

    .footer {{
        text-align: center; color: var(--text-muted);
        padding: 1.5rem 0;
        border-top: 1px solid var(--border);
        margin-top: 2.5rem; font-size: 0.9rem;
    }}
    .footer a {{ color: var(--brand-1); text-decoration: none; }}
    .footer a:hover {{ color: var(--brand-2); }}

    .main .block-container > div {{ animation: fadeInUp .5s ease-out; }}
    @keyframes fadeInUp {{
        from {{ opacity: 0; transform: translateY(8px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}

    ::-webkit-scrollbar {{ width: 10px; height: 10px; }}
    ::-webkit-scrollbar-track {{ background: transparent; }}
    ::-webkit-scrollbar-thumb {{ background: var(--border); border-radius: 10px; }}
    ::-webkit-scrollbar-thumb:hover {{ background: var(--brand-1); }}
</style>
""", unsafe_allow_html=True)

# ── ICON CSS — radio, tabs, button, slider ──────────────────────
st.markdown("""
<style>
    /* SIDEBAR RADIO NAV — icon per item via ::before */
    [data-testid="stSidebar"] div[role="radiogroup"] label p::before {
        content: '';
        display: inline-block;
        width: 15px; height: 15px;
        vertical-align: -3px;
        margin-right: 8px;
        background-color: currentColor;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(1) p::before {
        -webkit-mask: url(https://unpkg.com/lucide-static@latest/icons/home.svg) no-repeat center/contain;
        mask:         url(https://unpkg.com/lucide-static@latest/icons/home.svg) no-repeat center/contain;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(2) p::before {
        -webkit-mask: url(https://unpkg.com/lucide-static@latest/icons/bar-chart-3.svg) no-repeat center/contain;
        mask:         url(https://unpkg.com/lucide-static@latest/icons/bar-chart-3.svg) no-repeat center/contain;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(3) p::before {
        -webkit-mask: url(https://unpkg.com/lucide-static@latest/icons/sparkles.svg) no-repeat center/contain;
        mask:         url(https://unpkg.com/lucide-static@latest/icons/sparkles.svg) no-repeat center/contain;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(4) p::before {
        -webkit-mask: url(https://unpkg.com/lucide-static@latest/icons/flask-conical.svg) no-repeat center/contain;
        mask:         url(https://unpkg.com/lucide-static@latest/icons/flask-conical.svg) no-repeat center/contain;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(5) p::before {
        -webkit-mask: url(https://unpkg.com/lucide-static@latest/icons/git-compare.svg) no-repeat center/contain;
        mask:         url(https://unpkg.com/lucide-static@latest/icons/git-compare.svg) no-repeat center/contain;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(6) p::before {
        -webkit-mask: url(https://unpkg.com/lucide-static@latest/icons/zap.svg) no-repeat center/contain;
        mask:         url(https://unpkg.com/lucide-static@latest/icons/zap.svg) no-repeat center/contain;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(7) p::before {
        -webkit-mask: url(https://unpkg.com/lucide-static@latest/icons/search.svg) no-repeat center/contain;
        mask:         url(https://unpkg.com/lucide-static@latest/icons/search.svg) no-repeat center/contain;
    }
    [data-testid="stSidebar"] div[role="radiogroup"] label:nth-child(8) p::before {
        -webkit-mask: url(https://unpkg.com/lucide-static@latest/icons/book-open.svg) no-repeat center/contain;
        mask:         url(https://unpkg.com/lucide-static@latest/icons/book-open.svg) no-repeat center/contain;
    }

    /* TABS EDA — icon per tab via ::before */
    .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"] p::before {
        content: '';
        display: inline-block;
        width: 14px; height: 14px;
        vertical-align: -2px;
        margin-right: 7px;
        background-color: currentColor;
    }
    .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(1) p::before {
        -webkit-mask: url(https://unpkg.com/lucide-static@latest/icons/table-2.svg) no-repeat center/contain;
        mask:         url(https://unpkg.com/lucide-static@latest/icons/table-2.svg) no-repeat center/contain;
    }
    .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(2) p::before {
        -webkit-mask: url(https://unpkg.com/lucide-static@latest/icons/pie-chart.svg) no-repeat center/contain;
        mask:         url(https://unpkg.com/lucide-static@latest/icons/pie-chart.svg) no-repeat center/contain;
    }
    .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(3) p::before {
        -webkit-mask: url(https://unpkg.com/lucide-static@latest/icons/ruler.svg) no-repeat center/contain;
        mask:         url(https://unpkg.com/lucide-static@latest/icons/ruler.svg) no-repeat center/contain;
    }
    .stTabs [data-baseweb="tab-list"] [data-baseweb="tab"]:nth-child(4) p::before {
        -webkit-mask: url(https://unpkg.com/lucide-static@latest/icons/hash.svg) no-repeat center/contain;
        mask:         url(https://unpkg.com/lucide-static@latest/icons/hash.svg) no-repeat center/contain;
    }

    /* PRIMARY BUTTON — icon di kiri */
    .stButton > button[kind="primary"]::before {
        content: '';
        display: inline-block;
        width: 16px; height: 16px;
        vertical-align: -3px;
        margin-right: 8px;
        background-color: #ffffff;
        -webkit-mask: url(https://unpkg.com/lucide-static@latest/icons/cpu.svg) no-repeat center/contain;
        mask:         url(https://unpkg.com/lucide-static@latest/icons/cpu.svg) no-repeat center/contain;
    }

    /* SLIDER — icon di depan label */
    [data-testid="stSlider"] label {
        display: flex !important;
        align-items: center;
        gap: 7px;
    }
    [data-testid="stSlider"] label::before {
        content: '';
        display: inline-block;
        width: 15px; height: 15px;
        flex-shrink: 0;
        background-color: currentColor;
        -webkit-mask: url(https://unpkg.com/lucide-static@latest/icons/sliders-horizontal.svg) no-repeat center/contain;
        mask:         url(https://unpkg.com/lucide-static@latest/icons/sliders-horizontal.svg) no-repeat center/contain;
    }
</style>
""", unsafe_allow_html=True)


# ───────────────────────────────────────────────────────────────────────────
# KONSTANTA
# ───────────────────────────────────────────────────────────────────────────
TOP10 = ['earn', 'acq', 'crude', 'trade', 'money-fx',
         'interest', 'ship', 'sugar', 'coffee', 'gold']

PALETTE = ['#2E75B6', '#ED7D31', '#70AD47', '#C00000',
           '#7030A0', '#FFC000', '#00B0F0', '#595959',
           '#FF6B6B', '#4ECDC4']

# Plotly template adaptif (transparan, mengikuti warna teks tema Streamlit)
import plotly.io as _pio
_pio.templates["lovable"] = go.layout.Template(
    layout=dict(
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Inter, Segoe UI, sans-serif", color=("#e2e8f0" if _IS_DARK else "#0f172a")),
        colorway=PALETTE,
        xaxis=dict(gridcolor="rgba(148,163,184,0.18)", zerolinecolor="rgba(148,163,184,0.25)"),
        yaxis=dict(gridcolor="rgba(148,163,184,0.18)", zerolinecolor="rgba(148,163,184,0.25)"),
        hoverlabel=dict(font_size=12, font_family="Inter"),
        margin=dict(t=40, b=40, l=40, r=20),
    )
)
_pio.templates.default = "lovable"

# ───────────────────────────────────────────────────────────────────────────
# NLTK
# ───────────────────────────────────────────────────────────────────────────
@st.cache_resource
def init_nltk():
    nltk.download('stopwords', quiet=True)
    return set(stopwords.words('english')), PorterStemmer()

STOPWORDS, STEMMER = init_nltk()

# ───────────────────────────────────────────────────────────────────────────
# HELPERS
# ───────────────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def load_csv(file) -> pd.DataFrame:
    return pd.read_csv(file)

def parse_topics(t):
    if pd.isna(t):
        return []
    if isinstance(t, list):
        return t
    try:
        return ast.literal_eval(t)
    except Exception:
        return []

@st.cache_data(show_spinner=False)
def preprocess_text(text: str) -> str:
    """
    Pipeline identik dengan notebook Cell 7:
    1. Cleaning  - hapus karakter non-alfabet, normalkan spasi
    2. Case Fold - lowercase
    3. Stopword  - hapus stopword NLTK + len(w) > 2  (sesuai notebook)
    4. Stemming  - Porter Stemmer
    """
    # Step 1: Cleaning
    text = re.sub(r'[^a-zA-Z\s]', ' ', str(text))
    text = re.sub(r'\s+', ' ', text).strip()
    # Step 2: Case Folding
    text = text.lower()
    # Step 3: Stopword Removal (len > 2 sesuai notebook)
    tokens = [w for w in text.split() if w not in STOPWORDS and len(w) > 2]
    # Step 4: Stemming
    tokens = [STEMMER.stem(w) for w in tokens]
    return ' '.join(tokens)

@st.cache_data(show_spinner="Memproses dataset...")
def prepare_dataset(train_df: pd.DataFrame, test_df: pd.DataFrame, top_n=10):
    top_classes = TOP10[:top_n]

    def filter_top(df):
        df = df.copy()
        df['topics_list'] = df['topics'].apply(parse_topics)
        # Sesuai notebook: iterasi topik URUTAN DOKUMEN, ambil yang PERTAMA ada di top_classes.
        # Contoh: topics=['wheat','earn'] → primary='earn' (wheat bukan top-class)
        #         topics=['crude','earn'] → primary='crude' (crude lebih dahulu di dokumen)
        # Berbeda dari: next(c for c in TOP10 if c in lst) → biased ke earn/acq
        # Berbeda dari: lst[0]                             → exclude dokumen yg lst[0] bukan top-class
        df['primary_topic'] = df['topics_list'].apply(
            lambda lst: next((t for t in lst if t in top_classes), None)
        )
        df = df[df['primary_topic'].notna()].reset_index(drop=True)
        df['clean_text'] = df['text'].astype(str).apply(preprocess_text)
        df = df[df['clean_text'].str.len() > 0].reset_index(drop=True)
        return df

    return filter_top(train_df), filter_top(test_df), top_classes

@st.cache_data(show_spinner=False)
def vectorize(_train_text, _test_text, method, max_features=10000, ngram=(1, 1)):
    if method == "BoW":
        vec = CountVectorizer(max_features=max_features, ngram_range=(1, 1))
    elif method == "N-gram":
        vec = CountVectorizer(max_features=max_features, ngram_range=ngram)
    else:  # TF-IDF
        vec = TfidfVectorizer(max_features=max_features, ngram_range=(1, 2),
                              sublinear_tf=True)
    Xtr = vec.fit_transform(_train_text)
    Xte = vec.transform(_test_text)
    return Xtr, Xte, vec

def train_dt(X_train, y_train, X_test, y_test, max_depth=None, min_samples_split=2):
    """Sesuai laporan: criterion=gini, splitter=best, max_depth=None (bebas tumbuh)."""
    t0 = time.time()
    clf = DecisionTreeClassifier(
        criterion='gini', splitter='best',
        max_depth=max_depth, min_samples_split=min_samples_split,
        random_state=42,
    )
    clf.fit(X_train, y_train)
    train_t = time.time() - t0

    t0 = time.time()
    pred = clf.predict(X_test)
    pred_t = time.time() - t0

    acc = accuracy_score(y_test, pred)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(
        y_test, pred, average='weighted', zero_division=0
    )
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(
        y_test, pred, average='macro', zero_division=0
    )
    return clf, pred, dict(
        accuracy=acc,
        precision=p_w, recall=r_w, f1=f1_w,
        precision_macro=p_m, recall_macro=r_m, f1_macro=f1_m,
        train_time=train_t, predict_time=pred_t,
        tree_depth=clf.get_depth(), tree_leaves=clf.get_n_leaves(),
    )

def train_linearsvc(X_train, y_train, X_test, y_test):
    """LinearSVC sesuai Tahap 3-4 laporan/notebook."""
    t0 = time.time()
    clf = LinearSVC(max_iter=2000, random_state=42)
    clf.fit(X_train, y_train)
    train_t = time.time() - t0
    t0 = time.time()
    pred = clf.predict(X_test)
    pred_t = time.time() - t0
    acc = accuracy_score(y_test, pred)
    p_w, r_w, f1_w, _ = precision_recall_fscore_support(y_test, pred, average='weighted', zero_division=0)
    p_m, r_m, f1_m, _ = precision_recall_fscore_support(y_test, pred, average='macro', zero_division=0)
    return pred, dict(
        accuracy=acc,
        precision_macro=p_m, recall_macro=r_m, f1_macro=f1_m,
        precision_weighted=p_w, recall_weighted=r_w, f1_weighted=f1_w,
        train_time=train_t, predict_time=pred_t,
    )

def df_to_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode('utf-8')

# ───────────────────────────────────────────────────────────────────────────
# PENINGKATAN MODEL — Balancing & Text Representation (Embeddings)
# ───────────────────────────────────────────────────────────────────────────
import random

def class_distribution(y):
    """Hitung distribusi kelas dan rasio imbalance (max/min)."""
    cnt = Counter(y)
    df = pd.DataFrame({'kelas': list(cnt.keys()), 'jumlah': list(cnt.values())})
    df = df.sort_values('jumlah', ascending=False).reset_index(drop=True)
    ratio = df['jumlah'].max() / max(df['jumlah'].min(), 1)
    return df, ratio

# ---- BALANCING METHODS -----------------------------------------------------
def balance_resample(X, y, strategy="oversample", random_state=42):
    """Resample SETELAH representasi teks (sparse matrix-friendly via imblearn)."""
    try:
        from imblearn.over_sampling import RandomOverSampler, SMOTE
        from imblearn.under_sampling import RandomUnderSampler
    except ImportError:
        raise RuntimeError("Modul `imbalanced-learn` belum terpasang. Jalankan: pip install imbalanced-learn")
    if strategy == "oversample":
        sampler = RandomOverSampler(random_state=random_state)
    elif strategy == "undersample":
        sampler = RandomUnderSampler(random_state=random_state)
    elif strategy == "smote":
        sampler = SMOTE(random_state=random_state, k_neighbors=3)
    else:
        raise ValueError(f"Unknown strategy {strategy}")
    X_res, y_res = sampler.fit_resample(X, y)
    return X_res, y_res

# ---- DATA AUGMENTATION (text-level, EDA-style) -----------------------------
_AUG_STOP = None
def _aug_stopwords():
    global _AUG_STOP
    if _AUG_STOP is None:
        try:
            _AUG_STOP = set(stopwords.words('english'))
        except LookupError:
            nltk.download('stopwords', quiet=True)
            _AUG_STOP = set(stopwords.words('english'))
    return _AUG_STOP

def aug_random_swap(text, n=1):
    words = text.split()
    if len(words) < 2: return text
    for _ in range(n):
        i, j = random.sample(range(len(words)), 2)
        words[i], words[j] = words[j], words[i]
    return " ".join(words)

def aug_random_deletion(text, p=0.1):
    words = text.split()
    if len(words) < 2: return text
    kept = [w for w in words if random.random() > p]
    return " ".join(kept) if kept else random.choice(words)

def aug_synonym_replacement(text, n=1):
    """Synonym replacement via WordNet (NLTK)."""
    try:
        from nltk.corpus import wordnet
        wordnet.synsets("test")
    except LookupError:
        nltk.download('wordnet', quiet=True)
        nltk.download('omw-1.4', quiet=True)
        from nltk.corpus import wordnet
    words = text.split()
    sw = _aug_stopwords()
    candidates = [i for i, w in enumerate(words) if w.lower() not in sw and len(w) > 3]
    random.shuffle(candidates)
    replaced = 0
    for idx in candidates:
        syns = wordnet.synsets(words[idx])
        lemmas = {l.name().replace("_", " ").lower() for s in syns for l in s.lemmas()}
        lemmas.discard(words[idx].lower())
        if lemmas:
            words[idx] = random.choice(list(lemmas))
            replaced += 1
        if replaced >= n: break
    return " ".join(words)

def augment_text(text, technique="swap"):
    if technique == "swap":     return aug_random_swap(text, n=2)
    if technique == "deletion": return aug_random_deletion(text, p=0.15)
    if technique == "synonym":  return aug_synonym_replacement(text, n=2)
    return text

def balance_via_augmentation(texts, labels, technique="swap", random_state=42):
    """Oversample minoritas dengan teknik augmentasi sampai = ukuran mayoritas."""
    random.seed(random_state)
    cnt = Counter(labels)
    target = max(cnt.values())
    texts, labels = list(texts), list(labels)
    by_class = {c: [t for t, y in zip(texts, labels) if y == c] for c in cnt}
    new_t, new_y = list(texts), list(labels)
    for c, samples in by_class.items():
        need = target - len(samples)
        for _ in range(need):
            base = random.choice(samples)
            new_t.append(augment_text(base, technique))
            new_y.append(c)
    return new_t, new_y

# ---- BACK TRANSLATION ------------------------------------------------------
def back_translate(text, intermediate="fr"):
    """English → intermediate → English (butuh internet, deep-translator)."""
    try:
        from deep_translator import GoogleTranslator
    except ImportError:
        raise RuntimeError("Modul `deep-translator` belum terpasang. Jalankan: pip install deep-translator")
    if not text.strip(): return text
    txt = text[:4500]  # batas API
    mid = GoogleTranslator(source='en', target=intermediate).translate(txt)
    back = GoogleTranslator(source=intermediate, target='en').translate(mid or "")
    return back or text

def balance_via_back_translation(texts, labels, intermediate="fr", random_state=42, max_per_class=None):
    """Augmentasi back-translation untuk kelas minoritas. LAMBAT (panggil API)."""
    random.seed(random_state)
    cnt = Counter(labels)
    target = max(cnt.values())
    if max_per_class: target = min(target, max_per_class)
    texts, labels = list(texts), list(labels)
    by_class = {c: [t for t, y in zip(texts, labels) if y == c] for c in cnt}
    new_t, new_y = list(texts), list(labels)
    for c, samples in by_class.items():
        need = max(target - len(samples), 0)
        for _ in range(need):
            base = random.choice(samples)
            try:
                new_t.append(back_translate(base))
            except Exception:
                new_t.append(augment_text(base, "swap"))
            new_y.append(c)
    return new_t, new_y

# ---- WORD EMBEDDINGS (non-contextual) --------------------------------------
@st.cache_resource(show_spinner=False)
def train_word2vec(corpus_tokens, vector_size=100, window=5, min_count=2, sg=0):
    from gensim.models import Word2Vec
    return Word2Vec(sentences=corpus_tokens, vector_size=vector_size,
                    window=window, min_count=min_count, sg=sg, workers=2, epochs=10)

@st.cache_resource(show_spinner=False)
def train_fasttext(corpus_tokens, vector_size=100, window=5, min_count=2):
    from gensim.models import FastText
    return FastText(sentences=corpus_tokens, vector_size=vector_size,
                    window=window, min_count=min_count, workers=2, epochs=10)

@st.cache_resource(show_spinner=False)
def load_glove(name="glove-wiki-gigaword-100"):
    import gensim.downloader as api
    return api.load(name)

def docs_to_mean_vec(corpus_tokens, kv, dim):
    """Rata-rata vektor kata per dokumen (handles KeyedVectors atau Model)."""
    import numpy as _np
    vecs = []
    get_vec = (lambda w: kv[w]) if hasattr(kv, "__getitem__") and not hasattr(kv, "wv") else (lambda w: kv.wv[w])
    has_word = (lambda w: w in kv) if not hasattr(kv, "wv") else (lambda w: w in kv.wv)
    for toks in corpus_tokens:
        v = [get_vec(w) for w in toks if has_word(w)]
        vecs.append(_np.mean(v, axis=0) if v else _np.zeros(dim))
    return _np.vstack(vecs)

# ---- CONTEXTUAL EMBEDDINGS (BERT / RoBERTa / DeBERTa) ---------------------
@st.cache_resource(show_spinner=False)
def load_contextual_model(model_name):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)

CONTEXTUAL_MODELS = {
    "BERT":    "sentence-transformers/all-MiniLM-L6-v2",
    "RoBERTa": "sentence-transformers/all-distilroberta-v1",
    "DeBERTa": "sentence-transformers/all-mpnet-base-v2",
}



# ───────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* ====== SIDEBAR REDESIGN — dark gradient + brand accents ====== */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #16203F 0%, #2A1B4F 100%) !important;
    border-top-right-radius: 38px;
    border-bottom-right-radius: 38px;
    box-shadow: 10px 0 40px rgba(10, 15, 45, 0.45);
    padding: 0 .25rem 1.25rem .25rem;
}
[data-testid="stSidebar"] > div:first-child {
    background: transparent !important;
    border-radius: inherit;
}
[data-testid="stSidebar"], [data-testid="stSidebar"] * {
    color: #F1F5FF !important;
}
[data-testid="stSidebar"] hr {
    border-color: rgba(255,255,255,0.10) !important;
    margin: .8rem 0 !important;
}

/* ── Profile block (avatar + nama) ── */
[data-testid="stSidebar"] img[src^="data:image"],
[data-testid="stSidebar"] img[src*="ui-avatars"] {
    border: 3px solid transparent !important;
    background:
        linear-gradient(#16203F, #16203F) padding-box,
        linear-gradient(135deg, #5BA3E8, #B584D9) border-box !important;
    box-shadow: 0 8px 24px rgba(91,163,232,0.25) !important;
    padding: 3px;
}

/* ── Section heading kecil bergaya UPPERCASE ── */
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    font-size: 12px !important;
    text-transform: uppercase;
    letter-spacing: 2px !important;
    color: #B584D9 !important;
    font-weight: 700 !important;
    margin: 1.2rem .5rem .4rem !important;
}
/* Tetap putih utk heading utama "NLP Dashboard" yg pakai gradient bullet */
[data-testid="stSidebar"] h2 span:last-child {
    color: #F1F5FF !important;
    font-size: 18px !important;
    text-transform: none;
    letter-spacing: .3px !important;
}

/* ====== NAV RADIO sebagai pill items ====== */
[data-testid="stSidebar"] div[role="radiogroup"] {
    gap: 6px !important;
    display: flex; flex-direction: column;
    padding: 0 .25rem;
}
[data-testid="stSidebar"] div[role="radiogroup"] label {
    background: transparent;
    border-radius: 14px;
    padding: 12px 18px !important;
    margin: 0 !important;
    transition: background .2s ease, color .2s ease, transform .15s ease;
    cursor: pointer;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background: rgba(255,255,255,0.06);
}
/* Sembunyikan bullet bawaan radio */
[data-testid="stSidebar"] div[role="radiogroup"] label > div:first-child {
    display: none !important;
}
[data-testid="stSidebar"] div[role="radiogroup"] label p {
    font-weight: 600 !important;
    font-size: 13.5px !important;
    letter-spacing: 1.2px !important;
    text-transform: uppercase;
    color: #E8ECFB !important;
    display: flex; align-items: center;
}
/* Item AKTIF — pill terang melengkung ke kanan */
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {
    background: #F4F6FF !important;
    border-radius: 14px 28px 28px 14px;
    box-shadow: 0 10px 28px rgba(0,0,0,0.28),
                inset 0 0 0 1px rgba(91,163,232,0.25);
    transform: translateX(6px);
}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p {
    background: linear-gradient(135deg, #2E75B6, #7030A0);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p::before {
    background-color: #2E75B6 !important; /* icon mask color */
}

/* ====== Inputs di dalam sidebar ====== */
[data-testid="stSidebar"] .stFileUploader,
[data-testid="stSidebar"] .stCheckbox {
    background: rgba(255,255,255,0.05);
    border-radius: 12px;
    padding: 10px 12px;
}
[data-testid="stSidebar"] .stSlider [data-baseweb="slider"] div[role="slider"] {
    background: linear-gradient(135deg,#5BA3E8,#B584D9) !important;
    border-color: #ffffff !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] > div {
    background: rgba(255,255,255,0.92) !important;
    border-radius: 10px !important;
}
[data-testid="stSidebar"] [data-baseweb="select"] * {
    color: #2A1B4F !important;
}

/* ====== Footer "Visit site" card ====== */
.sidebar-cta {
    margin: 22px 8px 4px;
    padding: 16px 18px;
    border-radius: 22px;
    background: rgba(255,255,255,0.06);
    border: 1px solid rgba(255,255,255,0.12);
    backdrop-filter: blur(6px);
    display: flex; align-items: center; gap: 12px;
    transition: transform .2s ease, background .2s ease;
}
.sidebar-cta:hover { background: rgba(255,255,255,0.10); transform: translateY(-2px); }
.sidebar-cta .arrow {
    width: 34px; height: 34px; border-radius: 50%;
    background: linear-gradient(135deg,#5BA3E8,#B584D9);
    display: inline-flex; align-items: center; justify-content: center;
    box-shadow: 0 6px 16px rgba(91,163,232,0.35);
    flex-shrink: 0;
}
.sidebar-cta .arrow span {
    width: 16px; height: 16px; background-color: #ffffff;
    -webkit-mask: url(https://unpkg.com/lucide-static@latest/icons/arrow-up-right.svg) no-repeat center/contain;
    mask: url(https://unpkg.com/lucide-static@latest/icons/arrow-up-right.svg) no-repeat center/contain;
}
.sidebar-cta .label {
    font-size: 16px; font-weight: 700; letter-spacing: .3px;
    color: #F1F5FF;
}
.sidebar-cta a, .sidebar-cta a * { text-decoration: none !important; color: inherit !important; }

/* ====== FIX KONTRAS — File Uploader (dropzone putih) ====== */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"],
[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"],
[data-testid="stSidebar"] section[data-testid="stFileUploaderDropzone"] {
    background: #ffffff !important;
    border: 2px dashed #5BA3E8 !important;
    border-radius: 12px !important;
}
/* Semua teks & ikon di dalam dropzone putih → ungu gelap */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] *,
[data-testid="stSidebar"] [data-testid="stFileUploadDropzone"] *,
[data-testid="stSidebar"] .stFileUploader small,
[data-testid="stSidebar"] .stFileUploader span,
[data-testid="stSidebar"] .stFileUploader div[data-testid="stFileUploaderDropzoneInstructions"] *,
[data-testid="stSidebar"] .stFileUploader [data-testid="stFileUploaderFileName"] {
    color: #2A1B4F !important;
    -webkit-text-fill-color: #2A1B4F !important;
    fill: #2A1B4F !important;
}
/* Tombol "Browse files" / "Upload" di dalam dropzone */
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button,
[data-testid="stSidebar"] .stFileUploader button {
    background: linear-gradient(135deg,#2E75B6,#7030A0) !important;
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button *,
[data-testid="stSidebar"] .stFileUploader button * {
    color: #ffffff !important;
    -webkit-text-fill-color: #ffffff !important;
    fill: #ffffff !important;
}

/* ====== FIX KONTRAS — Selectbox (latar putih, teks gelap) ====== */
[data-testid="stSidebar"] [data-baseweb="select"] *,
[data-testid="stSidebar"] [data-baseweb="select"] input,
[data-testid="stSidebar"] [data-baseweb="select"] div[role="button"],
[data-testid="stSidebar"] [data-baseweb="select"] svg {
    color: #2A1B4F !important;
    -webkit-text-fill-color: #2A1B4F !important;
    fill: #2A1B4F !important;
}
/* Dropdown menu (popover) options */
[data-baseweb="popover"] [role="listbox"] *,
[data-baseweb="menu"] * {
    color: #2A1B4F !important;
    -webkit-text-fill-color: #2A1B4F !important;
}

/* ====== FIX KONTRAS — Item nav aktif (BERANDA, dll) ====== */
/* Hilangkan gradient-clip yang membuat teks tipis, pakai warna solid */
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p,
[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) p * {
    background: none !important;
    -webkit-background-clip: initial !important;
    background-clip: initial !important;
    -webkit-text-fill-color: #2E75B6 !important;
    color: #2E75B6 !important;
    font-weight: 800 !important;
}

/* ====== FIX KONTRAS — Caption / label dengan inline code badges ====== */
[data-testid="stSidebar"] code,
[data-testid="stSidebar"] .stMarkdown code {
    background: #F4F6FF !important;
    color: #2A1B4F !important;
    -webkit-text-fill-color: #2A1B4F !important;
    padding: 1px 6px !important;
    border-radius: 6px !important;
    font-weight: 700 !important;
}

/* Label uploader & selectbox di sidebar tetap putih terang */
[data-testid="stSidebar"] .stFileUploader > label,
[data-testid="stSidebar"] .stFileUploader > label *,
[data-testid="stSidebar"] .stSelectbox > label,
[data-testid="stSidebar"] .stSelectbox > label * {
    color: #F1F5FF !important;
    -webkit-text-fill-color: #F1F5FF !important;
    font-weight: 600 !important;
}
</style>
""", unsafe_allow_html=True)

# SIDEBAR — NAVIGASI & DATA
# ───────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""<h2 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:40px;height:40px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:26px;height:26px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/newspaper.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/newspaper.svg) no-repeat center/contain;"></span></span><span> NLP Dashboard</span></h2>""", unsafe_allow_html=True)
    st.caption("Reuters-21578 · Decision Tree")
    st.divider()

    page = st.radio(
        "Navigasi",
        ["Beranda", "EDA", "Pre-processing",
         "Eksperimen Model", "Perbandingan",
         "Peningkatan Model", "Prediksi", "Panduan"],
        label_visibility="collapsed",
    )

    st.divider()
    st.markdown("""<h3 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:22px;height:22px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/folder.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/folder.svg) no-repeat center/contain;"></span></span><span>Dataset</span></h3>""", unsafe_allow_html=True)
    train_file = st.file_uploader("ModApte_train.csv", type=["csv"], key="tr")
    test_file = st.file_uploader("ModApte_test.csv", type=["csv"], key="te")

    use_demo = st.checkbox("Gunakan data demo (jika belum upload)", value=False)

    st.divider()
    st.markdown("""<h3 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:22px;height:22px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/settings.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/settings.svg) no-repeat center/contain;"></span></span><span>Parameter Global</span></h3>""", unsafe_allow_html=True)
    top_n = st.slider("Top-N kelas", 3, 10, 10)
    max_features = st.select_slider(
        "Max features", options=[500, 1000, 2000, 3000, 5000, 8000, 10000],
        value=10000,
    )
    _depth_opts = ["None (Bebas — sesuai laporan)", 5, 10, 15, 20, 30, 40, 50]
    _depth_sel  = st.selectbox("Max depth tree", _depth_opts, index=0)
    max_depth   = None if _depth_sel == "None (Bebas — sesuai laporan)" else _depth_sel
    min_samples = st.slider("Min samples split", 2, 20, 2)

    st.divider()
    st.caption("Upload CSV ModApte (kolom `text` & `topics`).")

# ───────────────────────────────────────────────────────────────────────────
# LOAD DATA
# ───────────────────────────────────────────────────────────────────────────
train_df, test_df = None, None
if train_file and test_file:
    train_df, test_df = load_csv(train_file), load_csv(test_file)
elif use_demo:
    # demo kecil
    demo_train = pd.DataFrame({
        'text': [
            'shr profit earnings rose dlrs net income',
            'company acquired stake merger acquisition deal',
            'crude oil prices barrel opec output',
            'trade deficit exports imports tariff',
            'dollar yen currency forex money market',
        ] * 40,
        'topics': [
            "['earn']", "['acq']", "['crude']", "['trade']", "['money-fx']"
        ] * 40,
    })
    demo_test = demo_train.sample(40, random_state=1).reset_index(drop=True)
    train_df, test_df = demo_train, demo_test

ready = train_df is not None and test_df is not None

# ───────────────────────────────────────────────────────────────────────────
# HERO HEADER
# ───────────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <h1 style="display:flex;align-items:center;gap:16px;"><span style="display:inline-flex;align-items:center;justify-content:center;width:50px;height:50px;background:rgba(255,255,255,0.95);border-radius:50%;box-shadow:0 4px 18px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:28px;height:28px;background-color:#2E75B6;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/newspaper.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/newspaper.svg) no-repeat center/contain;"></span></span><span> NLP Dashboard — Reuters-21578</span></h1>
    <p>Klasifikasi teks berita menggunakan Decision Tree (CART) dengan 3 eksperimen feature extraction:
    <b>Bag-of-Words</b>, <b>N-gram</b>, dan <b>TF-IDF</b>.</p>
</div>
""", unsafe_allow_html=True)

if not ready:
    st.warning("Silakan upload `ModApte_train.csv` & `ModApte_test.csv` di sidebar, atau aktifkan **data demo**.")
    st.stop()

# Prepare
train_clean, test_clean, top_classes = prepare_dataset(train_df, test_df, top_n)

# ═══════════════════════════════════════════════════════════════════════════
# <span style="display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.25);vertical-align:-7px;margin-right:6px;flex-shrink:0;"><span style="display:inline-block;width:18px;height:18px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/home.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/home.svg) no-repeat center/contain;"></span></span> BERANDA
# ═══════════════════════════════════════════════════════════════════════════
if page == "Beranda":
    # ── Verifikasi jumlah dokumen vs laporan (5.701 train, 2.246 test) ──
    _tr_ok = abs(len(train_clean) - 5701) < 50
    _te_ok = abs(len(test_clean)  - 2246) < 50
    if _tr_ok and _te_ok:
        st.success(f"✅ Berhasil memuat dataset")
    else:
        st.warning(
            f"⚠️ Dataset ini merupadan data demo."
            f"\nPastikan file CSV yang diupload adalah dataset yang sesuai."
        )

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Train docs", f"{len(train_clean):,}", f"raw {len(train_df):,}")
    c2.metric("Test docs", f"{len(test_clean):,}", f"raw {len(test_df):,}")
    c3.metric("Kelas (Top-N)", len(top_classes))
    c4.metric("Avg. tokens/doc",
              f"{train_clean['clean_text'].str.split().str.len().mean():.0f}")

    st.markdown("""<h3 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:22px;height:22px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/target.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/target.svg) no-repeat center/contain;"></span></span><span> Tujuan Bisnis</span></h3>""", unsafe_allow_html=True)
    st.info(
        "Membangun **sistem klasifikasi otomatis** berita Reuters ke dalam kategori topik utama "
        "(earn, acq, crude, dll) untuk mempercepat **kategorisasi konten**, **monitoring pasar**, "
        "dan **rekomendasi berita** kepada analis keuangan."
    )

    col1, col2 = st.columns([2, 1])
    with col1:
        st.markdown("""<h3 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:22px;height:22px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/bar-chart-3.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/bar-chart-3.svg) no-repeat center/contain;"></span></span><span> Distribusi Kelas (Train)</span></h3>""", unsafe_allow_html=True)
        dist = train_clean['primary_topic'].value_counts().reindex(top_classes).fillna(0)
        fig = px.bar(
            x=dist.index, y=dist.values,
            color=dist.index, color_discrete_sequence=PALETTE,
            labels={'x': 'Kategori', 'y': 'Jumlah dokumen'},
        )
        fig.update_layout(showlegend=False, height=380, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("""<h3 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:22px;height:22px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/award.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/award.svg) no-repeat center/contain;"></span></span><span> Class Imbalance</span></h3>""", unsafe_allow_html=True)
        fig = px.pie(
            values=dist.values, names=dist.index,
            color_discrete_sequence=PALETTE, hole=0.5,
        )
        fig.update_layout(height=380, margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("""<h3 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:22px;height:22px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/compass.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/compass.svg) no-repeat center/contain;"></span></span><span> Metode</span></h3>""", unsafe_allow_html=True)
    st.markdown("""
    1. **Tahap 1** — Original Data + EDA  
    2. **Tahap 2** — Target Data (Top-N kelas)  
    3. **Tahap 3-4** — Pre-processing & Transformasi (BoW / N-gram / TF-IDF)  
    4. **Tahap 5** — Data Mining: 3 eksperimen Decision Tree  
    5. **Tahap 6** — Knowledge Interpretation & Perbandingan
    """, unsafe_allow_html=True)

# ═══════════════════════════════════════════════════════════════════════════
# 📊 EDA
# ═══════════════════════════════════════════════════════════════════════════
elif page == "EDA":
    st.markdown("""<h2 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:40px;height:40px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:26px;height:26px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/bar-chart-3.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/bar-chart-3.svg) no-repeat center/contain;"></span></span><span> Exploratory Data Analysis</span></h2>""", unsafe_allow_html=True)

    tab1, tab2, tab3, tab4 = st.tabs(["Sample Data", "Distribusi",
                                       "Panjang Dokumen", "Top Tokens"])

    with tab1:
        st.markdown("#### 10 baris pertama (train)")
        st.dataframe(train_df.head(10), use_container_width=True, height=350)
        st.caption(f"Shape: {train_df.shape} | Kolom: {list(train_df.columns)}")

    with tab2:
        col1, col2 = st.columns(2)
        for col, df, title in [(col1, train_clean, "Train"), (col2, test_clean, "Test")]:
            with col:
                st.markdown(f"#### Distribusi — {title}")
                d = df['primary_topic'].value_counts().reindex(top_classes).fillna(0)
                fig = px.bar(x=d.values, y=d.index, orientation='h',
                             color=d.index, color_discrete_sequence=PALETTE,
                             labels={'x': 'Count', 'y': ''})
                fig.update_layout(showlegend=False, height=400, margin=dict(t=20))
                st.plotly_chart(fig, use_container_width=True)

    with tab3:
        train_clean['n_tokens'] = train_clean['clean_text'].str.split().str.len()
        fig = px.histogram(train_clean, x='n_tokens', nbins=50,
                           color='primary_topic',
                           color_discrete_sequence=PALETTE,
                           labels={'n_tokens': 'Jumlah token / dokumen'})
        fig.update_layout(height=420)
        st.plotly_chart(fig, use_container_width=True)

        st.markdown("#### Statistik panjang per kelas")
        stats = train_clean.groupby('primary_topic')['n_tokens'].describe().round(1)
        st.dataframe(stats, use_container_width=True)

    with tab4:
        sel_topic = st.selectbox("Pilih kategori:", top_classes)
        subset = train_clean[train_clean['primary_topic'] == sel_topic]
        all_words = ' '.join(subset['clean_text']).split()
        common = Counter(all_words).most_common(20)
        if common:
            words, counts = zip(*common)
            fig = px.bar(x=list(counts), y=list(words), orientation='h',
                         color_discrete_sequence=['#2E75B6'],
                         labels={'x': 'Frekuensi', 'y': 'Token'})
            fig.update_layout(height=500, yaxis=dict(autorange='reversed'))
            st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════════════════════════
# <span style="display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.25);vertical-align:-7px;margin-right:6px;flex-shrink:0;"><span style="display:inline-block;width:18px;height:18px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/sparkles.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/sparkles.svg) no-repeat center/contain;"></span></span> PRE-PROCESSING
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Pre-processing":
    st.markdown("""<h2 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:40px;height:40px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:26px;height:26px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/sparkles.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/sparkles.svg) no-repeat center/contain;"></span></span><span> Pre-processing Pipeline</span></h2>""", unsafe_allow_html=True)

    st.markdown("""
    Tahapan: **lowercase → hapus non-alfabet → tokenisasi → stopword removal → Porter stemming**.
    """, unsafe_allow_html=True)

    st.markdown('''<div style="display:flex;align-items:center;gap:10px;margin:.5rem 0 .25rem;">'''
        '''<span style="display:inline-flex;align-items:center;justify-content:center;width:28px;height:28px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 2px 8px rgba(0,0,0,.25);">'''
        '''<span style="display:inline-block;width:16px;height:16px;background:#fff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/file-text.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/file-text.svg) no-repeat center/contain;"></span></span>'''
        '''<span style="font-weight:600;color:var(--text);">Pilih dokumen (index train)</span></div>''', unsafe_allow_html=True)
    idx = st.slider("Pilih dokumen (index train)", 0, len(train_df) - 1, 0, label_visibility="collapsed")
    raw = train_df['text'].iloc[idx]
    clean = preprocess_text(raw)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown("""<h4 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:32px;height:32px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:18px;height:18px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/pencil.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/pencil.svg) no-repeat center/contain;"></span></span><span> Teks Asli</span></h4>""", unsafe_allow_html=True)
        st.text_area(" ", raw[:1500], height=300, label_visibility="collapsed")
    with col2:
        st.markdown("""<h4 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:32px;height:32px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:18px;height:18px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/sparkles.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/sparkles.svg) no-repeat center/contain;"></span></span><span> Teks Bersih</span></h4>""", unsafe_allow_html=True)
        st.text_area("  ", clean[:1500], height=300, label_visibility="collapsed")

    c1, c2, c3 = st.columns(3)
    c1.metric("Karakter asli", len(str(raw)))
    c2.metric("Karakter bersih", len(clean))
    c3.metric("Token bersih", len(clean.split()))

    st.markdown("""<h3 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:22px;height:22px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/flask-conical.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/flask-conical.svg) no-repeat center/contain;"></span></span><span> Coba teks bebas</span></h3>""", unsafe_allow_html=True)
    user_text = st.text_area("Masukkan teks:", "Oil prices rose sharply after OPEC announced production cuts.")
    if user_text:
        st.code(preprocess_text(user_text), language="text")

# ═══════════════════════════════════════════════════════════════════════════
# <span style="display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.25);vertical-align:-7px;margin-right:6px;flex-shrink:0;"><span style="display:inline-block;width:18px;height:18px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/flask-conical.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/flask-conical.svg) no-repeat center/contain;"></span></span> EKSPERIMEN MODEL
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Eksperimen Model":
    st.markdown("""<h2 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:40px;height:40px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:26px;height:26px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/flask-conical.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/flask-conical.svg) no-repeat center/contain;"></span></span><span> Eksperimen Decision Tree</span></h2>""", unsafe_allow_html=True)

    method = st.radio("Pilih metode feature extraction:",
                      ["BoW", "N-gram", "TF-IDF"], horizontal=True)

    ngram = (1, 2)
    if method == "N-gram":
        n_max = st.slider("N-gram max", 2, 3, 2)
        ngram = (1, n_max)

    # ── Tombol latih: hanya melatih & simpan ke session_state ──────────────
    if st.button("Latih Model", type="primary", use_container_width=True):
        with st.spinner("Vectorizing & training..."):
            Xtr, Xte, vec = vectorize(
                train_clean['clean_text'].tolist(),
                test_clean['clean_text'].tolist(),
                method, max_features, ngram,
            )
            clf, pred, m = train_dt(
                Xtr, train_clean['primary_topic'],
                Xte, test_clean['primary_topic'],
                max_depth, min_samples,
            )
        st.session_state[f'result_{method}'] = {
            'metrics': m, 'pred': pred, 'clf': clf, 'vec': vec,
        }
        st.success(f"✅ Model **{method}** berhasil dilatih dan tersimpan!")

    # ── Tampilkan hasil dari session_state (persisten antar halaman) ─────────
    _key = f'result_{method}'
    if _key in st.session_state:
        _r   = st.session_state[_key]
        m    = dict(_r['metrics'])
        # (Override metrik laporan dihapus — gunakan hasil aktual dari dataset yang dimuat)
        pred = _r['pred']
        clf  = _r['clf']
        vec  = _r['vec']

        st.divider()

        # Badge status tersimpan
        st.markdown(
            f'''<div style="display:inline-flex;align-items:center;gap:8px;
            background:linear-gradient(135deg,#2E75B6,#7030A0);
            color:#fff;padding:6px 16px;border-radius:20px;font-size:.85rem;
            font-weight:600;margin-bottom:1rem;">
            <span style="display:inline-block;width:13px;height:13px;background:#fff;
            -webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/check-circle.svg) no-repeat center/contain;
            mask:url(https://unpkg.com/lucide-static@latest/icons/check-circle.svg) no-repeat center/contain;"></span>
            Model {method} tersimpan</div>''',
            unsafe_allow_html=True,
        )

        c1, c2 = st.columns(2)
        c1.metric("Accuracy", f"{m['accuracy']:.4f}")
        c2.metric("F1 Macro ★", f"{m['f1_macro']:.4f}",
                  help="Metrik utama per laporan — tidak bias terhadap kelas mayoritas")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Precision (Macro)", f"{m['precision_macro']:.4f}")
        c2.metric("Recall (Macro)",    f"{m['recall_macro']:.4f}")
        c3.metric("Precision (W)",     f"{m['precision']:.4f}")
        c4.metric("F1 (Weighted)",     f"{m['f1']:.4f}")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Train time",   f"{m['train_time']:.2f} s")
        c2.metric("Predict time", f"{m['predict_time']:.3f} s")
        c3.metric("Tree Depth",   f"{m['tree_depth']} level")
        c4.metric("Tree Leaves",  f"{m['tree_leaves']} daun")

        # Confusion matrix
        st.markdown("""<h3 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:22px;height:22px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/grid-3x3.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/grid-3x3.svg) no-repeat center/contain;"></span></span><span> Confusion Matrix</span></h3>""", unsafe_allow_html=True)
        cm = confusion_matrix(test_clean['primary_topic'], pred, labels=top_classes)
        fig = px.imshow(cm, x=top_classes, y=top_classes,
                        text_auto=True, color_continuous_scale='Blues',
                        labels=dict(x="Predicted", y="Actual"))
        fig.update_layout(height=500)
        st.plotly_chart(fig, use_container_width=True)

        # Classification report
        st.markdown("""<h3 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:22px;height:22px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/clipboard-list.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/clipboard-list.svg) no-repeat center/contain;"></span></span><span> Classification Report</span></h3>""", unsafe_allow_html=True)
        rep = classification_report(test_clean['primary_topic'], pred,
                                     output_dict=True, zero_division=0)
        rep_df = pd.DataFrame(rep).T.round(3)
        st.dataframe(rep_df, use_container_width=True)

        # Top features
        st.markdown("""<h3 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:22px;height:22px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/star.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/star.svg) no-repeat center/contain;"></span></span><span> Top-20 Feature Importance</span></h3>""", unsafe_allow_html=True)
        feat_names = vec.get_feature_names_out()
        imp = clf.feature_importances_
        top_idx = np.argsort(imp)[::-1][:20]
        top_df = pd.DataFrame({
            'feature':    feat_names[top_idx],
            'importance': imp[top_idx],
        })
        fig = px.bar(top_df, x='importance', y='feature', orientation='h',
                     color='importance', color_continuous_scale='Viridis')
        fig.update_layout(height=520, yaxis=dict(autorange='reversed'))
        st.plotly_chart(fig, use_container_width=True)

        # Tree plot — depth 3 dan depth 4 (sesuai Gambar 9 & 10 laporan)
        st.markdown("""<h3 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:22px;height:22px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/git-branch.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/git-branch.svg) no-repeat center/contain;"></span></span><span> Visualisasi Pohon Keputusan</span></h3>""", unsafe_allow_html=True)
        _col1, _col2 = st.columns(2)
        with _col1:
            with st.expander("Pohon — depth ≤ 3 (Gambar 9)"):
                _fig, _ax = plt.subplots(figsize=(18, 8))
                plot_tree(clf, max_depth=3, feature_names=feat_names,
                          class_names=top_classes, filled=True, fontsize=8, ax=_ax)
                st.pyplot(_fig)
                plt.close(_fig)
        with _col2:
            with st.expander("Pohon — depth ≤ 4 (Gambar 10)"):
                _fig, _ax = plt.subplots(figsize=(22, 10))
                plot_tree(clf, max_depth=4, feature_names=feat_names,
                          class_names=top_classes, filled=True, fontsize=7, ax=_ax)
                st.pyplot(_fig)
                plt.close(_fig)

        # Kurva Overfitting: Accuracy vs max_depth (Gambar 12 laporan)
        st.markdown("""<h3 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:22px;height:22px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/trending-up.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/trending-up.svg) no-repeat center/contain;"></span></span><span> Kurva Overfitting — Accuracy vs max_depth (Gambar 12)</span></h3>""", unsafe_allow_html=True)
        if st.button("Hitung kurva overfitting", key="ovfit_btn"):
            with st.spinner("Training 15 model untuk kurva depth…"):
                _depths    = [1,2,3,4,5,7,10,12,15,20,25,30,40,50,None]
                _tr_accs, _te_accs, _xlabels = [], [], []
                _Xtr_cv, _Xte_cv, _vec_cv = vectorize(
                    train_clean['clean_text'].tolist(),
                    test_clean['clean_text'].tolist(),
                    method, max_features, ngram,
                )
                for _d in _depths:
                    _clf_d = DecisionTreeClassifier(criterion='gini', splitter='best',
                                                    max_depth=_d, random_state=42)
                    _clf_d.fit(_Xtr_cv, train_clean['primary_topic'])
                    _tr_accs.append(accuracy_score(train_clean['primary_topic'],
                                                   _clf_d.predict(_Xtr_cv)))
                    _te_accs.append(accuracy_score(test_clean['primary_topic'],
                                                   _clf_d.predict(_Xte_cv)))
                    _xlabels.append("None" if _d is None else str(_d))
            _ov_df = pd.DataFrame({
                'max_depth': _xlabels, 'Training': _tr_accs, 'Testing': _te_accs
            })
            _fig_ov = px.line(_ov_df, x='max_depth', y=['Training','Testing'],
                              markers=True, color_discrete_map={'Training':'#ED7D31','Testing':'#2E75B6'},
                              labels={'value':'Accuracy','variable':'Set','max_depth':'max_depth'})
            _fig_ov.update_layout(height=420, yaxis=dict(range=[0.5, 1.02]),
                                  title="Training vs Testing Accuracy — Fenomena Overfitting")
            st.plotly_chart(_fig_ov, use_container_width=True)
            st.caption("Training Accuracy mencapai 100% pada pohon penuh → tanda klasik overfitting. Testing Accuracy optimal di sekitar depth 10–15.")
    else:
        st.info("Klik **Latih Model** untuk memulai eksperimen. Hasil akan otomatis tersimpan dan tetap tampil saat kamu berpindah halaman.")

# ═══════════════════════════════════════════════════════════════════════════
# <span style="display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.25);vertical-align:-7px;margin-right:6px;flex-shrink:0;"><span style="display:inline-block;width:18px;height:18px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/git-compare.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/git-compare.svg) no-repeat center/contain;"></span></span> PERBANDINGAN
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Perbandingan":
    st.markdown("""<h2 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:40px;height:40px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:26px;height:26px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/git-compare.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/git-compare.svg) no-repeat center/contain;"></span></span><span> Perbandingan 3 Eksperimen</span></h2>""", unsafe_allow_html=True)

    # ── Parameter FIXED sesuai laporan — tidak ikut sidebar ──────────────
    # Laporan: max_features=10000, max_depth=None, min_samples_split=2
    # BoW: ngram(1,1) | N-gram: ngram(1,2) | TF-IDF: ngram(1,2)+sublinear_tf=True
    _EXP_CONFIG = {
        "BoW"   : {"method": "BoW",    "ngram": (1, 1)},
        "N-gram": {"method": "N-gram", "ngram": (1, 2)},
        "TF-IDF": {"method": "TF-IDF", "ngram": (1, 2)},
    }
    _MF = 10000   # max_features sesuai laporan
    _MD = None    # max_depth=None (pohon tumbuh bebas, depth 54-57 sesuai laporan)
    _MS = 2       # min_samples_split sesuai laporan

    st.info(
        "⚙️ **Parameter eksperimen dikunci:** "
        "`max_features=10.000` · `max_depth=None` · `min_samples_split=2` · "
        "BoW `ngram(1,1)` · N-gram `ngram(1,2)` · TF-IDF `ngram(1,2)+sublinear_tf=True`"
    )

    if st.button("Jalankan semua eksperimen", type="primary"):
        results  = {}
        progress = st.progress(0)
        for i, (exp_name, cfg) in enumerate(_EXP_CONFIG.items()):
            Xtr, Xte, vec = vectorize(
                train_clean['clean_text'].tolist(),
                test_clean['clean_text'].tolist(),
                cfg["method"], _MF, cfg["ngram"],
            )
            clf, pred, m = train_dt(
                Xtr, train_clean['primary_topic'],
                Xte, test_clean['primary_topic'],
                _MD, _MS,
            )
            results[exp_name] = {'metrics': m, 'pred': pred}
            progress.progress((i + 1) / 3)

        # (Override metrik laporan dihapus — gunakan hasil aktual dari dataset yang dimuat)

        st.session_state['compare_results'] = results
        progress.empty()
        st.success("✅ Eksperimen selesai.")

    if 'compare_results' in st.session_state:
        results      = st.session_state['compare_results']
        _metrics_dict = {k: v['metrics'] for k, v in results.items()}
        df = pd.DataFrame(_metrics_dict).T.round(4)
        df.index.name = 'Metode'

        st.markdown("""<h3 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:22px;height:22px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/bar-chart-3.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/bar-chart-3.svg) no-repeat center/contain;"></span></span><span> Tabel Perbandingan</span></h3>""", unsafe_allow_html=True)
        # Kolom tampilan utama — sesuai tabel laporan
        _display_cols = ['accuracy','precision_macro','recall_macro','f1_macro','f1','train_time','tree_depth','tree_leaves']
        _col_labels   = {
            'accuracy':'Accuracy','precision_macro':'Macro Precision',
            'recall_macro':'Macro Recall','f1_macro':'Macro F1 ★',
            'f1':'Weighted F1','train_time':'Waktu (s)',
            'tree_depth':'Depth','tree_leaves':'Leaves',
        }
        _df_show = df[[c for c in _display_cols if c in df.columns]].rename(columns=_col_labels)
        st.dataframe(_df_show.style.highlight_max(axis=0, color='#bbf7d0'),
                     use_container_width=True)

        st.download_button(
            "Download CSV", df_to_csv_bytes(df.reset_index()),
            "perbandingan_eksperimen.csv", "text/csv",
        )

        # Bar chart — pakai Macro metrics sesuai laporan
        metric_df = df[['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1']].reset_index().melt(
            id_vars='Metode', var_name='Metric', value_name='Score'
        )
        metric_df['Metric'] = metric_df['Metric'].map({
            'accuracy'        : 'Accuracy',
            'precision_macro' : 'Precision (Macro)',
            'recall_macro'    : 'Recall (Macro)',
            'f1_macro'        : 'F1 Macro ★',
            'f1'              : 'F1 (Weighted)',
        })
        fig = px.bar(metric_df, x='Metric', y='Score', color='Metode',
                     barmode='group', color_discrete_sequence=PALETTE,
                     text_auto='.3f')
        fig.update_layout(height=450, yaxis=dict(range=[0, 1]))
        st.plotly_chart(fig, use_container_width=True)

        # Radar — pakai Macro metrics sesuai laporan
        st.markdown("""<h3 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:22px;height:22px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/target.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/target.svg) no-repeat center/contain;"></span></span><span> Radar Comparison</span></h3>""", unsafe_allow_html=True)
        cats = ['accuracy', 'precision_macro', 'recall_macro', 'f1_macro', 'f1']
        cat_labels = ['Accuracy', 'Precision\n(Macro)', 'Recall\n(Macro)', 'F1 Macro ★', 'F1\n(Weighted)']
        fig = go.Figure()
        for i, m in enumerate(results.keys()):
            fig.add_trace(go.Scatterpolar(
                r=[results[m]['metrics'][c] for c in cats], theta=cat_labels, fill='toself',
                name=m, line_color=PALETTE[i],
            ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            height=500,
        )
        st.plotly_chart(fig, use_container_width=True)

        # F1-Score per Kategori — chart komparatif (Gambar 13 laporan)
        st.markdown("""<h3 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:36px;height:36px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:22px;height:22px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/bar-chart-3.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/bar-chart-3.svg) no-repeat center/contain;"></span></span><span>F1-Score per Kategori — 3 Eksperimen (Gambar 13)</span></h3>""", unsafe_allow_html=True)
        _f1_rows = []
        for _mname, _mdata in results.items():
            _rep = classification_report(
                test_clean['primary_topic'], _mdata['pred'],
                output_dict=True, zero_division=0, labels=top_classes
            )
            for _cls in top_classes:
                _f1_rows.append({
                    'Kategori': _cls, 'Metode': _mname,
                    'F1-Score': round(_rep[_cls]['f1-score'], 4)
                })
        _f1_df = pd.DataFrame(_f1_rows)
        _fig_f1 = px.bar(_f1_df, x='Kategori', y='F1-Score', color='Metode',
                         barmode='group', color_discrete_sequence=PALETTE,
                         text_auto='.2f')
        _fig_f1.update_layout(height=480, yaxis=dict(range=[0, 1.05]))
        _fig_f1.add_hline(y=0.8, line_dash='dot', line_color='gray',
                          annotation_text='F1=0.8 threshold')
        st.plotly_chart(_fig_f1, use_container_width=True)

        # Time comparison
        col1, col2 = st.columns(2)
        with col1:
            fig = px.bar(df.reset_index(), x='Metode', y='train_time',
                         color='Metode', color_discrete_sequence=PALETTE,
                         title="Train Time (detik)")
            fig.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            fig = px.bar(df.reset_index(), x='Metode', y='predict_time',
                         color='Metode', color_discrete_sequence=PALETTE,
                         title="Predict Time (detik)")
            fig.update_layout(showlegend=False, height=380)
            st.plotly_chart(fig, use_container_width=True)

        # Insight
        best_acc  = df['accuracy'].idxmax()
        best_f1m  = df['f1_macro'].idxmax()
        st.success(
            f"**Accuracy tertinggi:** `{best_acc}` ({df.loc[best_acc,'accuracy']:.4f})  \n"
            f"**Macro F1 tertinggi (rekomendasi notebook):** `{best_f1m}` "
            f"({df.loc[best_f1m,'f1_macro']:.4f})"
        )

    else:
        st.info("Klik tombol di atas untuk menjalankan ketiga eksperimen.")



# ═══════════════════════════════════════════════════════════════════════════
# PENINGKATAN MODEL — Balancing + Word/Contextual Embeddings
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Peningkatan Model":
    st.markdown("""<h2 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:40px;height:40px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:26px;height:26px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/zap.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/zap.svg) no-repeat center/contain;"></span></span><span> Peningkatan Kinerja Model</span></h2>""", unsafe_allow_html=True)
    st.caption("Cek balance kelas → terapkan **balancing** (resample / augmentation / back-translation) "
               "dan/atau ganti representasi teks dengan **word embedding** (Word2Vec, FastText, GloVe) "
               "atau **contextual embedding** (BERT, RoBERTa, DeBERTa). Latih ulang Decision Tree & bandingkan.")

    tab_dist, tab_bal, tab_emb, tab_run = st.tabs(
        ["📊 Distribusi Kelas", "⚖️ Balancing", "🧬 Text Representation", "🚀 Latih & Bandingkan"]
    )

    # ── 1) DISTRIBUSI ───────────────────────────────────────────────────
    with tab_dist:
        st.subheader("Distribusi kelas pada data training")
        dist_df, ratio = class_distribution(train_clean['primary_topic'])
        c1, c2, c3 = st.columns(3)
        c1.metric("Jumlah kelas", f"{len(dist_df)}")
        c2.metric("Imbalance ratio (max/min)", f"{ratio:.2f}×")
        status = "Balanced ✅" if ratio < 1.5 else ("Mild imbalance ⚠️" if ratio < 5 else "Imbalanced ❌")
        c3.metric("Status", status)

        fig = px.bar(dist_df, x='kelas', y='jumlah', color='jumlah',
                     color_continuous_scale='Viridis', text='jumlah')
        fig.update_layout(height=380, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

        if ratio >= 1.5:
            st.warning(f"Dataset **belum balance** (rasio {ratio:.2f}×). Disarankan terapkan balancing "
                       f"pada tab berikutnya — terutama untuk meningkatkan **F1 Macro**.")
        else:
            st.success("Dataset relatif balance — balancing opsional.")

    # ── 2) BALANCING ────────────────────────────────────────────────────
    with tab_bal:
        st.subheader("Pilih strategi balancing")
        bal_method = st.radio(
            "Strategi:",
            ["Tidak (baseline)",
             "Resample — Random Oversample",
             "Resample — Random Undersample",
             "Resample — SMOTE (di atas TF-IDF)",
             "Data Augmentation — Random Swap",
             "Data Augmentation — Random Deletion",
             "Data Augmentation — Synonym Replacement (WordNet)",
             "Back Translation (en→fr→en)"],
            help="Resample dilakukan SETELAH representasi teks. "
                 "Augmentation & Back-Translation dilakukan pada level TEKS sebelum vectorize."
        )
        st.session_state['bal_method'] = bal_method
        if bal_method.startswith("Back Translation"):
            st.info("⚠️ Back-translation memanggil Google Translate (butuh internet) dan **lambat**. "
                    "Gunakan parameter `max_per_class` untuk membatasi sampel.")
            st.session_state['bt_max'] = st.slider("max samples / kelas (target)", 50, 1000, 200, 50)

    # ── 3) TEXT REPRESENTATION ──────────────────────────────────────────
    with tab_emb:
        st.subheader("Pilih representasi fitur")
        rep_group = st.radio("Kelompok:",
            ["Klasik (BoW / N-gram / TF-IDF)",
             "Word Embedding non-contextual (Word2Vec / FastText / GloVe)",
             "Contextual Embedding (BERT / RoBERTa / DeBERTa)"],
            horizontal=False)
        st.session_state['rep_group'] = rep_group

        if rep_group.startswith("Klasik"):
            st.session_state['rep_method'] = st.selectbox("Metode klasik:", ["BoW", "N-gram", "TF-IDF"])
        elif rep_group.startswith("Word Embedding"):
            st.session_state['rep_method'] = st.selectbox("Embedding:", ["Word2Vec", "FastText", "GloVe (pretrained)"])
            st.session_state['emb_dim'] = st.select_slider("Vector size", [50, 100, 200, 300], value=100)
            if st.session_state['rep_method'].startswith("GloVe"):
                st.session_state['glove_name'] = st.selectbox(
                    "GloVe pretrained:", ["glove-wiki-gigaword-50", "glove-wiki-gigaword-100", "glove-wiki-gigaword-200"]
                )
                st.warning("Pertama kali load GloVe akan **download ~100-400 MB** via `gensim.downloader`.")
        else:
            st.session_state['rep_method'] = st.selectbox("Model kontekstual:", list(CONTEXTUAL_MODELS.keys()))
            st.info(f"Akan menggunakan: `{CONTEXTUAL_MODELS[st.session_state['rep_method']]}` "
                    f"(via `sentence-transformers`). Pertama kali akan **download model**.")
            st.session_state['ctx_batch'] = st.slider("Encode batch size", 8, 64, 32, 8)

    # ── 4) LATIH & BANDINGKAN ───────────────────────────────────────────
    with tab_run:
        st.subheader("Latih Decision Tree pada konfigurasi terpilih")
        bal_method = st.session_state.get('bal_method', "Tidak (baseline)")
        rep_group  = st.session_state.get('rep_group', "Klasik (BoW / N-gram / TF-IDF)")
        rep_method = st.session_state.get('rep_method', "TF-IDF")

        st.markdown(f"**Balancing:** `{bal_method}`  ·  **Representasi:** `{rep_group}` → `{rep_method}`")

        if st.button("🚀 Jalankan eksperimen peningkatan", type="primary", use_container_width=True):
            try:
                texts_tr = train_clean['clean_text'].tolist()
                y_tr     = train_clean['primary_topic'].tolist()
                texts_te = test_clean['clean_text'].tolist()
                y_te     = test_clean['primary_topic'].tolist()

                # --- (a) BALANCING di level TEKS untuk augmentation/back-translation ---
                if bal_method.startswith("Data Augmentation"):
                    tech = ("swap" if "Swap" in bal_method
                            else "deletion" if "Deletion" in bal_method
                            else "synonym")
                    with st.spinner(f"Augmenting text ({tech})…"):
                        texts_tr, y_tr = balance_via_augmentation(texts_tr, y_tr, technique=tech)
                    st.success(f"Augmentation selesai. Ukuran train: **{len(texts_tr)}**")
                elif bal_method.startswith("Back Translation"):
                    with st.spinner("Back-translating (lambat)…"):
                        texts_tr, y_tr = balance_via_back_translation(
                            texts_tr, y_tr, intermediate="fr",
                            max_per_class=st.session_state.get('bt_max', 200),
                        )
                    st.success(f"Back-translation selesai. Ukuran train: **{len(texts_tr)}**")

                # --- (b) TEXT REPRESENTATION ---
                t0 = time.time()
                if rep_group.startswith("Klasik"):
                    ngram = (1, 2) if rep_method in ("N-gram", "TF-IDF") else (1, 1)
                    Xtr, Xte, _vec = vectorize(texts_tr, texts_te, rep_method, max_features, ngram)
                elif rep_group.startswith("Word Embedding"):
                    dim = st.session_state.get('emb_dim', 100)
                    toks_tr = [t.split() for t in texts_tr]
                    toks_te = [t.split() for t in texts_te]
                    if rep_method == "Word2Vec":
                        with st.spinner("Training Word2Vec…"):
                            kv = train_word2vec(toks_tr, vector_size=dim)
                        Xtr = docs_to_mean_vec(toks_tr, kv, dim)
                        Xte = docs_to_mean_vec(toks_te, kv, dim)
                    elif rep_method == "FastText":
                        with st.spinner("Training FastText…"):
                            kv = train_fasttext(toks_tr, vector_size=dim)
                        Xtr = docs_to_mean_vec(toks_tr, kv, dim)
                        Xte = docs_to_mean_vec(toks_te, kv, dim)
                    else:  # GloVe pretrained
                        name = st.session_state.get('glove_name', "glove-wiki-gigaword-100")
                        with st.spinner(f"Loading GloVe `{name}`…"):
                            kv = load_glove(name)
                        gdim = kv.vector_size
                        Xtr = docs_to_mean_vec(toks_tr, kv, gdim)
                        Xte = docs_to_mean_vec(toks_te, kv, gdim)
                else:
                    model_name = CONTEXTUAL_MODELS[rep_method]
                    with st.spinner(f"Loading {rep_method} (`{model_name}`)…"):
                        encoder = load_contextual_model(model_name)
                    bs = st.session_state.get('ctx_batch', 32)
                    with st.spinner("Encoding train…"):
                        Xtr = encoder.encode(texts_tr, batch_size=bs, show_progress_bar=False,
                                             convert_to_numpy=True)
                    with st.spinner("Encoding test…"):
                        Xte = encoder.encode(texts_te, batch_size=bs, show_progress_bar=False,
                                             convert_to_numpy=True)
                feat_time = time.time() - t0

                # --- (c) RESAMPLE di level FITUR (sparse / dense) ---
                if bal_method.startswith("Resample"):
                    strat = ("oversample" if "Oversample" in bal_method
                             else "undersample" if "Undersample" in bal_method
                             else "smote")
                    if strat == "smote" and hasattr(Xtr, "toarray") and Xtr.shape[1] > 5000:
                        st.info("SMOTE pada matrix besar bisa lambat — pertimbangkan turunkan `max_features`.")
                    with st.spinner(f"Resampling ({strat})…"):
                        Xtr, y_tr = balance_resample(Xtr, y_tr, strategy=strat)
                    st.success(f"Resample selesai. Ukuran train: **{Xtr.shape[0]}**")

                # --- (d) TRAIN DECISION TREE ---
                with st.spinner("Training Decision Tree…"):
                    clf, pred, m = train_dt(Xtr, y_tr, Xte, y_te, max_depth, min_samples)
                m['feature_time'] = feat_time

                # --- (e) METRICS UI ---
                st.divider()
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Accuracy",       f"{m['accuracy']:.4f}")
                c2.metric("F1 Macro ★",     f"{m['f1_macro']:.4f}")
                c3.metric("Precision (M)",  f"{m['precision_macro']:.4f}")
                c4.metric("Recall (M)",     f"{m['recall_macro']:.4f}")
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Train time",   f"{m['train_time']:.2f} s")
                c2.metric("Feature time", f"{m['feature_time']:.2f} s")
                c3.metric("Tree depth",   f"{m['tree_depth']}")
                c4.metric("Tree leaves",  f"{m['tree_leaves']}")

                # confusion matrix
                cm = confusion_matrix(y_te, pred, labels=top_classes)
                fig = px.imshow(cm, x=top_classes, y=top_classes, text_auto=True,
                                color_continuous_scale='Blues',
                                labels=dict(x="Predicted", y="Actual"))
                fig.update_layout(height=500, title="Confusion Matrix")
                st.plotly_chart(fig, use_container_width=True)

                # simpan ke history utk perbandingan kumulatif
                hist = st.session_state.setdefault('peningkatan_history', [])
                hist.append({
                    'Balancing': bal_method,
                    'Representasi': f"{rep_method}",
                    'Accuracy': round(m['accuracy'], 4),
                    'F1 Macro': round(m['f1_macro'], 4),
                    'F1 Weighted': round(m['f1'], 4),
                    'Train (s)': round(m['train_time'], 2),
                    'Feature (s)': round(m['feature_time'], 2),
                })
            except RuntimeError as e:
                st.error(str(e))
            except Exception as e:
                st.error(f"Gagal: {e}")

        # ── HISTORY ─────────────────────────────────────────────────────
        hist = st.session_state.get('peningkatan_history', [])
        if hist:
            st.divider()
            st.subheader("📋 Riwayat eksperimen peningkatan")
            hdf = pd.DataFrame(hist)
            st.dataframe(hdf, use_container_width=True)
            fig = px.bar(hdf, x='Representasi', y='F1 Macro', color='Balancing',
                         barmode='group', text='F1 Macro',
                         color_discrete_sequence=px.colors.qualitative.Set2)
            fig.update_layout(height=420, title="F1 Macro per Eksperimen")
            st.plotly_chart(fig, use_container_width=True)
            st.download_button("⬇️ Download CSV history",
                               df_to_csv_bytes(hdf),
                               file_name="peningkatan_history.csv",
                               mime="text/csv")
            if st.button("🗑️ Reset history"):
                st.session_state.pop('peningkatan_history', None)
                st.rerun()



# ═══════════════════════════════════════════════════════════════════════════
# <span style="display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.25);vertical-align:-7px;margin-right:6px;flex-shrink:0;"><span style="display:inline-block;width:18px;height:18px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/search.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/search.svg) no-repeat center/contain;"></span></span> PREDIKSI
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Prediksi":
    st.markdown("""<h2 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:40px;height:40px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:26px;height:26px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/search.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/search.svg) no-repeat center/contain;"></span></span><span> Prediksi Teks Baru</span></h2>""", unsafe_allow_html=True)

    method = st.selectbox("Model yang digunakan:", ["BoW", "N-gram", "TF-IDF"])
    key = f'result_{method}'

    if key not in st.session_state:
        st.warning(f"Belum ada model `{method}`. Latih dulu di tab **Eksperimen Model**.")
    else:
        clf = st.session_state[key]['clf']
        vec = st.session_state[key]['vec']

        mode_tab1, mode_tab2 = st.tabs(["🔹 Single Prediction", "📦 Batch Prediction"])

        # ─────────────────────── SINGLE PREDICTION ───────────────────────
        with mode_tab1:
            st.markdown("Masukkan **satu** teks berita untuk diprediksi kategorinya.")
            text = st.text_area(
                "Masukkan teks berita (English):",
                "Oil prices surged after OPEC announced production cuts, "
                "boosting crude futures across global markets.",
                height=160, key="single_text"
            )

            if st.button("Prediksi", type="primary", key="btn_single"):
                clean = preprocess_text(text)
                X = vec.transform([clean])
                pred = clf.predict(X)[0]
                proba = clf.predict_proba(X)[0]

                st.success(f"### Prediksi: **`{pred}`**")
                st.caption(f"Confidence tertinggi: **{proba.max()*100:.2f}%**")

                prob_df = pd.DataFrame({
                    'Kategori': clf.classes_, 'Probabilitas': proba,
                }).sort_values('Probabilitas', ascending=True)

                fig = px.bar(prob_df, x='Probabilitas', y='Kategori', orientation='h',
                             color='Probabilitas', color_continuous_scale='Viridis',
                             text_auto='.3f')
                fig.update_layout(height=420)
                st.plotly_chart(fig, use_container_width=True)

        # ─────────────────────── BATCH PREDICTION ───────────────────────
        with mode_tab2:
            st.markdown(
                "Unggah **CSV** yang berisi kolom teks, atau tempel beberapa teks "
                "(satu teks per baris). Sistem akan memprediksi seluruh data sekaligus."
            )

            input_mode = st.radio(
                "Sumber data batch:",
                ["Upload CSV", "Tempel teks (multi-baris)"],
                horizontal=True, key="batch_source"
            )

            batch_df = None

            if input_mode == "Upload CSV":
                up = st.file_uploader("Upload file CSV", type=["csv"], key="batch_csv")
                if up is not None:
                    try:
                        batch_df = pd.read_csv(up)
                    except Exception as e:
                        st.error(f"Gagal membaca CSV: {e}")
                    if batch_df is not None and len(batch_df.columns) > 0:
                        text_col = st.selectbox(
                            "Pilih kolom teks:", batch_df.columns.tolist(),
                            key="batch_col"
                        )
                        batch_df = batch_df.rename(columns={text_col: "text"})
            else:
                raw = st.text_area(
                    "Tempel teks (satu per baris):",
                    "Oil prices surged after OPEC announced production cuts.\n"
                    "The central bank raised interest rates to curb inflation.\n"
                    "Wheat exports increased sharply this quarter.",
                    height=200, key="batch_text_area"
                )
                lines = [l.strip() for l in raw.splitlines() if l.strip()]
                if lines:
                    batch_df = pd.DataFrame({"text": lines})

            if batch_df is not None and "text" in batch_df.columns:
                st.info(f"Total data: **{len(batch_df)}** baris")
                st.dataframe(batch_df.head(10), use_container_width=True)

                if st.button("🚀 Jalankan Batch Prediction", type="primary",
                             key="btn_batch"):
                    with st.spinner("Memprediksi seluruh data..."):
                        texts = batch_df["text"].astype(str).tolist()
                        cleaned = [preprocess_text(t) for t in texts]
                        X = vec.transform(cleaned)
                        preds = clf.predict(X)
                        probas = clf.predict_proba(X)
                        conf = probas.max(axis=1)

                        out = batch_df.copy()
                        out["prediksi"] = preds
                        out["confidence"] = np.round(conf, 4)

                    st.success(f"✅ Selesai memprediksi {len(out)} baris.")
                    st.dataframe(out, use_container_width=True, height=380)

                    # Distribusi hasil prediksi
                    dist = (out["prediksi"].value_counts()
                            .rename_axis("Kategori").reset_index(name="Jumlah"))
                    fig_dist = px.bar(
                        dist, x="Kategori", y="Jumlah",
                        color="Jumlah", color_continuous_scale="Viridis",
                        text_auto=True, title="Distribusi Hasil Prediksi Batch"
                    )
                    fig_dist.update_layout(height=420)
                    st.plotly_chart(fig_dist, use_container_width=True)

                    # Download hasil
                    csv_bytes = out.to_csv(index=False).encode("utf-8")
                    st.download_button(
                        "⬇️ Download Hasil Prediksi (CSV)",
                        data=csv_bytes,
                        file_name=f"batch_prediction_{method}.csv",
                        mime="text/csv",
                    )

# ═══════════════════════════════════════════════════════════════════════════
# <span style="display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.25);vertical-align:-7px;margin-right:6px;flex-shrink:0;"><span style="display:inline-block;width:18px;height:18px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/book-open.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/book-open.svg) no-repeat center/contain;"></span></span> PANDUAN
# ═══════════════════════════════════════════════════════════════════════════
elif page == "Panduan":
    st.markdown("""<h2 style="display:flex;align-items:center;gap:10px;margin:1rem 0 .5rem;"><span style="display:inline-flex;align-items:center;justify-content:center;width:40px;height:40px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.30);flex-shrink:0;"><span style="display:inline-block;width:26px;height:26px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/book-open.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/book-open.svg) no-repeat center/contain;"></span></span><span> Panduan Penggunaan</span></h2>""", unsafe_allow_html=True)
    st.markdown("""
    ### <span style="display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.25);vertical-align:-7px;margin-right:6px;flex-shrink:0;"><span style="display:inline-block;width:18px;height:18px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/rocket.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/rocket.svg) no-repeat center/contain;"></span></span> Instalasi
    ```bash
    pip install -r requirements.txt
    ```

    ### <span style="display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.25);vertical-align:-7px;margin-right:6px;flex-shrink:0;"><span style="display:inline-block;width:18px;height:18px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/play.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/play.svg) no-repeat center/contain;"></span></span> Menjalankan
    ```bash
    streamlit run app.py
    ```
    Buka browser di `http://localhost:8501`.

    ### <span style="display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.25);vertical-align:-7px;margin-right:6px;flex-shrink:0;"><span style="display:inline-block;width:18px;height:18px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/folder-open.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/folder-open.svg) no-repeat center/contain;"></span></span> Struktur Dataset
    Pastikan file CSV memiliki kolom **`text`** (isi berita) dan **`topics`**
    (list string Python, misal `"['earn']"`).

    ### <span style="display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.25);vertical-align:-7px;margin-right:6px;flex-shrink:0;"><span style="display:inline-block;width:18px;height:18px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/compass.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/compass.svg) no-repeat center/contain;"></span></span> Alur Penggunaan Dashboard
    1. **Upload** `ModApte_train.csv` & `ModApte_test.csv` di sidebar.  
    2. **Beranda** → cek ringkasan dataset.  
    3. **EDA** → eksplorasi distribusi, panjang dokumen, top tokens.  
    4. **Pre-processing** → lihat efek pembersihan teks.  
    5. **Eksperimen Model** → latih DT dengan BoW / N-gram / TF-IDF.  
    6. **Perbandingan** → bandingkan ketiga eksperimen.  
    7. **Prediksi** → uji teks baru.

    ### <span style="display:inline-flex;align-items:center;justify-content:center;width:30px;height:30px;background:linear-gradient(135deg,#2E75B6,#7030A0);border-radius:50%;box-shadow:0 3px 10px rgba(0,0,0,0.25);vertical-align:-7px;margin-right:6px;flex-shrink:0;"><span style="display:inline-block;width:18px;height:18px;background-color:#ffffff;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/lightbulb.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/lightbulb.svg) no-repeat center/contain;"></span></span> Tips
    - Mulai dari `max_features = 10000`, `max_depth = None` (bebas tumbuh, sesuai laporan).  
    - Naikkan `max_features` jika ingin akurasi lebih tinggi (waktu training naik).  
    - Untuk laporan, gunakan tab **Perbandingan** sebagai screenshot utama.
    """, unsafe_allow_html=True)

# Footer
st.markdown("""
<div class="footer">
    <span style="display:inline-block;width:18px;height:18px;background-color:currentColor;vertical-align:-4px;margin-right:8px;-webkit-mask:url(https://unpkg.com/lucide-static@latest/icons/wrench.svg) no-repeat center/contain;mask:url(https://unpkg.com/lucide-static@latest/icons/wrench.svg) no-repeat center/contain;"></span>Built with Streamlit · Plotly · scikit-learn  ·  © Mini Project NLP
</div>
""", unsafe_allow_html=True)
