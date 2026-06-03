import streamlit as st
import joblib
import numpy as np
import time
import re
from datetime import datetime
import warnings
warnings.filterwarnings("ignore")

# ─── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="SentimentIQ · AI Text Analyzer",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Google Font */
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

  html, body, [class*="css"] {
      font-family: 'Inter', sans-serif;
  }

  /* Hero gradient banner */
  .hero-banner {
      background: linear-gradient(135deg, #667eea 0%, #764ba2 50%, #f64f59 100%);
      border-radius: 18px;
      padding: 2.2rem 2rem 1.8rem;
      text-align: center;
      margin-bottom: 2rem;
      box-shadow: 0 8px 32px rgba(102,126,234,0.35);
      animation: fadeSlideDown 0.7s ease forwards;
  }
  .hero-banner h1 { color: #fff; font-size: 2.4rem; font-weight: 700; margin: 0; letter-spacing:-0.5px; }
  .hero-banner p  { color: rgba(255,255,255,0.85); font-size: 1rem; margin-top: 0.5rem; }

  /* Fade-in animations */
  @keyframes fadeSlideDown {
      from { opacity:0; transform:translateY(-24px); }
      to   { opacity:1; transform:translateY(0); }
  }
  @keyframes fadeSlideUp {
      from { opacity:0; transform:translateY(20px); }
      to   { opacity:1; transform:translateY(0); }
  }
  @keyframes pulse {
      0%,100% { transform: scale(1); }
      50%      { transform: scale(1.08); }
  }
  @keyframes shimmer {
      0%   { background-position: -200% 0; }
      100% { background-position:  200% 0; }
  }

  /* Result card */
  .result-card {
      border-radius: 16px;
      padding: 2rem;
      text-align: center;
      margin-top: 1.5rem;
      animation: fadeSlideUp 0.5s ease forwards;
      box-shadow: 0 6px 24px rgba(0,0,0,0.12);
  }
  .result-card.negative { background: linear-gradient(135deg,#ff6b6b,#ee0979); }
  .result-card.neutral  { background: linear-gradient(135deg,#f7971e,#ffd200); }
  .result-card.positive { background: linear-gradient(135deg,#56ab2f,#a8e063); }

  .result-card .emoji  { font-size: 3.5rem; animation: pulse 1.6s ease infinite; }
  .result-card .label  { font-size: 1.8rem; font-weight: 700; color: #fff; margin: 0.4rem 0; }
  .result-card .conf   { font-size: 1rem; color: rgba(255,255,255,0.85); }

  /* Probability bars */
  .prob-row {
      display: flex; align-items: center; gap: 0.8rem;
      margin-bottom: 0.6rem; animation: fadeSlideUp 0.6s ease forwards;
  }
  .prob-label { width: 80px; font-weight: 600; font-size: 0.85rem; color: #555; text-align:right; }
  .prob-bar-bg { flex:1; background:#eee; border-radius:999px; height:14px; overflow:hidden; }
  .prob-bar-fill { height:100%; border-radius:999px; transition: width 1s ease; }
  .prob-value { width: 44px; font-size: 0.82rem; color:#777; font-weight:500; }

  /* History item */
  .history-item {
      background: #f8f9fc;
      border-left: 4px solid #667eea;
      border-radius: 0 10px 10px 0;
      padding: 0.75rem 1rem;
      margin-bottom: 0.6rem;
      font-size: 0.88rem;
      animation: fadeSlideUp 0.4s ease;
  }
  .history-item .hist-text  { color: #333; font-weight:500; }
  .history-item .hist-meta  { color: #999; font-size: 0.78rem; margin-top:2px; }

  /* Stat cards */
  .stat-card {
      background: linear-gradient(135deg,#667eea,#764ba2);
      border-radius: 14px; padding: 1rem;
      text-align: center; color: #fff;
      box-shadow: 0 4px 14px rgba(102,126,234,0.3);
  }
  .stat-card .stat-num  { font-size: 2rem; font-weight: 700; }
  .stat-card .stat-name { font-size: 0.8rem; opacity: 0.85; }

  /* Text area */
  .stTextArea textarea {
      border-radius: 12px !important;
      border: 2px solid #e0e4ef !important;
      font-size: 1rem !important;
      transition: border-color 0.2s;
  }
  .stTextArea textarea:focus { border-color: #667eea !important; }

  /* Button */
  .stButton > button {
      width: 100%; border-radius: 12px !important;
      background: linear-gradient(135deg,#667eea,#764ba2) !important;
      color: white !important; font-weight: 600 !important;
      font-size: 1.05rem !important; padding: 0.65rem !important;
      border: none !important;
      transition: opacity 0.2s, transform 0.15s !important;
      box-shadow: 0 4px 14px rgba(102,126,234,0.35) !important;
  }
  .stButton > button:hover { opacity:0.88 !important; transform:translateY(-1px) !important; }

  /* Sidebar */
  section[data-testid="stSidebar"] { background: #f0f2f9; }
  .sidebar-logo { font-size: 1.6rem; font-weight: 700; color: #667eea; text-align:center; margin-bottom: 1rem; }

  /* Shimmer loading bar */
  .loading-shimmer {
      height: 6px; border-radius: 999px; margin: 0.5rem 0;
      background: linear-gradient(90deg, #e0e4ef 25%, #c5cbe8 50%, #e0e4ef 75%);
      background-size: 200% 100%;
      animation: shimmer 1.2s infinite;
  }
  .divider { border: none; border-top: 1px solid #e4e7f0; margin: 1.5rem 0; }

  /* Word counter badge */
  .word-badge {
      display:inline-block; background:#eef0fb; color:#667eea;
      border-radius:999px; padding:2px 10px; font-size:0.78rem; font-weight:600;
  }
</style>
""", unsafe_allow_html=True)


# ─── Load Model ─────────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load("model.pkl")
    tfidf = joblib.load("tfidf.pkl")
    return model, tfidf

try:
    model, tfidf = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    load_error = str(e)


# ─── Constants ──────────────────────────────────────────────────────────────────
LABELS = {
    0: ("Negative", "😔", "#ee0979", "negative"),
    1: ("Positive", "😊", "#56ab2f", "positive"),
    2: ("Neutral",  "😐", "#f7971e", "neutral"),
}

EXAMPLE_TEXTS = {
    "😊 Happy review":
        "Absolutely love this product! It exceeded every expectation I had. The quality is outstanding and delivery was super fast. Highly recommended!",
    "😔 Negative feedback":
        "Terrible experience. The product broke after one day, customer service was unresponsive, and I'm extremely disappointed. Total waste of money.",
    "😐 Neutral statement":
        "I received the item on Wednesday. It came in the box as described. The packaging was fine. I'll update the review after using it more.",
}

BAR_COLORS = ["#ee0979", "#56ab2f", "#f7971e"]


# ─── Session State ───────────────────────────────────────────────────────────────
if "history"      not in st.session_state: st.session_state.history      = []
if "total_pos"    not in st.session_state: st.session_state.total_pos    = 0
if "total_neg"    not in st.session_state: st.session_state.total_neg    = 0
if "total_neu"    not in st.session_state: st.session_state.total_neu    = 0
if "total_count"  not in st.session_state: st.session_state.total_count  = 0


# ─── Sidebar ─────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="sidebar-logo">🧠 SentimentIQ</div>', unsafe_allow_html=True)
    st.markdown("---")

    st.markdown("### 📋 About")
    st.markdown(
        "A **Logistic Regression** model trained on text data with "
        "**TF-IDF** vectorization. Classifies input into three sentiment classes."
    )

    st.markdown("---")
    st.markdown("### ⚙️ Model Info")
    if model_loaded:
        st.markdown(f"- **Model**: Logistic Regression")
        st.markdown(f"- **Vectorizer**: TF-IDF")
        st.markdown(f"- **Vocabulary**: {len(tfidf.vocabulary_):,} terms")
        st.markdown(f"- **Classes**: Negative · Neutral · Positive")
        st.success("✅ Model loaded successfully", icon="🟢")
    else:
        st.error(f"❌ Failed to load model:\n{load_error}")

    st.markdown("---")
    st.markdown("### 📊 Session Stats")
    c1, c2 = st.columns(2)
    c1.markdown(
        f'<div class="stat-card"><div class="stat-num">{st.session_state.total_count}</div>'
        f'<div class="stat-name">Analyzed</div></div>', unsafe_allow_html=True
    )
    c2.markdown(
        f'<div class="stat-card"><div class="stat-num">{st.session_state.total_pos}</div>'
        f'<div class="stat-name">Positive</div></div>', unsafe_allow_html=True
    )
    st.markdown("<br>", unsafe_allow_html=True)
    c3, c4 = st.columns(2)
    c3.markdown(
        f'<div class="stat-card"><div class="stat-num">{st.session_state.total_neu}</div>'
        f'<div class="stat-name">Neutral</div></div>', unsafe_allow_html=True
    )
    c4.markdown(
        f'<div class="stat-card"><div class="stat-num">{st.session_state.total_neg}</div>'
        f'<div class="stat-name">Negative</div></div>', unsafe_allow_html=True
    )

    if st.button("🗑️ Clear History", use_container_width=True):
        st.session_state.history     = []
        st.session_state.total_pos   = 0
        st.session_state.total_neg   = 0
        st.session_state.total_neu   = 0
        st.session_state.total_count = 0
        st.rerun()


# ─── Main UI ──────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
  <h1>🧠 SentimentIQ</h1>
  <p>AI-Powered Text Sentiment Analyzer · Powered by Logistic Regression + TF-IDF</p>
</div>
""", unsafe_allow_html=True)

if not model_loaded:
    st.error("⚠️ Model files not found. Place `model.pkl` and `tfidf.pkl` in the same directory as `app.py`.")
    st.stop()

# ── Example selector ─────────────────────────────────────────────────────────────
st.markdown("#### 💡 Try an example")
example_choice = st.selectbox(
    "Load a sample text:", ["— select —"] + list(EXAMPLE_TEXTS.keys()),
    label_visibility="collapsed"
)
prefill = EXAMPLE_TEXTS.get(example_choice, "") if example_choice != "— select —" else ""

# ── Text Input ───────────────────────────────────────────────────────────────────
st.markdown("#### ✏️ Enter your text")
user_input = st.text_area(
    "Text to analyze",
    value=prefill,
    height=150,
    placeholder="Type or paste any text here — a review, tweet, comment, sentence…",
    label_visibility="collapsed",
)

word_count = len(user_input.split()) if user_input.strip() else 0
char_count = len(user_input)
st.markdown(
    f'<span class="word-badge">📝 {word_count} words · {char_count} chars</span>',
    unsafe_allow_html=True,
)

st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button("🔍 Analyze Sentiment", use_container_width=True)


# ── Prediction ───────────────────────────────────────────────────────────────────
if analyze_btn:
    text = user_input.strip()

    if not text:
        st.warning("⚠️ Please enter some text before analyzing.", icon="⚠️")
    elif len(text.split()) < 2:
        st.warning("⚠️ Text is too short. Please enter at least a few words.", icon="⚠️")
    else:
        # Loading animation
        progress_placeholder = st.empty()
        with progress_placeholder.container():
            st.markdown("**⚡ Analyzing…**")
            for _ in range(3):
                st.markdown('<div class="loading-shimmer"></div>', unsafe_allow_html=True)
            time.sleep(0.6)
        progress_placeholder.empty()

        # Predict
        X        = tfidf.transform([text])
        pred     = int(model.predict(X)[0])
        probs    = model.predict_proba(X)[0]
        label, emoji, color, css_class = LABELS[pred]
        conf     = float(probs[pred])

        # Result card
        st.markdown(f"""
        <div class="result-card {css_class}">
            <div class="emoji">{emoji}</div>
            <div class="label">{label}</div>
            <div class="conf">Confidence: {conf*100:.1f}%</div>
        </div>
        """, unsafe_allow_html=True)

        # Probability bars
        st.markdown("<br>**📊 Class Probabilities**", unsafe_allow_html=True)
        for i, (lbl, em, col, _) in LABELS.items():
            p = float(probs[i])
            st.markdown(f"""
            <div class="prob-row">
              <span class="prob-label">{em} {lbl}</span>
              <div class="prob-bar-bg">
                <div class="prob-bar-fill" style="width:{p*100:.1f}%;background:{col};"></div>
              </div>
              <span class="prob-value">{p*100:.1f}%</span>
            </div>
            """, unsafe_allow_html=True)

        # Top TF-IDF words
        feature_names = tfidf.get_feature_names_out()
        tfidf_scores  = X.toarray()[0]
        top_indices   = np.argsort(tfidf_scores)[::-1][:8]
        top_words     = [(feature_names[i], round(float(tfidf_scores[i]), 4))
                         for i in top_indices if tfidf_scores[i] > 0]

        if top_words:
            st.markdown("<hr class='divider'>", unsafe_allow_html=True)
            st.markdown("**🔑 Key Influencing Words**")
            cols = st.columns(min(len(top_words), 4))
            for idx, (word, score) in enumerate(top_words[:4]):
                with cols[idx]:
                    st.markdown(
                        f'<div style="background:#eef0fb;border-radius:10px;padding:8px;text-align:center;">'
                        f'<b style="color:#667eea">{word}</b><br>'
                        f'<span style="font-size:0.75rem;color:#888">{score:.4f}</span></div>',
                        unsafe_allow_html=True
                    )

        # Update session state
        st.session_state.total_count += 1
        if pred == 0: st.session_state.total_neg += 1
        elif pred == 1: st.session_state.total_pos += 1
        else: st.session_state.total_neu += 1

        # Add to history
        st.session_state.history.insert(0, {
            "text":  text[:120] + ("…" if len(text) > 120 else ""),
            "label": label,
            "emoji": emoji,
            "conf":  conf,
            "time":  datetime.now().strftime("%H:%M:%S"),
        })
        if len(st.session_state.history) > 10:
            st.session_state.history.pop()


# ── History ───────────────────────────────────────────────────────────────────────
if st.session_state.history:
    st.markdown("<hr class='divider'>", unsafe_allow_html=True)
    st.markdown("### 🕓 Recent Analyses")
    for item in st.session_state.history:
        st.markdown(f"""
        <div class="history-item">
          <div class="hist-text">{item['emoji']} <b>{item['label']}</b>
            &nbsp;<span style="color:#667eea;font-size:0.82rem">({item['conf']*100:.1f}%)</span>
            &nbsp;— {item['text']}
          </div>
          <div class="hist-meta">🕐 {item['time']}</div>
        </div>
        """, unsafe_allow_html=True)


# ── Footer ────────────────────────────────────────────────────────────────────────

st.markdown(
    """
    <style>
    .footer-divider {
        border: 0;
        height: 1px;
        background-image: linear-gradient(to right, rgba(0, 0, 0, 0), rgba(170, 170, 170, 0.75), rgba(0, 0, 0, 0));
        margin-top: 30px;
        margin-bottom: 15px;
    }
    .footer-text {
        text-align: center;
        color: #888888;
        font-size: 0.85rem;
        font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
        letter-spacing: 0.5px;
    }
    .footer-brand {
        color: #ff4b4b; 
        font-weight: 600;
    }
    </style>
    
    <hr class="footer-divider">
    <div class="footer-text">
        🧠 <span class="footer-brand">SentimentIQ</span> · Built with Streamlit · Logistic Regression + TF-IDF
        <br>
        <p style="margin-top: 5px; font-size: 0.8rem; color: #aaa;">
            © 2026 | Developed with ❤️ by <b>M.A.T</b>
        </p>
    </div>
    """,
    unsafe_allow_html=True
)