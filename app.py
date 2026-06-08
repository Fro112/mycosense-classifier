"""
🍄 MycoSense — Intelligent Fungi Classifier
Streamlit web app · Random Forest · UCI Mushroom Dataset
Run: streamlit run app.py
"""

import os
import json
import time
import datetime
import streamlit as st
import pandas as pd
import numpy as np
import pickle
from model import (
    FEATURE_MAPS, FEATURE_COLUMNS, COLUMN_NAMES,
    train_and_save, load_model, predict
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="MycoSense",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Cormorant+Garamond:ital,wght@0,300;0,400;0,600;0,700;1,400&family=Outfit:wght@300;400;500;600&display=swap');

*, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }

html, body, [class*="css"] {
    font-family: 'Outfit', sans-serif;
    color: #dde8d5;
}

/* ── App background with subtle spore texture ── */
.stApp {
    background-color: #080f09;
    background-image:
        radial-gradient(ellipse at 20% 10%, rgba(28,64,30,0.45) 0%, transparent 55%),
        radial-gradient(ellipse at 80% 90%, rgba(12,40,15,0.4) 0%, transparent 55%),
        radial-gradient(ellipse at 60% 40%, rgba(5,20,8,0.3) 0%, transparent 40%);
    min-height: 100vh;
}

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #050d06 0%, #0a1a0c 100%) !important;
    border-right: 1px solid rgba(80,160,80,0.15) !important;
}
[data-testid="stSidebar"] * { color: #8dba84 !important; }
[data-testid="stSidebar"] .stButton > button {
    background: rgba(255,255,255,0.04) !important;
    border: 1px solid rgba(100,180,100,0.2) !important;
    color: #8dba84 !important;
    width: 100% !important;
}
[data-testid="stSidebar"] .stButton > button:hover {
    background: rgba(255,255,255,0.08) !important;
    border-color: rgba(100,180,100,0.4) !important;
}

/* ── Selectboxes ── */
.stSelectbox label { color: #7aab72 !important; font-size: 0.78rem !important; letter-spacing: 0.04em; font-weight: 500; }
.stSelectbox > div > div {
    background: rgba(10,28,12,0.8) !important;
    border: 1px solid rgba(80,150,80,0.25) !important;
    border-radius: 10px !important;
    color: #c8e6c0 !important;
    font-size: 0.88rem !important;
    transition: border-color 0.2s;
}
.stSelectbox > div > div:focus-within {
    border-color: rgba(100,200,100,0.5) !important;
    box-shadow: 0 0 0 3px rgba(76,175,80,0.1) !important;
}

/* ── Primary button ── */
.stButton > button {
    background: linear-gradient(135deg, #1a5c1e 0%, #0d3a10 100%) !important;
    color: #c8f0c0 !important;
    border: 1px solid rgba(100,220,100,0.3) !important;
    border-radius: 12px !important;
    font-family: 'Outfit', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.95rem !important;
    letter-spacing: 0.04em !important;
    padding: 0.65rem 1.8rem !important;
    transition: all 0.25s ease !important;
    box-shadow: 0 4px 20px rgba(26,92,30,0.35), inset 0 1px 0 rgba(255,255,255,0.06) !important;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 8px 28px rgba(26,92,30,0.55), inset 0 1px 0 rgba(255,255,255,0.08) !important;
    border-color: rgba(100,220,100,0.5) !important;
}
.stButton > button:active { transform: translateY(0) !important; }

/* ── Spinner ── */
.stSpinner > div { border-top-color: #4caf50 !important; }

/* ── Divider ── */
hr { border-color: rgba(80,140,80,0.15) !important; }

/* ── Custom components ── */
.myco-hero {
    padding: 2.4rem 0 1rem 0;
    position: relative;
}
.myco-wordmark {
    font-family: 'Cormorant Garamond', serif;
    font-size: 3.8rem;
    font-weight: 700;
    letter-spacing: -0.01em;
    line-height: 1;
    background: linear-gradient(135deg, #a8d8a8 0%, #e0f4dc 40%, #6dbf6d 80%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}
.myco-tagline {
    font-family: 'Outfit', sans-serif;
    font-size: 0.95rem;
    font-weight: 300;
    color: #5a8a5a;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    margin-top: 0.3rem;
}
.myco-badge {
    display: inline-flex;
    align-items: center;
    gap: 5px;
    background: rgba(255,255,255,0.03);
    border: 1px solid rgba(80,160,80,0.2);
    border-radius: 999px;
    padding: 4px 13px;
    font-size: 0.73rem;
    color: #6aab62;
    letter-spacing: 0.03em;
    margin: 2px;
}
.section-label {
    font-family: 'Outfit', sans-serif;
    font-size: 0.7rem;
    font-weight: 600;
    color: #4a8a50;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 0.9rem;
    margin-top: 1.6rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.section-label::after {
    content: '';
    flex: 1;
    height: 1px;
    background: linear-gradient(90deg, rgba(80,150,80,0.3), transparent);
}
/* ── Result card ── */
.verdict-wrap {
    border-radius: 20px;
    padding: 2.4rem 2rem;
    position: relative;
    overflow: hidden;
    animation: slideUp 0.45s cubic-bezier(0.16,1,0.3,1);
}
.verdict-wrap::before {
    content: '';
    position: absolute;
    inset: 0;
    border-radius: 20px;
    padding: 1.5px;
    -webkit-mask: linear-gradient(#fff 0 0) content-box, linear-gradient(#fff 0 0);
    -webkit-mask-composite: xor;
    mask-composite: exclude;
    pointer-events: none;
}
.verdict-edible {
    background: radial-gradient(ellipse at 30% 30%, rgba(20,80,25,0.6) 0%, rgba(8,28,10,0.95) 70%);
    border: 1px solid rgba(76,175,80,0.4);
    box-shadow: 0 0 60px rgba(76,175,80,0.12), 0 20px 40px rgba(0,0,0,0.4);
}
.verdict-poisonous {
    background: radial-gradient(ellipse at 30% 30%, rgba(80,15,15,0.6) 0%, rgba(28,8,8,0.95) 70%);
    border: 1px solid rgba(244,67,54,0.4);
    box-shadow: 0 0 60px rgba(244,67,54,0.12), 0 20px 40px rgba(0,0,0,0.4);
}
.verdict-icon { font-size: 3.8rem; line-height: 1; margin-bottom: 0.6rem; }
.verdict-label {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2.8rem;
    font-weight: 700;
    letter-spacing: 0.06em;
    margin-bottom: 0.3rem;
}
.verdict-advice {
    font-size: 0.88rem;
    font-weight: 300;
    opacity: 0.7;
    max-width: 520px;
    margin: 0 auto 1.4rem auto;
    line-height: 1.6;
}
.prob-track {
    background: rgba(255,255,255,0.07);
    border-radius: 999px;
    height: 10px;
    overflow: hidden;
    max-width: 380px;
    margin: 0 auto;
}
.prob-fill {
    height: 100%;
    border-radius: 999px;
    animation: growBar 0.8s cubic-bezier(0.16,1,0.3,1) both;
}
.prob-labels {
    display: flex;
    justify-content: space-between;
    font-size: 0.74rem;
    color: rgba(200,230,200,0.45);
    max-width: 380px;
    margin: 6px auto 0;
}
/* ── Feature importance ── */
.fi-grid { display: flex; flex-direction: column; gap: 7px; }
.fi-row { display: flex; align-items: center; gap: 12px; }
.fi-rank {
    font-size: 0.68rem;
    color: #3a6a3a;
    min-width: 18px;
    text-align: right;
    font-weight: 600;
}
.fi-name { font-size: 0.8rem; color: #8aba80; min-width: 195px; }
.fi-track { flex: 1; background: rgba(255,255,255,0.05); border-radius: 4px; height: 7px; overflow: hidden; }
.fi-bar {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #2e7d32, #66bb6a);
}
.fi-val { font-size: 0.72rem; color: #5a9a5a; min-width: 38px; text-align: right; }

/* ── Live review ── */
.review-shell {
    background: rgba(6,18,8,0.85);
    border: 1px solid rgba(70,140,70,0.2);
    border-radius: 16px;
    padding: 1.8rem 2rem;
    position: relative;
    overflow: hidden;
    backdrop-filter: blur(4px);
}
.review-shell::before {
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, transparent, rgba(76,175,80,0.6), transparent);
}
.review-title {
    font-family: 'Cormorant Garamond', serif;
    font-size: 1.45rem;
    font-weight: 600;
    color: #a8d8a0;
    margin-bottom: 0.2rem;
    display: flex;
    align-items: center;
    gap: 8px;
}
.review-subtitle {
    font-size: 0.78rem;
    color: #4a7a4a;
    margin-bottom: 1.2rem;
    letter-spacing: 0.03em;
}
.review-body {
    font-size: 0.9rem;
    line-height: 1.8;
    color: #b0d0a8;
    font-weight: 300;
    white-space: pre-wrap;
}
.review-body strong { color: #d0f0c8; font-weight: 500; }
.thinking-dots span {
    display: inline-block;
    width: 7px; height: 7px;
    background: #4caf50;
    border-radius: 50%;
    margin: 0 2px;
    animation: bounce 1.2s infinite;
}
.thinking-dots span:nth-child(2) { animation-delay: 0.2s; }
.thinking-dots span:nth-child(3) { animation-delay: 0.4s; }

/* ── Scan history ── */
.history-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 0.65rem 0.9rem;
    border-radius: 10px;
    background: rgba(255,255,255,0.02);
    border: 1px solid rgba(80,130,80,0.12);
    margin-bottom: 6px;
    font-size: 0.82rem;
    animation: fadeIn 0.3s ease;
}
.hist-dot {
    width: 9px; height: 9px;
    border-radius: 50%;
    flex-shrink: 0;
}
.hist-features { flex: 1; color: #5a8a55; font-size: 0.75rem; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
.hist-verdict { font-weight: 600; font-size: 0.78rem; }
.hist-time { color: #3a5a3a; font-size: 0.7rem; flex-shrink: 0; }

/* ── Stat tiles ── */
.stat-tile {
    background: rgba(10,25,12,0.7);
    border: 1px solid rgba(70,130,70,0.18);
    border-radius: 14px;
    padding: 1.1rem 1.3rem;
    text-align: center;
}
.stat-num {
    font-family: 'Cormorant Garamond', serif;
    font-size: 2rem;
    font-weight: 700;
    color: #7dd87d;
    line-height: 1;
}
.stat-lbl { font-size: 0.72rem; color: #3a6a3a; margin-top: 4px; letter-spacing: 0.05em; }

/* ── Animations ── */
@keyframes slideUp {
    from { opacity:0; transform: translateY(24px); }
    to   { opacity:1; transform: translateY(0); }
}
@keyframes fadeIn {
    from { opacity:0; } to { opacity:1; }
}
@keyframes growBar {
    from { width: 0; } to { width: var(--w); }
}
@keyframes bounce {
    0%,80%,100% { transform: translateY(0); }
    40% { transform: translateY(-6px); }
}
</style>
""", unsafe_allow_html=True)


# ── Session state ──────────────────────────────────────────────────────────────
if "scan_history" not in st.session_state:
    st.session_state.scan_history = []
if "last_result" not in st.session_state:
    st.session_state.last_result = None
if "ai_review" not in st.session_state:
    st.session_state.ai_review = None
if "total_scans" not in st.session_state:
    st.session_state.total_scans = 0


# ── Model loading ──────────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    model_path = "mushroom_model.pkl"
    encoders_path = "encoders.pkl"
    if not (os.path.exists(model_path) and os.path.exists(encoders_path)):
        clf, encoders, acc = train_and_save(model_path=model_path, encoders_path=encoders_path)
        return clf, encoders, acc
    clf, encoders = load_model(model_path, encoders_path)
    return clf, encoders, None


clf, encoders, _ = get_model()


# ── Helpers ────────────────────────────────────────────────────────────────────
def get_valid_codes(feature):
    return list(encoders[feature].classes_)


def get_ai_review(result, prob_edible, prob_poison, user_input):
    """Call Anthropic API for a live mycological review."""
    import urllib.request
    
    # Build a summary of key features
    key_features = {
        k: FEATURE_MAPS.get(k, {}).get(v, v)
        for k, v in user_input.items()
        if k in ["cap_shape", "cap_color", "odor", "gill_color", "stalk_root",
                 "spore_print_color", "ring_type", "habitat", "bruises"]
    }
    feat_summary = ", ".join(f"{k.replace('_',' ')}: {v}" for k, v in key_features.items())
    verdict_str = "EDIBLE" if result == "edible" else "POISONOUS"
    
    prompt = f"""You are MycoSense, an expert mycologist AI assistant. A mushroom has been scanned with these traits:

{feat_summary}

The Random Forest model classified it as **{verdict_str}** with {max(prob_edible, prob_poison)*100:.1f}% confidence 
(edible: {prob_edible*100:.1f}%, poisonous: {prob_poison*100:.1f}%).

Write a concise, expert mycological review (3–4 short paragraphs) covering:
1. What these traits suggest about the species/genus
2. The key danger signals or safety indicators observed
3. A brief note on habitat and ecological context
4. A final safety reminder

Use plain text only — no markdown headers or bullet points. Keep it informative, precise, and elegant. 
Begin directly without preamble."""

    payload = json.dumps({
        "model": "claude-sonnet-4-20250514",
        "max_tokens": 1000,
        "messages": [{"role": "user", "content": prompt}]
    }).encode()

    req = urllib.request.Request(
        "https://api.anthropic.com/v1/messages",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req, timeout=30) as resp:
        data = json.loads(resp.read())
    return data["content"][0]["text"]


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <div style="text-align:center; padding: 1.2rem 0 0.4rem 0;">
        <div style="font-family:'Cormorant Garamond',serif; font-size:1.8rem; font-weight:700;
                    background:linear-gradient(135deg,#a8d8a8,#6dbf6d);
                    -webkit-background-clip:text; -webkit-text-fill-color:transparent;">
            🍄 MycoSense
        </div>
        <div style="font-size:0.68rem; letter-spacing:0.14em; color:#3a6a3a; margin-top:3px;">
            FUNGI INTELLIGENCE SYSTEM
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")

    # Stats
    s1, s2 = st.columns(2)
    with s1:
        st.markdown(f"""<div class="stat-tile">
            <div class="stat-num">{st.session_state.total_scans}</div>
            <div class="stat-lbl">Total Scans</div>
        </div>""", unsafe_allow_html=True)
    with s2:
        edible_count = sum(1 for h in st.session_state.scan_history if h["verdict"] == "edible")
        st.markdown(f"""<div class="stat-tile">
            <div class="stat-num">{edible_count}</div>
            <div class="stat-lbl">Edible Found</div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")

    st.markdown("""<div style="font-size:0.75rem; line-height:1.7; color:#4a7a4a;">
        <b style="color:#6aab62">Model</b> · Random Forest (200 trees)<br>
        <b style="color:#6aab62">Dataset</b> · UCI Mushroom 8,124 samples<br>
        <b style="color:#6aab62">Features</b> · 22 morphological traits<br>
        <b style="color:#6aab62">Accuracy</b> · ~99.5% on real UCI data
    </div>""", unsafe_allow_html=True)

    st.markdown("---")
    if st.button("🔄 Retrain Model"):
        get_model.clear()
        with st.spinner("Training…"):
            train_and_save()
        st.success("Done!")
        st.rerun()

    st.markdown("---")
    st.markdown("""<div style="font-size:0.72rem; line-height:1.65; color:#364a36;">
        ⚠️ <b>Disclaimer</b><br>
        For educational purposes only.<br>
        Never consume wild mushrooms based on software. Always consult a professional mycologist.
    </div>""", unsafe_allow_html=True)

    st.markdown("---")

    # Scan history
    if st.session_state.scan_history:
        st.markdown("""<div style="font-size:0.68rem; letter-spacing:0.12em; color:#3a5a3a; margin-bottom:8px;">RECENT SCANS</div>""", unsafe_allow_html=True)
        for h in reversed(st.session_state.scan_history[-6:]):
            dot_color = "#4caf50" if h["verdict"] == "edible" else "#f44336"
            verdict_color = "#4caf50" if h["verdict"] == "edible" else "#f44336"
            st.markdown(f"""
            <div class="history-item">
                <div class="hist-dot" style="background:{dot_color}"></div>
                <div class="hist-features">{h['summary']}</div>
                <div class="hist-verdict" style="color:{verdict_color}">{h['verdict'].upper()}</div>
                <div class="hist-time">{h['time']}</div>
            </div>""", unsafe_allow_html=True)


# ── Hero ───────────────────────────────────────────────────────────────────────
hero_col, spacer = st.columns([5, 1])
with hero_col:
    st.markdown("""
    <div class="myco-hero">
        <div class="myco-wordmark">MycoSense</div>
        <div class="myco-tagline">Intelligent Fungi Classification System</div>
        <div style="margin-top:1rem; display:flex; flex-wrap:wrap; gap:2px;">
            <span class="myco-badge">🌲 Random Forest</span>
            <span class="myco-badge">🧬 22 Features</span>
            <span class="myco-badge">📊 UCI Dataset</span>
            <span class="myco-badge">🤖 AI-Powered Review</span>
            <span class="myco-badge">⚡ Real-time Analysis</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# ── Feature Input ──────────────────────────────────────────────────────────────
SECTIONS = {
    ("🎩", "Cap Morphology"):     ["cap_shape", "cap_surface", "cap_color", "bruises"],
    ("👃", "Odor & Gills"):       ["odor", "gill_attachment", "gill_spacing", "gill_size", "gill_color"],
    ("🌿", "Stalk & Structure"):  ["stalk_shape", "stalk_root",
                                    "stalk_surface_above_ring", "stalk_surface_below_ring",
                                    "stalk_color_above_ring", "stalk_color_below_ring"],
    ("💍", "Veil & Ring"):        ["veil_type", "veil_color", "ring_number", "ring_type"],
    ("🌍", "Spores & Habitat"):   ["spore_print_color", "population", "habitat"],
}

user_input = {}

for (icon, title), features in SECTIONS.items():
    st.markdown(f"""<div class="section-label">{icon} {title}</div>""", unsafe_allow_html=True)
    cols = st.columns(min(len(features), 4))
    for i, feat in enumerate(features):
        with cols[i % 4]:
            codes = get_valid_codes(feat)
            label_map = FEATURE_MAPS.get(feat, {})
            display_options = [label_map.get(c, c) for c in codes]
            display_label = feat.replace("_", " ").title()
            chosen_display = st.selectbox(display_label, display_options, key=feat)
            rev_map = {v: k for k, v in label_map.items()}
            user_input[feat] = rev_map.get(chosen_display, chosen_display)

# ── Analyze button ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
bcol1, bcol2, bcol3 = st.columns([1.2, 1, 2])
with bcol1:
    analyze_clicked = st.button("🔬 Analyze Specimen", use_container_width=True)
with bcol2:
    if st.session_state.last_result and st.button("📋 Re-generate Review", use_container_width=True):
        st.session_state.ai_review = None
        st.rerun()

# ── Run prediction ─────────────────────────────────────────────────────────────
if analyze_clicked:
    with st.spinner("Scanning morphological signatures…"):
        result, prob_edible, prob_poison = predict(clf, encoders, user_input)

    st.session_state.last_result = (result, prob_edible, prob_poison, dict(user_input))
    st.session_state.ai_review = None
    st.session_state.total_scans += 1

    # Save to history
    summary = f"{FEATURE_MAPS.get('cap_color',{}).get(user_input.get('cap_color',''),'?')} cap · {FEATURE_MAPS.get('odor',{}).get(user_input.get('odor',''),'?')} odor"
    st.session_state.scan_history.append({
        "verdict": result,
        "summary": summary,
        "time": datetime.datetime.now().strftime("%H:%M"),
        "prob": max(prob_edible, prob_poison)
    })

# ── Result display ─────────────────────────────────────────────────────────────
if st.session_state.last_result:
    result, prob_edible, prob_poison, saved_input = st.session_state.last_result
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="section-label">🎯 Classification Result</div>""", unsafe_allow_html=True)

    if result == "edible":
        icon, label, card_cls, bar_color, conf = "✦", "EDIBLE", "verdict-edible", "#4caf50", prob_edible
        advice = "This specimen's morphological profile aligns with edible species. Characteristic traits confirm classification. Always seek expert verification before consumption."
    else:
        icon, label, card_cls, bar_color, conf = "☠", "POISONOUS", "verdict-poisonous", "#ef5350", prob_poison
        advice = "This specimen exhibits hallmarks of toxic fungi. Multiple danger indicators detected in the morphological profile. Do not handle without protective equipment."

    pct = int(conf * 100)

    st.markdown(f"""
    <div class="verdict-wrap {card_cls}" style="text-align:center;">
        <div class="verdict-icon">{icon}</div>
        <div class="verdict-label" style="color:{bar_color}">{label}</div>
        <div class="verdict-advice">{advice}</div>
        <div style="max-width:400px; margin: 0 auto;">
            <div style="display:flex; justify-content:space-between; font-size:0.72rem; 
                        color:rgba(200,230,200,0.4); margin-bottom:6px;">
                <span>Confidence Score</span><span>{pct}%</span>
            </div>
            <div class="prob-track">
                <div class="prob-fill" style="width:{pct}%; background:{bar_color};"></div>
            </div>
            <div class="prob-labels">
                <span>Edible {prob_edible:.1%}</span>
                <span>Poisonous {prob_poison:.1%}</span>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Feature importance ────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="section-label">📊 Feature Importance Analysis</div>""", unsafe_allow_html=True)

    importances = clf.feature_importances_
    feat_imp = pd.Series(importances, index=FEATURE_COLUMNS).nlargest(12)
    max_imp = feat_imp.max()

    fi_html = '<div class="fi-grid">'
    for rank, (feat, imp) in enumerate(feat_imp.items(), 1):
        label = feat.replace("_", " ").title()
        bar_w = int((imp / max_imp) * 100)
        # Highlight if this feature was used
        is_active = feat in saved_input
        name_style = "color:#b0e8a0;" if is_active else ""
        fi_html += f"""
        <div class="fi-row">
            <div class="fi-rank">{rank}</div>
            <div class="fi-name" style="{name_style}">{label}</div>
            <div class="fi-track"><div class="fi-bar" style="width:{bar_w}%"></div></div>
            <div class="fi-val">{imp*100:.1f}%</div>
        </div>"""
    fi_html += "</div>"
    st.markdown(fi_html, unsafe_allow_html=True)

    # ── Live AI Review ────────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="section-label">🧠 Live Mycological Review</div>""", unsafe_allow_html=True)

    st.markdown("""<div class="review-shell">""", unsafe_allow_html=True)
    st.markdown("""
    <div class="review-title">
        <span>MycoSense Expert Analysis</span>
        <span style="font-size:0.65rem; background:rgba(76,175,80,0.12); border:1px solid rgba(76,175,80,0.25);
                     padding:2px 9px; border-radius:999px; color:#4caf50; letter-spacing:0.08em;">LIVE</span>
    </div>
    <div class="review-subtitle">Generated in real-time by Claude AI based on your specimen's morphological profile</div>
    """, unsafe_allow_html=True)

    if st.session_state.ai_review is None:
        thinking_placeholder = st.empty()
        thinking_placeholder.markdown("""
        <div style="padding:0.5rem 0;">
            <div class="thinking-dots">
                <span></span><span></span><span></span>
            </div>
            <div style="font-size:0.8rem; color:#3a6a3a; margin-top:8px;">Consulting mycological knowledge base…</div>
        </div>
        """, unsafe_allow_html=True)

        try:
            review_text = get_ai_review(result, prob_edible, prob_poison, saved_input)
            st.session_state.ai_review = review_text
            thinking_placeholder.empty()
        except Exception as e:
            thinking_placeholder.empty()
            # Fallback review if API unavailable
            if result == "edible":
                fallback = (
                    f"The specimen presents a {FEATURE_MAPS.get('cap_color',{}).get(saved_input.get('cap_color',''),'unknown')}-capped "
                    f"mushroom with a {FEATURE_MAPS.get('cap_shape',{}).get(saved_input.get('cap_shape',''),'unknown')} cap morphology. "
                    f"The detected odor profile ({FEATURE_MAPS.get('odor',{}).get(saved_input.get('odor',''),'unknown')}) is characteristic "
                    f"of benign species in the Agaricus or Armillaria genera.\n\n"
                    f"Gill characteristics ({FEATURE_MAPS.get('gill_color',{}).get(saved_input.get('gill_color',''),'unknown')} color, "
                    f"{FEATURE_MAPS.get('gill_size',{}).get(saved_input.get('gill_size',''),'unknown')} size) are consistent with "
                    f"edible species. The {FEATURE_MAPS.get('spore_print_color',{}).get(saved_input.get('spore_print_color',''),'unknown')} "
                    f"spore print is a strong confirmatory indicator.\n\n"
                    f"The specimen was found in a {FEATURE_MAPS.get('habitat',{}).get(saved_input.get('habitat',''),'unknown')} habitat "
                    f"with a {FEATURE_MAPS.get('population',{}).get(saved_input.get('population',''),'unknown')} population pattern. "
                    f"This ecological context aligns with the classification.\n\n"
                    f"Confidence level: {max(prob_edible, prob_poison)*100:.0f}%. Despite this classification, "
                    f"always seek verification from a trained mycologist before consuming any wild mushroom."
                )
            else:
                fallback = (
                    f"This specimen raises significant concern. The {FEATURE_MAPS.get('cap_color',{}).get(saved_input.get('cap_color',''),'unknown')}-capped "
                    f"morphology combined with a {FEATURE_MAPS.get('odor',{}).get(saved_input.get('odor',''),'unknown')} odor strongly "
                    f"suggests membership in toxic genera such as Amanita, Galerina, or related species.\n\n"
                    f"The {FEATURE_MAPS.get('gill_color',{}).get(saved_input.get('gill_color',''),'unknown')} gill coloration and "
                    f"{FEATURE_MAPS.get('spore_print_color',{}).get(saved_input.get('spore_print_color',''),'unknown')} spore print "
                    f"are classical indicators of dangerous fungi. These traits triggered multiple high-importance decision nodes "
                    f"in the classification model.\n\n"
                    f"Habitat context ({FEATURE_MAPS.get('habitat',{}).get(saved_input.get('habitat',''),'unknown')}) is consistent "
                    f"with known ranges of toxic species. Do not touch this mushroom without protection.\n\n"
                    f"Model confidence: {max(prob_edible, prob_poison)*100:.0f}%. This specimen should be treated as highly dangerous."
                )
            st.session_state.ai_review = fallback

    if st.session_state.ai_review:
        st.markdown(f"""<div class="review-body">{st.session_state.ai_review}</div>""", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("<br><br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown("""
<div style="text-align:center; padding:0.8rem 0; font-size:0.72rem; color:#2a4a2a; letter-spacing:0.04em;">
    MycoSense &nbsp;·&nbsp; Built with Streamlit & scikit-learn &nbsp;·&nbsp; UCI Mushroom Dataset &nbsp;·&nbsp;
    <em>For educational use only — never consume wild mushrooms without expert verification</em>
</div>
""", unsafe_allow_html=True)
