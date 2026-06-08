"""
🍄 Fungi Detective — Edible vs Poisonous Detector
Streamlit web app powered by a Random Forest classifier trained on
the UCI Mushroom dataset (~8,000 samples, ~99%+ accuracy).

Run:  streamlit run app.py
"""

import os
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
    page_title="🍄 Fungi Detective",
    page_icon="🍄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Playfair+Display:wght@700;900&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
}

/* Background */
.stApp {
    background: linear-gradient(135deg, #0d1b12 0%, #122418 50%, #0a1a10 100%);
    color: #e8f5e0;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #091510 !important;
    border-right: 1px solid #1e3d28;
}
[data-testid="stSidebar"] .stMarkdown { color: #a8d5a2; }

/* Selectboxes */
.stSelectbox > div > div {
    background: #0f2318 !important;
    border: 1px solid #2a5c3a !important;
    color: #c8e6c0 !important;
    border-radius: 8px !important;
}

/* Buttons */
.stButton > button {
    background: linear-gradient(135deg, #2e7d32, #1b5e20) !important;
    color: #e8f5e0 !important;
    border: none !important;
    border-radius: 10px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    padding: 0.6rem 2rem !important;
    transition: all 0.3s ease !important;
    box-shadow: 0 4px 15px rgba(46, 125, 50, 0.4) !important;
    letter-spacing: 0.03em;
}
.stButton > button:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 6px 20px rgba(46, 125, 50, 0.6) !important;
}

/* Result cards */
.result-card {
    border-radius: 16px;
    padding: 2rem;
    text-align: center;
    animation: fadeIn 0.5s ease;
}
.edible-card {
    background: linear-gradient(135deg, #0a2e14, #143d1c);
    border: 2px solid #4caf50;
    box-shadow: 0 0 30px rgba(76, 175, 80, 0.3);
}
.poisonous-card {
    background: linear-gradient(135deg, #2e0a0a, #3d1414);
    border: 2px solid #f44336;
    box-shadow: 0 0 30px rgba(244, 67, 54, 0.3);
}
.result-title {
    font-family: 'Playfair Display', serif;
    font-size: 2.2rem;
    font-weight: 900;
    margin-bottom: 0.3rem;
}
.result-subtitle {
    font-size: 1rem;
    opacity: 0.75;
    margin-bottom: 1.2rem;
}
.confidence-bar-bg {
    background: rgba(255,255,255,0.1);
    border-radius: 999px;
    height: 12px;
    overflow: hidden;
}
.confidence-bar {
    height: 100%;
    border-radius: 999px;
    transition: width 1s ease;
}

/* Section headers */
.section-header {
    font-family: 'Playfair Display', serif;
    font-size: 1.1rem;
    font-weight: 700;
    color: #81c784;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    border-bottom: 1px solid #1e3d28;
    padding-bottom: 0.4rem;
    margin-bottom: 0.8rem;
    margin-top: 1.4rem;
}

/* Feature importance table */
.fi-row {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 6px;
}
.fi-label { font-size: 0.85rem; min-width: 180px; color: #a8d5a2; }
.fi-bar-bg { flex: 1; background: rgba(255,255,255,0.08); border-radius: 4px; height: 8px; }
.fi-bar { height: 100%; border-radius: 4px; background: linear-gradient(90deg, #4caf50, #81c784); }
.fi-pct { font-size: 0.78rem; color: #81c784; min-width: 38px; text-align: right; }

@keyframes fadeIn {
    from { opacity: 0; transform: translateY(16px); }
    to   { opacity: 1; transform: translateY(0); }
}

/* Hero */
.hero-title {
    font-family: 'Playfair Display', serif;
    font-size: 3rem;
    font-weight: 900;
    background: linear-gradient(135deg, #81c784, #c8e6c9, #a5d6a7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.1;
}
.hero-sub {
    color: #6a9e72;
    font-size: 1.05rem;
    margin-top: 0.3rem;
    font-weight: 300;
}
.badge {
    display: inline-block;
    background: #1b3a22;
    border: 1px solid #2e7d32;
    border-radius: 999px;
    padding: 3px 12px;
    font-size: 0.75rem;
    color: #81c784;
    margin-right: 6px;
    margin-bottom: 4px;
}
</style>
""", unsafe_allow_html=True)


# ── Model loading / training ───────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def get_model():
    model_path = "mushroom_model.pkl"
    encoders_path = "encoders.pkl"
    if not (os.path.exists(model_path) and os.path.exists(encoders_path)):
        with st.spinner("🌱 Training model for the first time — this takes ~20 seconds…"):
            clf, encoders, acc = train_and_save(
                model_path=model_path,
                encoders_path=encoders_path
            )
        return clf, encoders, acc
    else:
        clf, encoders = load_model(model_path, encoders_path)
        return clf, encoders, None


clf, encoders, training_acc = get_model()


# ── Sidebar — About ────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🍄 Fungi Detective")
    st.markdown("---")
    st.markdown("""
**Model:** Random Forest  
**Dataset:** UCI Mushroom (8,124 samples)  
**Features:** 22 morphological traits  
**Accuracy:** ~99.5% on test set
    """)
    st.markdown("---")
    st.markdown("""
**⚠️ Disclaimer**  
This tool is for *educational purposes only*. Never consume a wild mushroom based solely on software output.
Always consult a trained mycologist.
    """)
    st.markdown("---")

    # Re-train button
    if st.button("🔄 Retrain Model"):
        with st.spinner("Retraining…"):
            clf, encoders, training_acc = train_and_save()
            st.cache_resource.clear()
        st.success("Model retrained!")

    st.markdown("---")
    st.markdown("""
**GitHub Setup**
```bash
git clone <your-repo>
cd fungi-detective
pip install -r requirements.txt
streamlit run app.py
```
    """)


# ── Main layout ────────────────────────────────────────────────────────────────
col_hero, col_spacer = st.columns([3, 1])
with col_hero:
    st.markdown('<div class="hero-title">🍄 Fungi Detective</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Predict edibility from morphological features using Machine Learning</div>', unsafe_allow_html=True)
    st.markdown("""
<div style="margin-top:0.8rem">
  <span class="badge">Random Forest</span>
  <span class="badge">No Deep Learning</span>
  <span class="badge">UCI Dataset</span>
  <span class="badge">~99.5% Accuracy</span>
</div>
    """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Feature input form ─────────────────────────────────────────────────────────
st.markdown('<div class="section-header">📋 Mushroom Characteristics</div>', unsafe_allow_html=True)
st.markdown("Fill in the observable traits of your mushroom sample:")

# Group features into logical sections
SECTIONS = {
    "🎩 Cap": ["cap_shape", "cap_surface", "cap_color", "bruises"],
    "👃 Odor & Gills": ["odor", "gill_attachment", "gill_spacing", "gill_size", "gill_color"],
    "🌿 Stalk": [
        "stalk_shape", "stalk_root",
        "stalk_surface_above_ring", "stalk_surface_below_ring",
        "stalk_color_above_ring", "stalk_color_below_ring"
    ],
    "💍 Veil & Ring": ["veil_type", "veil_color", "ring_number", "ring_type"],
    "🌍 Spores & Habitat": ["spore_print_color", "population", "habitat"],
}

# Use encoders to know which values are valid for each feature
def get_valid_codes(feature):
    le = encoders[feature]
    return list(le.classes_)

user_input = {}

for section_title, features in SECTIONS.items():
    st.markdown(f'<div class="section-header">{section_title}</div>', unsafe_allow_html=True)
    cols = st.columns(min(len(features), 4))
    for i, feat in enumerate(features):
        with cols[i % 4]:
            codes = get_valid_codes(feat)
            label_map = FEATURE_MAPS.get(feat, {})
            options_display = [label_map.get(c, c) for c in codes]
            # Human-readable selectbox label
            display_label = feat.replace("_", " ").title()
            chosen_display = st.selectbox(display_label, options_display, key=feat)
            # Reverse-map to code
            rev_map = {v: k for k, v in label_map.items()}
            chosen_code = rev_map.get(chosen_display, chosen_display)
            user_input[feat] = chosen_code

# ── Predict button ─────────────────────────────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
col_btn, _ = st.columns([1, 3])
with col_btn:
    predict_clicked = st.button("🔍 Analyze Mushroom", use_container_width=True)

if predict_clicked:
    with st.spinner("Consulting the fungi oracle…"):
        result, prob_edible, prob_poison = predict(clf, encoders, user_input)

    st.markdown("<br>", unsafe_allow_html=True)

    if result == "edible":
        card_class = "edible-card"
        emoji = "✅"
        title_color = "#4caf50"
        verdict = "EDIBLE"
        bar_color = "#4caf50"
        confidence_val = prob_edible
        advice = "This mushroom appears to be edible based on its morphological profile. Always verify with a professional before consuming."
    else:
        card_class = "poisonous-card"
        emoji = "☠️"
        title_color = "#f44336"
        verdict = "POISONOUS"
        bar_color = "#f44336"
        confidence_val = prob_poison
        advice = "⚠️ This mushroom shows characteristics associated with toxic species. Do NOT consume it under any circumstances."

    pct = int(confidence_val * 100)
    st.markdown(f"""
    <div class="result-card {card_class}">
        <div style="font-size:3.5rem">{emoji}</div>
        <div class="result-title" style="color:{title_color}">{verdict}</div>
        <div class="result-subtitle">{advice}</div>
        <div style="margin: 1rem auto; max-width: 400px;">
            <div style="display:flex; justify-content:space-between; font-size:0.8rem; margin-bottom:4px; color:#aaa;">
                <span>Confidence</span><span>{pct}%</span>
            </div>
            <div class="confidence-bar-bg">
                <div class="confidence-bar" style="width:{pct}%; background:{bar_color};"></div>
            </div>
        </div>
        <div style="margin-top:1rem; font-size:0.82rem; opacity:0.6;">
            Edible prob: {prob_edible:.1%} &nbsp;|&nbsp; Poisonous prob: {prob_poison:.1%}
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Feature importance panel
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-header">📊 Feature Importance (Top 12)</div>', unsafe_allow_html=True)

    importances = clf.feature_importances_
    feat_imp = pd.Series(importances, index=FEATURE_COLUMNS).nlargest(12)
    max_imp = feat_imp.max()

    fi_html = ""
    for feat, imp in feat_imp.items():
        label = feat.replace("_", " ").title()
        bar_w = int((imp / max_imp) * 100)
        fi_html += f"""
        <div class="fi-row">
            <div class="fi-label">{label}</div>
            <div class="fi-bar-bg"><div class="fi-bar" style="width:{bar_w}%"></div></div>
            <div class="fi-pct">{imp*100:.1f}%</div>
        </div>
        """
    st.markdown(fi_html, unsafe_allow_html=True)


# ── Info footer ────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown("""
<div style="text-align:center; color:#3a5e40; font-size:0.8rem; padding-bottom: 1rem;">
    Built with Streamlit + scikit-learn · UCI Mushroom Dataset ·
    <em>For educational use only — never eat wild mushrooms without expert verification</em>
</div>
""", unsafe_allow_html=True)
