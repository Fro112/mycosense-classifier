# 🍄 MycoSense — Intelligent Fungi Classification System

A web-based machine learning application that classifies mushrooms as **edible** or **poisonous** using morphological features, complete with a **live AI-powered mycological review** after each scan.

Built with Streamlit, scikit-learn, and the Anthropic Claude API.

---

## Features

- **Random Forest Classifier** — no deep learning, fully interpretable
- **~99.5% accuracy** on the UCI Mushroom Dataset (8,124 samples)
- **22 morphological features** — cap, gills, stalk, veil, habitat, odor, spores
- **Live AI Review** — Claude generates an expert mycological commentary after every scan
- **Scan history** — sidebar tracks your recent analyses with timestamps
- **Feature importance chart** — visualizes which traits drove the prediction
- **Auto-trains on first launch** — downloads or generates the dataset automatically
- **Fallback review** — if the API is unavailable, a structured review is generated locally

---

## Quickstart

### 1. Clone
```bash
git clone https://github.com/<your-username>/mycosense.git
cd mycosense
```

### 2. Install
```bash
pip install -r requirements.txt
```

### 3. Run
```bash
streamlit run app.py
```

On first launch the app trains the model automatically (~15–20 seconds). The model is then cached as `mushroom_model.pkl` for instant future starts.

---

## Project Structure

```
mycosense/
├── app.py              # Streamlit UI (MycoSense)
├── model.py            # Random Forest training, saving, prediction
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── mushroom_model.pkl  # Trained model (auto-generated)
├── encoders.pkl        # Label encoders (auto-generated)
└── mushrooms.csv       # Dataset (auto-downloaded or generated)
```

> `mushroom_model.pkl`, `encoders.pkl`, and `mushrooms.csv` are auto-generated — optionally add to `.gitignore` or commit them for faster cold starts.

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| Trees | 200 |
| Dataset | UCI Mushroom — Agaricus & Lepiota |
| Split | 80% train / 20% test |
| Accuracy | ~99.5% (real UCI data) |
| Features | 22 categorical morphological traits |

---

## AI Review

The live review uses `claude-sonnet-4-20250514` via the Anthropic API. It receives the morphological profile and model output, then generates a 3–4 paragraph expert commentary covering species indicators, danger signals, habitat context, and safety guidance.

If the API is unreachable, a structured fallback review is generated from the local feature data.

---

## ⚠️ Disclaimer

**For educational purposes only.**  
Never consume a wild mushroom based on software predictions.  
Always consult a trained mycologist before handling or consuming wild fungi.

---

## License

MIT
