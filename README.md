# 🍄 Fungi Detective — Edible vs Poisonous Mushroom Detector

A web-based machine learning application that classifies mushrooms as **edible** or **poisonous** based on morphological characteristics, built with Streamlit and scikit-learn.

---

## Features

- **Machine Learning model** — Random Forest Classifier (no deep learning)
- **~99.5% accuracy** on the UCI Mushroom Dataset (8,124 samples)
- **22 morphological features** — cap, gills, stalk, veil, habitat, odor, and more
- **Auto-trains on first launch** — downloads the UCI dataset automatically
- **Feature importance chart** — shows which traits matter most
- **Retrain button** — retrain the model from the sidebar at any time
- **100% Python** — compatible with any Python 3.9+ environment

---

## Quickstart

### 1. Clone the repo
```bash
git clone https://github.com/<your-username>/fungi-detective.git
cd fungi-detective
```

### 2. Install dependencies
```bash
pip install -r requirements.txt
```

### 3. Run the app
```bash
streamlit run app.py
```

The app will automatically download the UCI Mushroom dataset and train the model on first launch (~20 seconds). After that, the trained model is cached locally as `mushroom_model.pkl`.

---

## Project Structure

```
fungi-detective/
├── app.py              # Streamlit web application
├── model.py            # ML model: training, saving, prediction logic
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── mushroom_model.pkl  # Saved model (generated on first run)
├── encoders.pkl        # Label encoders (generated on first run)
└── mushrooms.csv       # Dataset (downloaded on first run)
```

> **Note:** `mushroom_model.pkl`, `encoders.pkl`, and `mushrooms.csv` are generated automatically — you can add them to `.gitignore` or commit them for faster cold starts.

---

## Model Details

| Property | Value |
|---|---|
| Algorithm | Random Forest Classifier |
| n_estimators | 200 |
| Dataset | UCI Mushroom (Agaricus & Lepiota) |
| Train/Test Split | 80% / 20% |
| Test Accuracy | ~99.5% |
| Features | 22 categorical morphological traits |

### Top predictive features (by importance)
1. Odor
2. Spore print color
3. Gill color
4. Gill size
5. Ring type

---

## Retrain

From the sidebar in the app, click **🔄 Retrain Model**, or run directly:

```bash
python model.py
```

---

## ⚠️ Disclaimer

This application is for **educational purposes only**.  
**Never** consume a wild mushroom based solely on software predictions.  
Always consult a trained mycologist before consuming any wild fungi.

---

## License

MIT License
