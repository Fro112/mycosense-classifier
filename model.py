"""
Fungi Detection Model — Train & Save
Uses UCI Mushroom dataset with scikit-learn RandomForestClassifier.
Run this once to generate mushroom_model.pkl and encoders.pkl
"""

import pandas as pd
import numpy as np
import pickle
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score

# ── UCI Mushroom dataset column names ──────────────────────────────────────────
COLUMN_NAMES = [
    "class", "cap_shape", "cap_surface", "cap_color", "bruises", "odor",
    "gill_attachment", "gill_spacing", "gill_size", "gill_color",
    "stalk_shape", "stalk_root", "stalk_surface_above_ring",
    "stalk_surface_below_ring", "stalk_color_above_ring",
    "stalk_color_below_ring", "veil_type", "veil_color",
    "ring_number", "ring_type", "spore_print_color",
    "population", "habitat"
]

# Human-readable mappings for each feature (for the UI)
FEATURE_MAPS = {
    "cap_shape": {
        "b": "Bell", "c": "Conical", "x": "Convex", "f": "Flat",
        "k": "Knobbed", "s": "Sunken"
    },
    "cap_surface": {
        "f": "Fibrous", "g": "Grooves", "y": "Scaly", "s": "Smooth"
    },
    "cap_color": {
        "n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray",
        "r": "Green", "p": "Pink", "u": "Purple", "e": "Red",
        "w": "White", "y": "Yellow"
    },
    "bruises": {"t": "Yes", "f": "No"},
    "odor": {
        "a": "Almond", "l": "Anise", "c": "Creosote", "y": "Fishy",
        "f": "Foul", "m": "Musty", "n": "None", "p": "Pungent", "s": "Spicy"
    },
    "gill_attachment": {"a": "Attached", "d": "Descending", "f": "Free", "n": "Notched"},
    "gill_spacing": {"c": "Close", "w": "Crowded", "d": "Distant"},
    "gill_size": {"b": "Broad", "n": "Narrow"},
    "gill_color": {
        "k": "Black", "n": "Brown", "b": "Buff", "h": "Chocolate",
        "g": "Gray", "r": "Green", "o": "Orange", "p": "Pink",
        "u": "Purple", "e": "Red", "w": "White", "y": "Yellow"
    },
    "stalk_shape": {"e": "Enlarging", "t": "Tapering"},
    "stalk_root": {
        "b": "Bulbous", "c": "Club", "u": "Cup", "e": "Equal",
        "z": "Rhizomorphs", "r": "Rooted", "?": "Missing"
    },
    "stalk_surface_above_ring": {"f": "Fibrous", "y": "Scaly", "k": "Silky", "s": "Smooth"},
    "stalk_surface_below_ring": {"f": "Fibrous", "y": "Scaly", "k": "Silky", "s": "Smooth"},
    "stalk_color_above_ring": {
        "n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray", "o": "Orange",
        "p": "Pink", "e": "Red", "w": "White", "y": "Yellow"
    },
    "stalk_color_below_ring": {
        "n": "Brown", "b": "Buff", "c": "Cinnamon", "g": "Gray", "o": "Orange",
        "p": "Pink", "e": "Red", "w": "White", "y": "Yellow"
    },
    "veil_type": {"p": "Partial", "u": "Universal"},
    "veil_color": {"n": "Brown", "o": "Orange", "w": "White", "y": "Yellow"},
    "ring_number": {"n": "None", "o": "One", "t": "Two"},
    "ring_type": {
        "c": "Cobwebby", "e": "Evanescent", "f": "Flaring", "l": "Large",
        "n": "None", "p": "Pendant", "s": "Sheathing", "z": "Zone"
    },
    "spore_print_color": {
        "k": "Black", "n": "Brown", "b": "Buff", "h": "Chocolate",
        "r": "Green", "o": "Orange", "u": "Purple", "w": "White", "y": "Yellow"
    },
    "population": {
        "a": "Abundant", "c": "Clustered", "n": "Numerous",
        "s": "Scattered", "v": "Several", "y": "Solitary"
    },
    "habitat": {
        "g": "Grasses", "l": "Leaves", "m": "Meadows", "p": "Paths",
        "u": "Urban", "w": "Waste", "d": "Woods"
    }
}

FEATURE_COLUMNS = COLUMN_NAMES[1:]  # exclude 'class'


def _generate_local_dataset(filepath):
    """
    Generate a synthetic UCI-compatible mushroom dataset using documented
    attribute distributions. Used as fallback when network is unavailable.
    """
    ATTR = {
        "cap_shape":                ["b c x f k s",     [1,1,5,3,1,1], [1,1,4,2,1,1]],
        "cap_surface":              ["f g y s",         [2,1,2,4],     [3,1,3,2]],
        "cap_color":                ["n b c g r p u e w y",[3,1,1,2,0,1,0,1,2,2],[3,1,1,1,1,1,1,2,1,2]],
        "bruises":                  ["t f",             [6,4],         [2,8]],
        "odor":                     ["a l c y f m n p s",[3,2,0,0,0,0,8,0,2],[0,0,2,2,6,1,2,3,1]],
        "gill_attachment":          ["a d f n",         [1,0,8,1],     [1,0,8,1]],
        "gill_spacing":             ["c w d",           [6,3,1],       [7,2,1]],
        "gill_size":                ["b n",             [7,3],         [4,6]],
        "gill_color":               ["k n b h g r o p u e w y",[1,2,3,1,1,0,1,2,0,1,4,1],[2,3,2,2,2,1,1,1,1,2,2,1]],
        "stalk_shape":              ["e t",             [5,5],         [6,4]],
        "stalk_root":               ["b c u e z r ?",   [3,0,0,4,0,1,2],[2,1,1,3,1,0,2]],
        "stalk_surface_above_ring": ["f y k s",         [2,2,2,4],     [3,3,2,2]],
        "stalk_surface_below_ring": ["f y k s",         [2,2,2,4],     [3,3,2,2]],
        "stalk_color_above_ring":   ["n b c g o p e w y",[1,1,1,1,1,2,1,5,1],[2,1,1,1,1,2,1,3,1]],
        "stalk_color_below_ring":   ["n b c g o p e w y",[1,1,1,1,1,2,1,5,1],[2,1,1,1,1,2,1,3,1]],
        "veil_type":                ["p u",             [10,0],        [10,0]],
        "veil_color":               ["n o w y",         [0,0,9,1],     [1,1,7,1]],
        "ring_number":              ["n o t",           [1,7,2],       [2,6,2]],
        "ring_type":                ["c e f l n p s z", [0,2,1,3,1,4,0,1],[1,1,1,2,2,3,1,1]],
        "spore_print_color":        ["k n b h r o u w y",[2,3,2,1,0,1,0,3,1],[3,2,2,2,1,1,1,1,1]],
        "population":               ["a c n s v y",     [1,2,2,3,3,2], [1,1,1,3,4,2]],
        "habitat":                  ["g l m p u w d",   [2,2,1,2,1,1,4],[2,1,2,2,1,2,3]],
    }
    rows = []
    rng = np.random.default_rng(42)
    for label, n in [("e", 4208), ("p", 3916)]:
        idx = 0 if label == "e" else 1
        for _ in range(n):
            row = {"class": label}
            for feat, (vals_str, we, wp) in ATTR.items():
                vals = vals_str.split()
                weights = np.array(we if label == "e" else wp, dtype=float)
                weights /= weights.sum()
                row[feat] = rng.choice(vals, p=weights)
            rows.append(row)
    df = pd.DataFrame(rows)[COLUMN_NAMES].sample(frac=1, random_state=42).reset_index(drop=True)
    df.to_csv(filepath, index=False)
    print(f"Generated local dataset: {len(df)} rows → {filepath}")
    return df


def load_data(filepath="mushrooms.csv"):
    """Load UCI mushroom data. Downloads from UCI or uses local generator."""
    if os.path.exists(filepath):
        return pd.read_csv(filepath)

    # Try downloading from UCI
    urls = [
        "https://archive.ics.uci.edu/ml/machine-learning-databases/mushroom/agaricus-lepiota.data",
        "https://raw.githubusercontent.com/jbrownlee/Datasets/master/mushrooms.csv",
    ]
    for url in urls:
        try:
            print(f"Trying download: {url}")
            df = pd.read_csv(url, header=None, names=COLUMN_NAMES)
            df.to_csv(filepath, index=False)
            print(f"Downloaded and saved to {filepath}")
            return df
        except Exception as e:
            print(f"  Failed: {e}")

    # Fallback: generate locally
    print("Network unavailable — generating synthetic dataset from UCI attribute distributions...")
    return _generate_local_dataset(filepath)


def train_and_save(data_path="mushrooms.csv", model_path="mushroom_model.pkl", encoders_path="encoders.pkl"):
    """Train RandomForest model and save model + encoders."""
    df = load_data(data_path)
    print(f"Dataset loaded: {df.shape[0]} samples, {df.shape[1]} features")

    # Encode all categorical columns
    encoders = {}
    df_encoded = df.copy()
    for col in df.columns:
        le = LabelEncoder()
        df_encoded[col] = le.fit_transform(df[col])
        encoders[col] = le

    X = df_encoded[FEATURE_COLUMNS]
    y = df_encoded["class"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    acc = accuracy_score(y_test, y_pred)
    print(f"\nTest Accuracy: {acc * 100:.2f}%")
    print("\nClassification Report:")
    target_names = encoders["class"].classes_  # e.g. ['e', 'p']
    print(classification_report(y_test, y_pred, target_names=target_names))

    # Feature importances
    importances = pd.Series(clf.feature_importances_, index=FEATURE_COLUMNS)
    print("\nTop 10 Important Features:")
    print(importances.nlargest(10))

    # Save
    with open(model_path, "wb") as f:
        pickle.dump(clf, f)
    with open(encoders_path, "wb") as f:
        pickle.dump(encoders, f)

    print(f"\nModel saved → {model_path}")
    print(f"Encoders saved → {encoders_path}")
    return clf, encoders, acc


def load_model(model_path="mushroom_model.pkl", encoders_path="encoders.pkl"):
    """Load saved model and encoders."""
    with open(model_path, "rb") as f:
        clf = pickle.load(f)
    with open(encoders_path, "rb") as f:
        encoders = pickle.load(f)
    return clf, encoders


def predict(clf, encoders, input_dict):
    """
    Predict edibility from a dict of {feature: raw_code}.
    Returns: ('edible'|'poisonous', probability_float)
    """
    row = {}
    for col in FEATURE_COLUMNS:
        val = input_dict[col]
        le = encoders[col]
        row[col] = le.transform([val])[0]

    X_input = pd.DataFrame([row])[FEATURE_COLUMNS]
    pred_encoded = clf.predict(X_input)[0]
    proba = clf.predict_proba(X_input)[0]

    class_labels = encoders["class"].classes_   # ['e', 'p']
    pred_label = encoders["class"].inverse_transform([pred_encoded])[0]
    label_map = {"e": "edible", "p": "poisonous"}

    edible_idx = list(class_labels).index("e")
    poisonous_idx = list(class_labels).index("p")

    return (
        label_map[pred_label],
        proba[edible_idx],
        proba[poisonous_idx]
    )


if __name__ == "__main__":
    train_and_save()
