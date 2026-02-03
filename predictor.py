"""
march_madness.py

1) Loads xTrain and yTrain
2) Trains a model
3) Evaluates it
4) Creates Kaggle submission file result.csv for Stage 1 or Stage 2

This is written to feel like a complete project file but stays clean and reproducible.
"""

import argparse
import os
import csv
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, log_loss, roc_auc_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.calibration import CalibratedClassifierCV
from joblib import dump, load


def load_training_set(precomputed_dir: str):
    x_path = os.path.join(precomputed_dir, "xTrain.npy")
    y_path = os.path.join(precomputed_dir, "yTrain.npy")
    if not os.path.exists(x_path) or not os.path.exists(y_path):
        raise FileNotFoundError(
            "Missing xTrain or yTrain. Run dataPreprocessing.py first.\n"
            f"Expected:\n  {x_path}\n  {y_path}"
        )
    X = np.load(x_path)
    y = np.load(y_path)
    return X, y


def build_model():
    base = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.06,
        max_iter=350,
        random_state=42,
    )
    # Calibrate probabilities so outputs behave better for bracket style usage
    model = CalibratedClassifierCV(base, method="isotonic", cv=3)
    return model


def train_and_eval(X, y):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    model = build_model()
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, proba)
    ll = log_loss(y_test, proba)

    print(f"Accuracy: {acc:.4f}")
    print(f"AUC: {auc:.4f}")
    print(f"LogLoss: {ll:.4f}")
    print("\nClassification Report:\n")
    print(classification_report(y_test, preds, digits=4))

    return model


def load_team_vectors(precomputed_dir: str, years):
    vecs = {}
    for yr in years:
        p = os.path.join(precomputed_dir, "TeamVectors", f"{yr}TeamVectors.npy")
        if not os.path.exists(p):
            raise FileNotFoundError(f"Missing team vectors for year {yr}: {p}")
        vecs[yr] = np.load(p, allow_pickle=True).item()
    return vecs


def parse_matchup_id(matchup_id: str):
    """
    Kaggle ID format is typically:
    YYYY_Team1ID_Team2ID
    Sometimes the separator is underscore.
    """
    s = str(matchup_id)
    parts = s.split("_")
    if len(parts) != 3:
        # fallback: try slicing like old scripts
        year = int(s[0:4])
        t1 = int(s[5:9])
        t2 = int(s[10:14])
        return year, t1, t2
    return int(parts[0]), int(parts[1]), int(parts[2])


def predict_game(team1_vec, team2_vec, home_val, model):
    diff = np.asarray(team1_vec, dtype=np.float32) - np.asarray(team2_vec, dtype=np.float32)
    x = np.concatenate([diff, np.asarray([home_val], dtype=np.float32)])
    return float(model.predict_proba([x])[0][1])


def create_submission(
    kaggle_dir: str,
    precomputed_dir: str,
    model,
    stage2: bool,
    cur_year: int,
    out_csv: str = "result.csv"
):
    sub_file = "SampleSubmissionStage2.csv" if stage2 else "SampleSubmissionStage1.csv"
    sub_path = os.path.join(kaggle_dir, sub_file)

    if not os.path.exists(sub_path):
        raise FileNotFoundError(f"Missing submission template: {sub_path}")

    sub = pd.read_csv(sub_path)

    # Use year range like the original style:
    years = [cur_year] if stage2 else list(range(cur_year - 4, cur_year))
    team_vectors_by_year = load_team_vectors(precomputed_dir, years)
    print(f"Loaded team vectors for years: {years}")

    rows = []
    for _, r in sub.iterrows():
        matchup_id = r["ID"]
        year, t1, t2 = parse_matchup_id(matchup_id)

        # If submission includes years outside our loaded range, handle safely
        if year not in team_vectors_by_year:
            # try load directly if exists
            p = os.path.join(precomputed_dir, "TeamVectors", f"{year}TeamVectors.npy")
            if os.path.exists(p):
                team_vectors_by_year[year] = np.load(p, allow_pickle=True).item()
            else:
                # Default to 0.5 if we cannot build a vector
                rows.append((matchup_id, 0.5))
                continue

        vecs = team_vectors_by_year[year]
        v1 = vecs.get(int(t1))
        v2 = vecs.get(int(t2))
        if v1 is None or v2 is None:
            rows.append((matchup_id, 0.5))
            continue

        p = predict_game(v1, v2, home_val=0, model=model)
        p = min(max(p, 0.0), 1.0)
        rows.append((matchup_id, p))

    with open(out_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ID", "Pred"])
        w.writerows(rows)

    print(f"Wrote submission: {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle_dir", default="Data/KaggleData")
    ap.add_argument("--precomputed_dir", default="Data/PrecomputedMatrices")
    ap.add_argument("--year", type=int, required=True, help="Prediction year (example: 2026)")
    ap.add_argument("--stage2", action="store_true", help="Use Stage 2 submission file")
    ap.add_argument("--save_model", default="models/model.joblib")
    ap.add_argument("--no_submit", action="store_true", help="Train only, do not create result.csv")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.save_model), exist_ok=True)

    X, y = load_training_set(args.precomputed_dir)
    print(f"Loaded training set: X={X.shape}, y={y.shape}")

    model = train_and_eval(X, y)

    dump(model, args.save_model)
    print(f"Saved model: {args.save_model}")

    if not args.no_submit:
        create_submission(
            kaggle_dir=args.kaggle_dir,
            precomputed_dir=args.precomputed_dir,
            model=model,
            stage2=args.stage2,
            cur_year=args.year,
            out_csv="result.csv",
        )


if __name__ == "__main__":
    main()
