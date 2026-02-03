"""
dataPreprocessing.py

Creates:
- Data/PrecomputedMatrices/xTrain.npy
- Data/PrecomputedMatrices/yTrain.npy
- Data/PrecomputedMatrices/TeamVectors/{year}TeamVectors.npy (for recent years)

This script builds a per season team feature vector using Kaggle compact results plus seeds and conferences,
then creates a training matrix of matchup diffs + location feature.

Expected Kaggle files (paths can be changed via args):
- RegularSeasonCompactResults.csv
- NCAATourneyCompactResults.csv
- NCAATourneySeeds.csv
- Teams.csv
- TeamConferences.csv   (optional but recommended)

The feature set is intentionally simple and robust.
You can extend it by adding new columns in build_team_season_features().
"""

import argparse
import os
import math
import json
import numpy as np
import pandas as pd
from dataclasses import dataclass
from typing import Dict, List, Tuple


@dataclass(frozen=True)
class Paths:
    kaggle_dir: str
    out_dir: str


POWER_CONFS = {
    "sec", "acc", "big_ten", "big_twelve", "big_east", "pac_twelve",
    # Some datasets use different abbreviations, keep it forgiving:
    "b10", "b12", "sec_", "acc_", "p12"
}


def ensure_dirs(paths: Paths) -> None:
    os.makedirs(paths.out_dir, exist_ok=True)
    os.makedirs(os.path.join(paths.out_dir, "TeamVectors"), exist_ok=True)


def read_csv(paths: Paths, name: str) -> pd.DataFrame:
    p = os.path.join(paths.kaggle_dir, name)
    if not os.path.exists(p):
        raise FileNotFoundError(f"Missing file: {p}")
    return pd.read_csv(p)


def seed_to_int(seed: str) -> int:
    """
    Kaggle seeds look like 'W01', 'X16b', etc.
    We take the numeric part safely.
    If missing, return 25 (meaning not in tournament / unknown).
    """
    if pd.isna(seed):
        return 25
    s = str(seed)
    digits = ""
    for ch in s:
        if ch.isdigit():
            digits += ch
    if digits == "":
        return 25
    val = int(digits[:2]) if len(digits) >= 2 else int(digits)
    return val


def wloc_to_home_val(wloc: str) -> int:
    # Winner location: H, A, N. Convert to +1, -1, 0.
    if wloc == "H":
        return 1
    if wloc == "A":
        return -1
    return 0


def build_team_season_features(
    season: int,
    reg: pd.DataFrame,
    seeds: pd.DataFrame,
    team_confs: pd.DataFrame | None
) -> pd.DataFrame:
    """
    Returns a DataFrame indexed by TeamID with columns for this season's features.

    Features included (simple but useful):
    - games, wins, losses, win_pct
    - avg_score_for, avg_score_against, margin
    - seed (tournament seed or 25)
    - power_conf (0/1) from TeamConferences if available
    """
    reg_s = reg[reg["Season"] == season].copy()

    # Winner rows
    w = reg_s[["Season", "WTeamID", "WScore", "LScore", "WLoc"]].copy()
    w.rename(columns={"WTeamID": "TeamID", "WScore": "ScoreFor", "LScore": "ScoreAgainst"}, inplace=True)
    w["Win"] = 1

    # Loser rows (flip scores)
    l = reg_s[["Season", "LTeamID", "LScore", "WScore", "WLoc"]].copy()
    l.rename(columns={"LTeamID": "TeamID", "LScore": "ScoreFor", "WScore": "ScoreAgainst"}, inplace=True)
    l["Win"] = 0

    games = pd.concat([w, l], ignore_index=True)

    agg = games.groupby("TeamID").agg(
        games=("Win", "count"),
        wins=("Win", "sum"),
        score_for=("ScoreFor", "sum"),
        score_against=("ScoreAgainst", "sum"),
    ).reset_index()

    agg["losses"] = agg["games"] - agg["wins"]
    agg["win_pct"] = agg["wins"] / agg["games"]
    agg["avg_score_for"] = agg["score_for"] / agg["games"]
    agg["avg_score_against"] = agg["score_against"] / agg["games"]
    agg["margin"] = agg["avg_score_for"] - agg["avg_score_against"]

    # Seeds
    seeds_s = seeds[seeds["Season"] == season][["TeamID", "Seed"]].copy()
    seeds_s["seed"] = seeds_s["Seed"].apply(seed_to_int)
    seeds_s = seeds_s[["TeamID", "seed"]]

    out = agg.merge(seeds_s, on="TeamID", how="left")
    out["seed"] = out["seed"].fillna(25).astype(int)

    # Power conference flag (optional)
    out["power_conf"] = 0
    if team_confs is not None and len(team_confs) > 0:
        tc = team_confs[team_confs["Season"] == season][["TeamID", "ConfAbbrev"]].copy()
        tc["ConfAbbrev"] = tc["ConfAbbrev"].astype(str).str.lower()
        tc["power_conf"] = tc["ConfAbbrev"].apply(lambda x: 1 if x in POWER_CONFS else 0)
        tc = tc[["TeamID", "power_conf"]]
        out = out.drop(columns=["power_conf"]).merge(tc, on="TeamID", how="left")
        out["power_conf"] = out["power_conf"].fillna(0).astype(int)

    # Final feature columns for vectors
    # Keep this list stable so your xTrain has consistent meaning.
    feature_cols = [
        "wins",
        "losses",
        "win_pct",
        "avg_score_for",
        "avg_score_against",
        "margin",
        "power_conf",
        "seed",
    ]

    out = out[["TeamID"] + feature_cols].copy()
    return out


def build_team_vectors_for_year(
    season: int,
    reg: pd.DataFrame,
    seeds: pd.DataFrame,
    team_confs: pd.DataFrame | None
) -> Dict[int, List[float]]:
    df = build_team_season_features(season, reg, seeds, team_confs)

    feature_cols = [c for c in df.columns if c != "TeamID"]
    vectors = {}
    for _, row in df.iterrows():
        tid = int(row["TeamID"])
        vec = [float(row[c]) for c in feature_cols]
        vectors[tid] = vec

    return vectors


def build_matchups_matrix(
    season: int,
    games_df: pd.DataFrame,
    team_vectors: Dict[int, List[float]],
    neutral_override: bool = False
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Creates training rows from game results:
    - X row is (winner_vec - loser_vec) + [home]
    - y is 1 for winner
    Also adds symmetric examples to reduce ordering bias:
    - X row is (loser_vec - winner_vec) + [-home]
    - y is 0
    """
    g = games_df[games_df["Season"] == season].copy()

    # Determine home feature for each game.
    # In tournament data, location is neutral.
    if neutral_override or "WLoc" not in g.columns:
        home_vals = np.zeros(len(g), dtype=np.float32)
    else:
        home_vals = g["WLoc"].astype(str).apply(wloc_to_home_val).to_numpy(dtype=np.float32)

    w_ids = g["WTeamID"].to_numpy(dtype=np.int32)
    l_ids = g["LTeamID"].to_numpy(dtype=np.int32)

    # Build arrays of vectors
    def vec_for(team_id: int) -> np.ndarray:
        v = team_vectors.get(int(team_id))
        if v is None:
            # Team missing stats, return zeros
            return None
        return np.asarray(v, dtype=np.float32)

    # Preallocate
    # Each game produces 2 rows (original + symmetric)
    any_vec = next(iter(team_vectors.values()))
    d = len(any_vec) + 1  # + home
    X = np.zeros((len(g) * 2, d), dtype=np.float32)
    y = np.zeros((len(g) * 2,), dtype=np.int8)

    row = 0
    for i in range(len(g)):
        wv = team_vectors.get(int(w_ids[i]))
        lv = team_vectors.get(int(l_ids[i]))
        if wv is None or lv is None:
            continue

        wv = np.asarray(wv, dtype=np.float32)
        lv = np.asarray(lv, dtype=np.float32)

        diff = wv - lv
        home = home_vals[i]

        X[row, :-1] = diff
        X[row, -1] = home
        y[row] = 1
        row += 1

        # symmetric example
        X[row, :-1] = -diff
        X[row, -1] = -home
        y[row] = 0
        row += 1

    # Trim if some games skipped
    X = X[:row]
    y = y[:row]
    return X, y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kaggle_dir", default="Data/KaggleData", help="Folder with Kaggle csv files")
    ap.add_argument("--out_dir", default="Data/PrecomputedMatrices", help="Output folder for npy files")
    ap.add_argument("--end_year", type=int, required=True, help="Last season year to include in training set")
    ap.add_argument("--start_year", type=int, default=2003, help="Start season year to include")
    ap.add_argument("--save_last_n_years", type=int, default=5, help="Save team vectors for last N years")
    args = ap.parse_args()

    paths = Paths(kaggle_dir=args.kaggle_dir, out_dir=args.out_dir)
    ensure_dirs(paths)

    reg = read_csv(paths, "RegularSeasonCompactResults.csv")
    tourney = read_csv(paths, "NCAATourneyCompactResults.csv")
    seeds = read_csv(paths, "NCAATourneySeeds.csv")
    teams = read_csv(paths, "Teams.csv")

    team_confs = None
    tc_path = os.path.join(paths.kaggle_dir, "TeamConferences.csv")
    if os.path.exists(tc_path):
        team_confs = pd.read_csv(tc_path)

    start_year = args.start_year
    end_year = args.end_year
    if end_year < start_year:
        raise ValueError("end_year must be >= start_year")

    years = list(range(start_year, end_year + 1))
    save_years = list(range(end_year - (args.save_last_n_years - 1), end_year + 1))

    all_X = []
    all_y = []

    for yr in years:
        team_vectors = build_team_vectors_for_year(yr, reg, seeds, team_confs)

        # Regular season games
        X1, y1 = build_matchups_matrix(yr, reg, team_vectors, neutral_override=False)
        # Tournament games (neutral)
        X2, y2 = build_matchups_matrix(yr, tourney, team_vectors, neutral_override=True)

        X = np.vstack([X1, X2]) if len(X1) and len(X2) else (X1 if len(X1) else X2)
        y = np.concatenate([y1, y2]) if len(y1) and len(y2) else (y1 if len(y1) else y2)

        all_X.append(X)
        all_y.append(y)

        print(f"Finished year {yr}: X={X.shape}, y={y.shape}")

        if yr in save_years:
            out_tv = os.path.join(paths.out_dir, "TeamVectors", f"{yr}TeamVectors.npy")
            np.save(out_tv, team_vectors, allow_pickle=True)
            print(f"Saved team vectors: {out_tv}")

    xTrain = np.vstack(all_X)
    yTrain = np.concatenate(all_y)

    out_x = os.path.join(paths.out_dir, "xTrain.npy")
    out_y = os.path.join(paths.out_dir, "yTrain.npy")
    np.save(out_x, xTrain)
    np.save(out_y, yTrain)

    meta = {
        "start_year": start_year,
        "end_year": end_year,
        "rows": int(xTrain.shape[0]),
        "cols": int(xTrain.shape[1]),
        "notes": "Columns are team_feature_diff plus home feature at end.",
    }
    with open(os.path.join(paths.out_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved: {out_x}")
    print(f"Saved: {out_y}")
    print("Done.")


if __name__ == "__main__":
    main()
