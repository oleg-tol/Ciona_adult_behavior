#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Compute per-animal, per-state shape metrics from DLC x/y data
and a state-labelled events DataFrame.

Metrics per (animal, state):
    EllipseAspectRatio  – ratio of principal axis lengths from PCA
    Solidity            – polygon area / convex hull area
    BodyAngle           – principal body axis angle in [0, 90] degrees
"""
import numpy as np
import pandas as pd
from shapely.geometry import Polygon

STATES = ["Quiescence", "Half Contraction", "Full Contraction"]


def build_frame_lookup(df_x: pd.DataFrame) -> tuple[dict, int]:
    """Return ({frame_index: column_label}, offset) from the t* columns of df_x."""
    skip = ('ExperimentID', 'Individuals', 'Bodyparts', 'Condition', 'Specific', 'Coordinates')
    coord_cols = [c for c in df_x.columns if c not in skip]
    lookup = {}
    for c in coord_cols:
        s = str(c).lstrip('t')
        if s.isdigit():
            lookup[int(s)] = c
    if not lookup:
        raise RuntimeError(f"No frame columns detected: {coord_cols[:10]}")
    offset = min(lookup)
    return lookup, offset


def extract_coords(df_x: pd.DataFrame, df_y: pd.DataFrame,
                   exp: str, indiv: str, label: str) -> list[tuple]:
    """Return list of (x, -y) landmark coords for one animal at one frame."""
    xs = df_x.query("ExperimentID==@exp & Individuals==@indiv")[label].astype(float).values
    ys = df_y.query("ExperimentID==@exp & Individuals==@indiv")[label].astype(float).values
    # invert y so that dorsal is up
    return list(zip(xs, -ys))


def compute_shape_metrics(df_x: pd.DataFrame, df_y: pd.DataFrame,
                           events_df: pd.DataFrame,
                           states: list = STATES) -> pd.DataFrame:
    """
    For each animal × condition, find the midpoint frame of each behavioural
    state and compute EllipseAspectRatio, Solidity, and BodyAngle.

    Parameters
    ----------
    df_x, df_y  : filtered x/y DataFrames (must contain Bodyparts rows)
    events_df   : DataFrame with columns ExperimentID, Individuals, Condition,
                  RealID, State, StartFrame, EndFrame
    states      : list of state labels to process

    Returns
    -------
    shape_df : one row per (animal × state)
    """
    frame_to_label, offset = build_frame_lookup(df_x)

    unique_animals = (
        events_df[["ExperimentID", "Individuals", "Condition", "RealID"]]
        .drop_duplicates()
    )

    records = []
    for exp, indiv, cond, real in unique_animals.itertuples(index=False, name=None):
        evs = events_df.query(
            "ExperimentID == @exp & Individuals == @indiv & Condition == @cond"
        )

        # Find the midpoint frame label for each state
        idx_map = {}
        ok = True
        for st in states:
            sub = evs[evs["State"] == st]
            if sub.empty:
                ok = False
                break
            mid_frame = int((sub.StartFrame.iloc[0] + sub.EndFrame.iloc[0]) // 2)
            lab = frame_to_label.get(offset + mid_frame)
            if lab is None:
                ok = False
                break
            idx_map[st] = lab
        if not ok:
            continue

        for st, lab in idx_map.items():
            coords = extract_coords(df_x, df_y, exp, indiv, lab)
            if len(coords) < 3:
                continue

            poly = Polygon(coords)
            if (not poly.is_valid) or (poly.area <= 0):
                continue

            arr = np.array(coords)
            cov = np.cov(arr.T)
            evals, evecs = np.linalg.eig(cov)
            lam_max, lam_min = evals.max(), evals.min()
            ear = np.sqrt(lam_max / lam_min) if lam_min > 0 else np.nan
            sol = poly.area / poly.convex_hull.area if poly.convex_hull.area > 0 else np.nan

            principal = evecs[:, np.argmax(evals)]
            ang       = np.degrees(np.arctan2(principal[1], principal[0]))
            body_ang  = abs(((ang + 90) % 180) - 90)

            records.append({
                "RealID":             real,
                "ExperimentID":       exp,
                "Individuals":        indiv,
                "Condition":          cond,
                "State":              st,
                "EllipseAspectRatio": ear,
                "Solidity":           sol,
                "BodyAngle":          body_ang,
            })

    shape_df = pd.DataFrame(records)
    print("Shape metrics DF shape:", shape_df.shape)
    print(shape_df.head())
    return shape_df


def main():
    X_PATH     = './output_data/Ciona_adult_plotting/x_final.csv'    # ← edit
    Y_PATH     = './output_data/Ciona_adult_plotting/y_final.csv'    # ← edit
    EVENTS_PATH = './output_data/events_df.csv'                      # ← edit
    OUTDIR     = './output_data/shape_metrics'
    OUTFILE    = 'shape_metrics.csv'

    import os
    os.makedirs(OUTDIR, exist_ok=True)

    df_x      = pd.read_csv(X_PATH)
    df_y      = pd.read_csv(Y_PATH)
    events_df = pd.read_csv(EVENTS_PATH)

    shape_df = compute_shape_metrics(df_x, df_y, events_df)
    out_path = os.path.join(OUTDIR, OUTFILE)
    shape_df.to_csv(out_path, index=False, float_format='%.4f')
    print(out_path)


if __name__ == '__main__':
    main()
