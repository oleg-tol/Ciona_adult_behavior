#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
HMM State Discovery
===================
Feature preparation from alpha-centred PCA data, then Gaussian HMM fitting
with multiple restarts and post-hoc smoothing.

Steps:
    1. Build HMM features: NPC + dNPC + speed + acceleration
    2. Within-animal z-score normalisation
    3. Cap frames per animal, downsample, build sequences
    4. Train/test split → fit sticky GaussianHMM (multiple restarts)
    5. Decode all sequences with post-hoc smoothing
    6. Save model parameters, decoded states, dwell summaries
"""
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.ndimage import median_filter
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from hmmlearn.hmm import GaussianHMM

warnings.filterwarnings('ignore')


# ── Feature preparation ───────────────────────────────────────────────────────

def build_hmm_features(df: pd.DataFrame, npc_cols: list) -> pd.DataFrame:
    """
    Add per-animal temporal derivatives, speed, and acceleration.
    NaNs from diff() at sequence starts are filled with 0.
    """
    df = df.copy()
    df['animal_id'] = df['ExperimentID'].astype(str) + '_' + df['Individuals'].astype(str)

    # Numeric timepoint for sorting
    tp = df['Timepoint'].astype(str)
    df['_tp_num'] = pd.to_numeric(tp.str.replace('t', '', regex=False), errors='coerce')
    if df['_tp_num'].isna().mean() > 0.5:
        df['_tp_num'] = 0
    df = df.sort_values(['ExperimentID', 'Individuals', '_tp_num'],
                         kind='mergesort').reset_index(drop=True)

    # Per-animal dNPC
    for col in npc_cols:
        df[f'd{col}'] = df.groupby('animal_id')[col].diff()

    dcols       = [f'd{c}' for c in npc_cols]
    df['speed'] = np.sqrt(np.sum(df[dcols].to_numpy() ** 2, axis=1))
    df['acc']   = df.groupby('animal_id')['speed'].diff()

    fill_cols = dcols + ['speed', 'acc']
    df[fill_cols] = df[fill_cols].fillna(0.0)

    return df, dcols


def zscore_within_animal(df: pd.DataFrame, feat_cols: list) -> pd.DataFrame:
    """Z-score each feature within each animal (prevents individual-level offsets)."""
    def _z(g):
        x  = g[feat_cols].to_numpy(dtype=np.float32)
        mu = np.nanmean(x, axis=0)
        sd = np.nanstd(x, axis=0)
        sd[sd == 0] = 1.0
        g = g.copy(); g[feat_cols] = (x - mu) / sd
        return g
    return df.groupby('animal_id', group_keys=False).apply(_z)


# ── Sequence building ─────────────────────────────────────────────────────────

def build_sequences(df: pd.DataFrame, feat_cols: list,
                     cap_frames: int, downsample: int,
                     min_seq_len: int, random_state: int) -> tuple:
    """
    Cap frames per animal, downsample, filter short sequences.
    Returns (X_list, meta, seq_lengths).
    meta = list of (animal_id, condition, specific)
    """
    # Cap
    parts = []
    for aid, g in df.groupby('animal_id'):
        parts.append(g if len(g) <= cap_frames else
                     g.sample(n=cap_frames, random_state=random_state))
    df_c = pd.concat(parts, ignore_index=True)

    # Downsample within each animal (preserve temporal order)
    df_c = (df_c.groupby('animal_id', group_keys=False)
                .apply(lambda g: g.iloc[::downsample])
                .reset_index(drop=True))

    X_list, meta = [], []
    for aid, g in df_c.groupby('animal_id'):
        if len(g) < min_seq_len:
            continue
        cond = g['Condition'].iat[0] if 'Condition' in g.columns else None
        spec = g['Specific'].iat[0]  if 'Specific'  in g.columns else None
        X_list.append(g[feat_cols].to_numpy(dtype=float))
        meta.append((aid, cond, spec))

    seq_lengths = [len(X) for X in X_list]
    return X_list, meta, seq_lengths


# ── HMM initialisation helpers ────────────────────────────────────────────────

def init_transmat(k: int, self_p: float) -> np.ndarray:
    off = (1.0 - self_p) / max(k - 1, 1)
    T   = np.full((k, k), off, dtype=float)
    np.fill_diagonal(T, self_p)
    return T / T.sum(axis=1, keepdims=True)


def init_params_kmeans(X: np.ndarray, k: int, seed: int,
                        cov_type: str) -> tuple:
    D      = X.shape[1]
    km     = KMeans(n_clusters=k, random_state=seed, n_init=10)
    labels = km.fit_predict(X)
    means  = km.cluster_centers_
    eps    = 1e-6

    if cov_type == 'full':
        covars = np.stack([
            np.cov(X[labels == k_], rowvar=False) + np.eye(D) * eps
            if (labels == k_).sum() > 1 else np.eye(D) * eps * 10
            for k_ in range(k)
        ])
    else:
        covars = np.stack([
            np.clip(np.var(X[labels == k_], axis=0), eps, None)
            if (labels == k_).sum() > 1 else np.ones(D) * eps * 10
            for k_ in range(k)
        ])

    counts = np.bincount(labels, minlength=k).astype(float)
    counts[counts == 0] = 1e-8
    return means, covars, counts / counts.sum()


# ── HMM training ─────────────────────────────────────────────────────────────

def fit_hmm(X_train: np.ndarray, L_train: list,
             X_test: np.ndarray, L_test: list,
             k: int, cov_type: str, n_iter: int, tol: float,
             n_restarts: int, self_p_init: float,
             alpha_stay: float, alpha_off: float,
             random_state: int) -> tuple:
    """
    Fit GaussianHMM with k-means init and multiple restarts.
    Returns (best_model, train_ll, test_ll, restart_log_df).
    """
    best_model, best_test, best_train = None, -np.inf, -np.inf
    rows = []

    trans_prior = np.full((k, k), alpha_off, dtype=float)
    np.fill_diagonal(trans_prior, alpha_stay)

    for r in range(n_restarts):
        seed = random_state + r
        print(f'  Restart {r + 1}/{n_restarts} (seed={seed})…', end=' ', flush=True)

        means, covars, startprob = init_params_kmeans(X_train, k, seed, cov_type)

        model = GaussianHMM(n_components=k, covariance_type=cov_type,
                             n_iter=n_iter, tol=tol,
                             random_state=seed, verbose=False)
        if hasattr(model, 'transmat_prior'):
            model.transmat_prior = trans_prior
        if hasattr(model, 'startprob_prior'):
            model.startprob_prior = np.ones(k)

        model.startprob_ = startprob
        model.transmat_  = init_transmat(k, self_p_init)
        model.means_     = means
        model.covars_    = covars
        model.init_params = ''

        try:
            model.fit(X_train, lengths=L_train)
            tr_ll = model.score(X_train, lengths=L_train)
            te_ll = model.score(X_test,  lengths=L_test)
            print(f'train_ll={tr_ll:.2f}  test_ll={te_ll:.2f}')
            rows.append({'restart': r, 'seed': seed,
                          'train_ll': float(tr_ll), 'test_ll': float(te_ll),
                          'converged': True, 'error': ''})
            if np.isfinite(te_ll) and te_ll > best_test:
                best_test, best_train, best_model = te_ll, tr_ll, model
                print(f'    ★ new best (test_ll={best_test:.2f})')
        except Exception as e:
            print(f'FAILED: {str(e)[:60]}')
            rows.append({'restart': r, 'seed': seed,
                          'train_ll': np.nan, 'test_ll': np.nan,
                          'converged': False, 'error': str(e)})

    return best_model, best_train, best_test, pd.DataFrame(rows)


# ── Post-hoc smoothing ────────────────────────────────────────────────────────

def smooth_state_sequence(states: np.ndarray,
                           median_filter_size: int,
                           min_bout_frames: int) -> np.ndarray:
    states = median_filter(np.asarray(states, dtype=int),
                            size=median_filter_size, mode='nearest')
    # Merge short bouts with longer neighbour
    runs = []
    cur, start = int(states[0]), 0
    for i in range(1, len(states)):
        s = int(states[i])
        if s != cur:
            runs.append([cur, start, i]); cur, start = s, i
    runs.append([cur, start, len(states)])

    out = states.copy()
    for i, (st, s, e) in enumerate(runs):
        if (e - s) < min_bout_frames:
            prev = runs[i - 1][0] if i > 0              else None
            nxt  = runs[i + 1][0] if i < len(runs) - 1 else None
            if prev is not None and nxt is not None:
                pd_ = runs[i - 1][2] - runs[i - 1][1]
                nd_ = runs[i + 1][2] - runs[i + 1][1]
                rep = prev if pd_ >= nd_ else nxt
            else:
                rep = prev if prev is not None else (nxt if nxt is not None else st)
            out[s:e] = rep
    return out


# ── Dwell time analysis ───────────────────────────────────────────────────────

def compute_dwell_runs(states: np.ndarray) -> dict:
    runs = {int(s): [] for s in np.unique(states)}
    cur, run = int(states[0]), 1
    for s in states[1:]:
        s = int(s)
        if s == cur:
            run += 1
        else:
            runs[cur].append(run); cur, run = s, 1
    runs[cur].append(run)
    return runs


def summarize_dwells(dwell_dict: dict, dt: float) -> pd.DataFrame:
    rows = []
    for s, runs in dwell_dict.items():
        if not runs:
            continue
        arr = np.asarray(runs, dtype=float) * dt
        rows.append({'state': int(s), 'n_runs': len(arr),
                      'min_s':    float(arr.min()),
                      'p25_s':    float(np.percentile(arr, 25)),
                      'median_s': float(np.median(arr)),
                      'mean_s':   float(arr.mean()),
                      'p75_s':    float(np.percentile(arr, 75)),
                      'p95_s':    float(np.percentile(arr, 95)),
                      'max_s':    float(arr.max())})
    return pd.DataFrame(rows).sort_values('state').reset_index(drop=True)


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_transition_matrix(transmat: np.ndarray, out_dir: str):
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(transmat, annot=True, fmt='.3f', cmap='Blues',
                vmin=0, vmax=1, ax=ax)
    ax.set_xlabel('To state'); ax.set_ylabel('From state')
    ax.set_title('HMM Transition Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'transition_matrix.png'), dpi=300)
    plt.close()


def plot_dwell_comparison(dwell_raw: pd.DataFrame, dwell_smooth: pd.DataFrame,
                           out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=False)
    for ax, dwell, title in zip(axes,
                                 [dwell_raw, dwell_smooth],
                                 ['Raw dwell times', 'Smoothed dwell times']):
        ax.bar(dwell['state'], dwell['median_s'], yerr=dwell['p75_s'] - dwell['p25_s'],
               capsize=4, alpha=0.8)
        ax.set_xlabel('State'); ax.set_ylabel('Dwell time (s)')
        ax.set_title(title); ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'dwell_times.png'), dpi=300)
    plt.close()


def plot_state_occupancy(decoded_df: pd.DataFrame, out_dir: str):
    """Fraction of time spent in each state, per condition."""
    occ = (decoded_df.groupby(['Condition', 'state'])
                     .size()
                     .reset_index(name='n'))
    occ['frac'] = occ.groupby('Condition')['n'].transform(lambda x: x / x.sum())
    pivot = occ.pivot(index='state', columns='Condition', values='frac').fillna(0)

    fig, ax = plt.subplots(figsize=(8, 4))
    pivot.plot(kind='bar', ax=ax, alpha=0.8)
    ax.set_xlabel('State'); ax.set_ylabel('Fraction of frames')
    ax.set_title('State occupancy by condition')
    ax.grid(True, alpha=0.3, axis='y')
    ax.legend(title='Condition', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'state_occupancy_by_condition.png'), dpi=300)
    plt.close()


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Paths ────────────────────────────────────────────────────────────────
    FULL_PATH = '/work/oleg/Adult_data_new/Data/posture_clustering1/pca_all_good_frames_ALPHA0.20.parquet'  # ← edit
    OUT_DIR   = '/work/oleg/Adult_data_new/Data/posture_clustering1/HMM_K8'                                # ← edit
    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Parameters ───────────────────────────────────────────────────────────
    N_PCS_USE         = 5
    CAP_FRAMES        = 6000
    DOWNSAMPLE        = 5          # effective DT = frame_interval * DOWNSAMPLE
    MIN_SEQ_LEN       = 200        # frames after downsampling
    FRAME_INTERVAL    = 0.25       # seconds per original frame (FPS=20, skip=5)
    K                 = 8          # HMM states
    COV_TYPE          = 'diag'
    N_ITER            = 300
    TOL               = 1e-4
    N_RESTARTS        = 4
    SELF_P_INIT       = 0.998
    ALPHA_STAY        = 100.0
    ALPHA_OFF         = 1.0
    MEDIAN_FILT_SIZE  = 5
    MIN_BOUT_S        = 5.0
    RANDOM_STATE      = 139

    np.random.seed(RANDOM_STATE)
    DT = FRAME_INTERVAL * DOWNSAMPLE  # seconds per HMM step

    # ── Load ─────────────────────────────────────────────────────────────────
    print('Loading data…')
    df = pd.read_parquet(FULL_PATH)
    print(f'  Frames: {len(df):,}')

    npc_cols = [f'NPC{i}' for i in range(1, N_PCS_USE + 1)]
    for col in npc_cols:
        if col not in df.columns:
            raise ValueError(f'Missing feature column: {col}')

    has_specific = 'Specific' in df.columns
    print(f'  Has Specific column: {has_specific}')
    if 'Condition' in df.columns:
        print(f'  Animals per condition:')
        df['_aid'] = df['ExperimentID'].astype(str) + '_' + df['Individuals'].astype(str)
        print(df.groupby('Condition')['_aid'].nunique().to_string())
        if has_specific:
            cross = (df.groupby(['Condition', 'Specific'])['_aid']
                       .nunique().reset_index(name='N_animals'))
            cross.to_csv(os.path.join(OUT_DIR, 'condition_specific_breakdown.csv'), index=False)
            print(cross.to_string(index=False))

    # ── Step 1: Build HMM features ────────────────────────────────────────────
    print('\nBuilding HMM features…')
    df, dcols = build_hmm_features(df, npc_cols)
    feat_cols = npc_cols + dcols + ['speed', 'acc']
    print(f'  Feature columns ({len(feat_cols)}): {feat_cols}')

    # ── Step 2: Within-animal z-score ────────────────────────────────────────
    print('Z-scoring within animals…')
    df = zscore_within_animal(df, feat_cols)

    # Save HMM-ready features
    meta_keep = ['ExperimentID', 'Individuals', 'animal_id', 'Timepoint', '_tp_num']
    for col in ['Condition', 'Specific', 'Stage']:
        if col in df.columns:
            meta_keep.append(col)
    hmm_df = df[meta_keep + feat_cols].copy()
    hmm_out = os.path.join(OUT_DIR, f'hmm_features_NPC{N_PCS_USE}.parquet')
    hmm_df.to_parquet(hmm_out, index=False)
    print(f'  ✓ Saved: hmm_features_NPC{N_PCS_USE}.parquet')

    # ── Step 3: Build sequences ───────────────────────────────────────────────
    print('\nBuilding sequences…')
    X_list, meta, seq_lengths = build_sequences(
        hmm_df, feat_cols, CAP_FRAMES, DOWNSAMPLE, MIN_SEQ_LEN, RANDOM_STATE)
    print(f'  Sequences: {len(X_list)}  |  Total obs: {sum(seq_lengths):,}')
    print(f'  Length  min={min(seq_lengths)}  '
          f'median={int(np.median(seq_lengths))}  max={max(seq_lengths)} frames')
    print(f'  Duration min={min(seq_lengths)*DT:.0f}s  '
          f'median={np.median(seq_lengths)*DT:.0f}s  max={max(seq_lengths)*DT:.0f}s')

    seq_meta_df = pd.DataFrame(
        [(aid, cond, spec, n, n * DT)
         for (aid, cond, spec), n in zip(meta, seq_lengths)],
        columns=['animal_id', 'Condition', 'Specific', 'n_frames', 'duration_s'])
    seq_meta_df.to_csv(os.path.join(OUT_DIR, 'sequences_metadata.csv'), index=False)
    print('  ✓ Saved: sequences_metadata.csv')

    # ── Step 4: Train/test split + fit HMM ───────────────────────────────────
    print(f'\nFitting HMM (K={K}, restarts={N_RESTARTS})…')
    idx     = np.arange(len(X_list))
    tr_idx, te_idx = train_test_split(idx, test_size=0.2, random_state=RANDOM_STATE)

    X_train = np.vstack([X_list[i] for i in tr_idx])
    L_train = [len(X_list[i]) for i in tr_idx]
    X_test  = np.vstack([X_list[i] for i in te_idx])
    L_test  = [len(X_list[i]) for i in te_idx]
    print(f'  Train: {len(tr_idx)} seqs / {X_train.shape[0]:,} obs  |  '
          f'Test: {len(te_idx)} seqs / {X_test.shape[0]:,} obs')

    model, train_ll, test_ll, restart_df = fit_hmm(
        X_train, L_train, X_test, L_test,
        K, COV_TYPE, N_ITER, TOL, N_RESTARTS,
        SELF_P_INIT, ALPHA_STAY, ALPHA_OFF, RANDOM_STATE)

    restart_df.to_csv(os.path.join(OUT_DIR, 'restarts_log.csv'), index=False)
    print(f'\n  Best  train_ll={train_ll:.2f}  test_ll={test_ll:.2f}')
    print(f'  Per obs  train={train_ll / X_train.shape[0]:.4f}  '
          f'test={test_ll / X_test.shape[0]:.4f}')

    if model is None:
        raise RuntimeError('All HMM restarts failed.')

    diag = np.diag(model.transmat_)
    print(f'  Self-transition probs  min={diag.min():.4f}  '
          f'mean={diag.mean():.4f}  max={diag.max():.4f}')

    # Save model parameters
    for name, arr in [('transmat',  model.transmat_),
                       ('means',     model.means_),
                       ('covars',    model.covars_),
                       ('startprob', model.startprob_)]:
        np.save(os.path.join(OUT_DIR, f'model_{name}.npy'), arr)
    print('  ✓ Saved model parameters (npy)')

    plot_transition_matrix(model.transmat_, OUT_DIR)

    # ── Step 5: Decode all sequences ─────────────────────────────────────────
    print('\nDecoding all sequences…')
    min_bout_frames = max(1, int(MIN_BOUT_S / DT))
    states_raw      = [model.predict(X) for X in X_list]
    states_smooth   = [smooth_state_sequence(s, MEDIAN_FILT_SIZE, min_bout_frames)
                       for s in states_raw]

    # Dwell time summaries
    dwell_dict_raw = {}
    for seq in states_raw:
        for s, runs in compute_dwell_runs(seq).items():
            dwell_dict_raw.setdefault(s, []).extend(runs)

    dwell_dict_sm = {}
    for seq in states_smooth:
        for s, runs in compute_dwell_runs(seq).items():
            dwell_dict_sm.setdefault(s, []).extend(runs)

    dwell_raw    = summarize_dwells(dwell_dict_raw, DT)
    dwell_smooth = summarize_dwells(dwell_dict_sm,  DT)

    dwell_raw.to_csv(   os.path.join(OUT_DIR, 'dwell_raw.csv'),      index=False)
    dwell_smooth.to_csv(os.path.join(OUT_DIR, 'dwell_smoothed.csv'), index=False)
    print('  ✓ Saved: dwell_raw.csv  dwell_smoothed.csv')
    print('\n  Smoothed dwell times:')
    print(dwell_smooth[['state', 'n_runs', 'min_s', 'median_s', 'mean_s', 'p95_s']].to_string(index=False))

    plot_dwell_comparison(dwell_raw, dwell_smooth, OUT_DIR)

    # ── Step 6: Save decoded states ───────────────────────────────────────────
    print('\nSaving decoded states…')
    rows = []
    for i, (aid, cond, spec) in enumerate(meta):
        for state in states_smooth[i]:
            rows.append({'animal_id': aid, 'Condition': cond,
                          'Specific': spec, 'state': int(state)})
    decoded_df = pd.DataFrame(rows)
    decoded_df.to_parquet(os.path.join(OUT_DIR, 'decoded_states_all.parquet'), index=False)
    print(f'  ✓ Saved: decoded_states_all.parquet  ({len(decoded_df):,} frames)')

    if 'Condition' in decoded_df.columns:
        plot_state_occupancy(decoded_df, OUT_DIR)

    print(f'\nDone. Outputs → {OUT_DIR}')
    print(f'  K={K}  |  test_ll={test_ll:.2f}  |  mean bout={dwell_smooth["mean_s"].mean():.1f}s')


if __name__ == '__main__':
    main()

