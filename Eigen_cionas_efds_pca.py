#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Eigen Ciona
===========
EFD extraction, harmonic amplitude computation, PCA (fit on discovery set,
project all good frames), and alpha-centering.

Steps:
    1. Identify common frames + build quality lookup (optional)
    2. Extract EFD coefficients in parallel
    3. Filter low-coverage animals
    4. Compute harmonic amplitudes (skip harmonic 1)
    5. Build balanced discovery set (fixed chemical doubling)
    6. PCA on discovery set → project all good frames
    7. Alpha-centering (partial per-animal normalisation)
"""
import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
from pyefd import elliptic_fourier_descriptors
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

warnings.filterwarnings('ignore')

META_BASE = ['ExperimentID', 'Individuals', 'Condition']


# ── Helpers ───────────────────────────────────────────────────────────────────

def frame_cols_n(df: pd.DataFrame, n_frames: int) -> list:
    """Return up to n_frames time columns (t0, t1, …) sorted numerically."""
    cols = [c for c in df.columns if isinstance(c, str) and re.fullmatch(r't\d+', c)]
    cols.sort(key=lambda s: int(s[1:]))
    return cols[:n_frames]


def build_quality_lookup(good_frames_df: pd.DataFrame,
                          common_frames: list) -> dict:
    """Return {(ExperimentID, Individuals): set_of_bad_frame_labels}."""
    lookup = {}
    for _, row in good_frames_df.iterrows():
        exp, ind = row['ExperimentID'], row['Individuals']
        n        = int(row['total_frames'])
        bad      = set()
        for i in range(n):
            if row.get(f'frame_{i}_bad', 0) == 1 and i < len(common_frames):
                bad.add(common_frames[i])
        lookup[(exp, ind)] = bad
    return lookup


# ── EFD extraction ────────────────────────────────────────────────────────────

def extract_efd_for_animal(animal_info: dict, common_frames: list,
                            grouped_x: dict, grouped_y: dict,
                            quality_lookup: dict | None,
                            efd_order: int) -> list:
    exp_id    = animal_info['ExperimentID']
    indiv_id  = animal_info['Individuals']
    condition = animal_info['Condition']
    specific  = animal_info.get('Specific', None)

    subx = grouped_x.get((exp_id, indiv_id))
    suby = grouped_y.get((exp_id, indiv_id))
    if subx is None or suby is None:
        return []

    bad_frames = set()
    if quality_lookup is not None:
        bad_frames = quality_lookup.get((exp_id, indiv_id), set())

    out = []
    for frame in common_frames:
        if frame in bad_frames:
            continue
        try:
            x = subx[frame].to_numpy(dtype=float, copy=False)
            y = suby[frame].to_numpy(dtype=float, copy=False)
        except Exception:
            continue

        ok = (~np.isnan(x)) & (~np.isnan(y))
        x, y = x[ok], y[ok]
        if x.size < 10:
            continue

        pts  = np.column_stack([x, -y])
        d    = np.diff(pts, axis=0)
        keep = np.ones(len(pts), dtype=bool)
        keep[1:] = np.any(d != 0, axis=1)
        pts  = pts[keep]
        if pts.shape[0] < 10:
            continue

        seg = np.diff(np.vstack([pts, pts[0]]), axis=0)
        if np.sum(np.hypot(seg[:, 0], seg[:, 1])) <= 1e-6:
            continue

        try:
            coeffs = elliptic_fourier_descriptors(pts, order=efd_order, normalize=False).ravel()
        except Exception:
            continue

        entry = {'ExperimentID': exp_id, 'Individuals': indiv_id,
                 'Condition': condition, 'Timepoint': frame}
        if specific is not None:
            entry['Specific'] = specific
        entry.update({f'EFD_{i}': v for i, v in enumerate(coeffs, start=1)})
        out.append(entry)

    return out


def build_efd_timeseries(df_x: pd.DataFrame, df_y: pd.DataFrame,
                          unique_animals: pd.DataFrame,
                          common_frames: list, efd_order: int,
                          quality_lookup: dict | None = None,
                          n_jobs: int = -1) -> pd.DataFrame:
    grouped_x = {(e, i): g for (e, i), g in df_x.groupby(['ExperimentID', 'Individuals'])}
    grouped_y = {(e, i): g for (e, i), g in df_y.groupby(['ExperimentID', 'Individuals'])}
    print(f'  Grouped {len(grouped_x)} animals')

    results = Parallel(n_jobs=n_jobs, verbose=5)(
        delayed(extract_efd_for_animal)(
            row.to_dict(), common_frames, grouped_x, grouped_y,
            quality_lookup, efd_order
        )
        for _, row in unique_animals.iterrows()
    )
    return pd.DataFrame([e for sub in results for e in sub])


# ── Harmonic amplitudes ───────────────────────────────────────────────────────

def compute_harmonic_amplitudes(efd_df: pd.DataFrame, efd_order: int,
                                 n_harmonics_use: int,
                                 meta_cols: list) -> tuple:
    """
    Returns (amp_df, amp_array_posture).
    amp_cols = amp_2 … amp_{n_harmonics_use+1}  (harmonic 1 skipped).
    """
    efd_cols = [c for c in efd_df.columns if c.startswith('EFD_')]
    E        = efd_df[efd_cols].values

    amp_all = []
    for n in range(efd_order):
        amp_all.append(np.linalg.norm(E[:, 4 * n : 4 * n + 4], axis=1))
    amp_array = np.vstack(amp_all).T                          # (frames, efd_order)

    amp_array_posture = amp_array[:, 1 : n_harmonics_use + 1]  # skip harmonic 1

    amp_df = efd_df[[c for c in meta_cols + ['Timepoint'] if c in efd_df.columns]].copy()
    for i in range(n_harmonics_use):
        amp_df[f'amp_{i + 2}'] = amp_array_posture[:, i]

    return amp_df, amp_array, amp_array_posture


# ── Discovery set ─────────────────────────────────────────────────────────────

def build_balanced_discovery(amp_df: pd.DataFrame,
                               has_specific: bool,
                               balance_mode: str,
                               min_animals: int,
                               frames_per_animal: int,
                               random_state: int = 42) -> pd.DataFrame:
    """
    balance_mode='per_condition'  – pool all Specifics within Condition
    balance_mode='per_subgroup'   – each Condition × Specific separately
    """
    base = amp_df.copy()

    if has_specific:
        base['Specific'] = base['Specific'].fillna('NA')
        # poke: keep only UP
        base = base[~(base['Condition'] == 'poke') | (base['Specific'] == 'UP')]
        # chemical: drop Water_Control
        base = base[~((base['Condition'] == 'chemical') &
                       (base['Specific'] == 'Water_Control'))]

    group_cols = (['Condition', 'Specific']
                  if balance_mode == 'per_subgroup' and has_specific
                  else ['Condition'])

    animal_counts = (base.groupby(group_cols)[['ExperimentID', 'Individuals']]
                     .apply(lambda x: x.drop_duplicates().shape[0])
                     .reset_index(name='n_animals'))
    print(f'\n  Animal counts per group ({balance_mode}):')
    print(animal_counts.to_string(index=False))

    valid = animal_counts[animal_counts['n_animals'] >= min_animals]
    print(f'\n  Including (≥{min_animals} animals):')
    print(valid.to_string(index=False))

    # filter to valid groups
    masks = []
    for _, row in valid.iterrows():
        m = pd.Series(True, index=base.index)
        for col in group_cols:
            m &= base[col] == row[col]
        masks.append(m)
    if not masks:
        raise RuntimeError('No groups meet minimum animal threshold.')
    disc = base[np.logical_or.reduce(masks)].copy()

    balanced = []
    for gvals, gdf in disc.groupby(group_cols):
        animals = gdf[['ExperimentID', 'Individuals']].drop_duplicates()
        if len(animals) < min_animals:
            continue
        animals_sample = animals.sample(n=min_animals, random_state=random_state)
        gdf_s = gdf.merge(animals_sample, on=['ExperimentID', 'Individuals'], how='inner')
        group_frames = []
        for (exp, ind), adf in gdf_s.groupby(['ExperimentID', 'Individuals']):
            n = min(frames_per_animal, len(adf))
            group_frames.append(adf.sample(n=n, random_state=random_state))
        balanced.extend(group_frames)
        total = sum(len(d) for d in group_frames)
        print(f'  {gvals}: {len(animals_sample)} animals × ~{frames_per_animal} frames = {total} total')

    return pd.concat(balanced, ignore_index=True)


# ── PCA ───────────────────────────────────────────────────────────────────────

def filter_low_coverage_animals(efd_df: pd.DataFrame,
                                 n_common_frames: int,
                                 min_good_frac: float) -> pd.DataFrame:
    min_good = int(min_good_frac * n_common_frames)
    counts   = efd_df.groupby(['ExperimentID', 'Individuals']).size()
    good     = counts[counts >= min_good].index
    print(f'  Coverage filter: need ≥{min_good} frames | '
          f'kept {len(good)}/{len(counts)} animals  '
          f'(dropped {(counts < min_good).sum()})')
    return efd_df.set_index(['ExperimentID', 'Individuals']).loc[good].reset_index()


def fit_pca_discovery(disc_df: pd.DataFrame, amp_cols: list,
                       random_state: int = 139) -> tuple:
    """Fit scaler + PCA (95% variance) on discovery set. Returns (scaler, pca, n_pcs)."""
    X      = disc_df[amp_cols].values
    scaler = StandardScaler()
    Xs     = scaler.fit_transform(X)

    var_cum = np.cumsum(PCA().fit(Xs).explained_variance_ratio_)
    n_pcs   = int(np.argmax(var_cum >= 0.95)) + 1
    print(f'  PCA variance: 85%→{int(np.argmax(var_cum >= 0.85)) + 1}  '
          f'95%→{n_pcs}  99%→{int(np.argmax(var_cum >= 0.99)) + 1} components')

    pca = PCA(n_components=n_pcs, random_state=random_state)
    pca.fit(Xs)
    print(f'  Using {n_pcs} components  (total var explained: {pca.explained_variance_ratio_.sum():.3f})')
    return scaler, pca, n_pcs


def project_pca(df: pd.DataFrame, amp_cols: list,
                 scaler, pca, n_pcs: int,
                 meta_cols: list) -> pd.DataFrame:
    Xs  = scaler.transform(df[amp_cols].values)
    Xp  = pca.transform(Xs)
    out = df[[c for c in meta_cols + ['Timepoint'] if c in df.columns]].copy()
    for i in range(n_pcs):
        out[f'PC{i + 1}'] = Xp[:, i]
    return out


def filter_pca_outliers(pca_df: pd.DataFrame, pc_cols: list,
                         quantile: float) -> pd.DataFrame:
    lo   = pca_df[pc_cols].quantile(quantile)
    hi   = pca_df[pc_cols].quantile(1 - quantile)
    mask = pca_df[pc_cols].ge(lo).all(axis=1) & pca_df[pc_cols].le(hi).all(axis=1)
    print(f'  Outlier filter (q={quantile}): kept {mask.sum()}/{len(pca_df)} frames '
          f'({mask.sum() / len(pca_df) * 100:.1f}%)')
    return pca_df[mask].copy()


# ── Alpha-centering ───────────────────────────────────────────────────────────

def compute_animal_medians(df: pd.DataFrame, pc_cols: list,
                            id_cols: tuple = ('ExperimentID', 'Individuals')) -> pd.DataFrame:
    meds = (df.groupby(list(id_cols), sort=False)[pc_cols]
              .median()
              .reset_index()
              .rename(columns={c: f'{c}_med' for c in pc_cols}))
    return meds


def apply_alpha_centering(df: pd.DataFrame, pc_cols: list,
                           animal_medians: pd.DataFrame, alpha: float,
                           id_cols: tuple = ('ExperimentID', 'Individuals'),
                           out_prefix: str = 'NPC') -> pd.DataFrame:
    out = df.merge(animal_medians, on=list(id_cols), how='left')
    for j, c in enumerate(pc_cols, start=1):
        out[f'{out_prefix}{j}'] = out[c] - alpha * out[f'{c}_med']
    out.drop(columns=[f'{c}_med' for c in pc_cols], inplace=True, errors='ignore')
    return out


# ── Plots ─────────────────────────────────────────────────────────────────────

def plot_animal_distribution(unique_animals: pd.DataFrame, out_dir: str):
    fig, ax = plt.subplots(figsize=(12, 6))
    unique_animals['Condition'].value_counts().plot(kind='bar', ax=ax)
    ax.set_xlabel('Condition'); ax.set_ylabel('Number of Animals')
    ax.set_title('Animal Distribution Across Conditions')
    plt.xticks(rotation=45, ha='right'); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '01_animal_distribution.png'), dpi=300)
    plt.close()
    print('  ✓ Saved: 01_animal_distribution.png')


def plot_frames_per_animal(efd_df: pd.DataFrame, out_dir: str,
                            quality_filtered: bool):
    fpa = efd_df.groupby(['ExperimentID', 'Individuals']).size()
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(fpa, bins=50, edgecolor='black')
    ax.axvline(fpa.median(), color='red', linestyle='--',
               label=f'Median: {fpa.median():.0f}')
    title = 'Valid Frames per Animal'
    if quality_filtered:
        title += ' (quality-filtered)'
    ax.set_xlabel('Frames per Animal'); ax.set_ylabel('Count')
    ax.set_title(title); ax.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '02_frames_per_animal.png'), dpi=300)
    plt.close()
    print('  ✓ Saved: 02_frames_per_animal.png')


def plot_power_spectrum(amp_array: np.ndarray, efd_order: int,
                         n_harmonics_use: int, out_dir: str):
    mean_amp = amp_array.mean(axis=0)
    std_amp  = amp_array.std(axis=0)
    cum_norm = np.cumsum(mean_amp) / np.cumsum(mean_amp)[-1]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.errorbar(np.arange(1, efd_order + 1), mean_amp, yerr=std_amp,
                 fmt='o-', capsize=3, alpha=0.7)
    ax1.axvline(1.5, color='red', linestyle='--', label='Skip harmonic 1')
    ax1.axvline(n_harmonics_use + 1.5, color='orange', linestyle='--',
                label=f'Use up to harmonic {n_harmonics_use + 1}')
    ax1.set_xlabel('Harmonic Number'); ax1.set_ylabel('Mean Amplitude')
    ax1.set_title('Shape Power Spectrum'); ax1.legend(); ax1.grid(True, alpha=0.3)

    ax2.plot(np.arange(1, efd_order + 1), cum_norm * 100, 'o-')
    ax2.axhline(95, color='red', linestyle='--', label='95%')
    ax2.axvline(n_harmonics_use + 1.5, color='orange', linestyle='--',
                label=f'Harmonic {n_harmonics_use + 1}')
    ax2.set_xlabel('Harmonic Number'); ax2.set_ylabel('Cumulative Power (%)')
    ax2.set_title('Cumulative Shape Power'); ax2.legend(); ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '03_power_spectrum.png'), dpi=300)
    plt.close()
    print(f'  ✓ Saved: 03_power_spectrum.png  '
          f'(power 2–{n_harmonics_use + 1}: {cum_norm[n_harmonics_use] * 100:.1f}%)')


def plot_discovery_distribution(disc_df: pd.DataFrame, group_cols: list,
                                 frames_per_animal: int, out_dir: str,
                                 balance_mode: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    if len(group_cols) == 1:
        disc_df['Condition'].value_counts().plot(kind='bar', ax=axes[0])
        axes[0].set_xlabel('Condition')
    else:
        disc_df.groupby(group_cols).size().plot(kind='bar', ax=axes[0])
        axes[0].set_xlabel('Condition + Specific')
    axes[0].set_ylabel('Frames')
    axes[0].set_title(f'Discovery Set Frame Distribution (mode={balance_mode})')
    plt.setp(axes[0].xaxis.get_majorticklabels(), rotation=45, ha='right')

    fpa = disc_df.groupby(['ExperimentID', 'Individuals']).size()
    axes[1].hist(fpa, bins=30, edgecolor='black')
    axes[1].axvline(frames_per_animal, color='red', linestyle='--',
                    label=f'Target: {frames_per_animal}')
    axes[1].set_xlabel('Frames per Animal'); axes[1].set_ylabel('Count')
    axes[1].set_title('Discovery Set: Frames per Animal'); axes[1].legend()

    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '04_discovery_set_distribution.png'), dpi=300)
    plt.close()
    print('  ✓ Saved: 04_discovery_set_distribution.png')


def plot_alpha_scan(pca_df: pd.DataFrame, pc_cols: list,
                    animal_medians: pd.DataFrame, alphas: list,
                    out_dir: str):
    fig, axes = plt.subplots(1, len(alphas), figsize=(5 * len(alphas), 4))
    for ax, a in zip(axes, alphas):
        tmp = apply_alpha_centering(pca_df, pc_cols, animal_medians, alpha=a, out_prefix='NPC')
        sub = tmp.sample(n=min(40000, len(tmp)), random_state=139)
        for cond in sorted(sub['Condition'].unique()):
            s = sub[sub['Condition'] == cond]
            ax.scatter(s['NPC1'], s['NPC2'], s=3, alpha=0.25, label=cond)
        ax.set_title(f'alpha={a}'); ax.set_xlabel('NPC1'); ax.set_ylabel('NPC2')
        ax.grid(True, alpha=0.3)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', markerscale=3)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'alpha_centering_scan.png'), dpi=300)
    plt.close()
    print('  ✓ Saved: alpha_centering_scan.png')


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Paths ────────────────────────────────────────────────────────────────
    X_PATH        = '/work/oleg/Adult_data_new/Data/Ciona_adult_plotting/df_x_classified.csv'   # ← edit
    Y_PATH        = '/work/oleg/Adult_data_new/Data/Ciona_adult_plotting/df_y_classified.csv'   # ← edit
    QUALITY_PATH  = '/work/oleg/Adult_data_new/Data/frame_quality/good_frames_df.csv'           # ← edit or None
    OUT_DIR       = '/work/oleg/Adult_data_new/Data/posture_clustering1'                        # ← edit

    # ── Parameters ───────────────────────────────────────────────────────────
    EFD_ORDER             = 30
    N_HARMONICS_USE       = 10
    TIME_LIMIT            = 3600       # max frames to use
    MIN_GOOD_FRAC         = 0.7
    OUTLIER_QUANTILE      = 0.005
    N_PCS_FOR_OUTLIER     = 8
    DISCOVERY_BALANCE_MODE = 'per_condition'   # 'per_condition' or 'per_subgroup'
    MIN_ANIMALS           = 20
    FRAMES_PER_ANIMAL     = 1000
    ALPHA                 = 0.2
    ALPHA_SCAN            = [0.0, 0.2, 0.35, 0.5, 1.0]
    RANDOM_STATE          = 139
    N_JOBS                = -1

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Load data ─────────────────────────────────────────────────────────────
    print('Loading coordinate data…')
    df_x = pd.read_csv(X_PATH)
    df_y = pd.read_csv(Y_PATH)
    print(f'  df_x: {df_x.shape}   df_y: {df_y.shape}')

    use_quality    = False
    quality_lookup = None
    if QUALITY_PATH and os.path.exists(QUALITY_PATH):
        good_frames_df = pd.read_csv(QUALITY_PATH)
        print(f'  Quality data: {len(good_frames_df)} animals  '
              f'(mean quality: {good_frames_df["fraction_good"].mean() * 100:.1f}%)')
        use_quality = True
    else:
        print('  No quality data found – proceeding without frame filtering')

    has_specific = 'Specific' in df_x.columns
    print(f'  Has Specific column: {has_specific}')

    meta_cols = META_BASE + (['Specific'] if has_specific else [])

    # ── Step 1: Common frames ─────────────────────────────────────────────────
    print(f'\nStep 1: Identifying common frames (limit={TIME_LIMIT})…')
    x_cols = frame_cols_n(df_x, TIME_LIMIT)
    y_cols = frame_cols_n(df_y, TIME_LIMIT)
    common_frames = [c for c in x_cols if c in set(y_cols)]
    print(f'  Common frames: {len(common_frames)}  ({common_frames[0]} … {common_frames[-1]})')

    unique_animals = df_x[meta_cols].drop_duplicates().reset_index(drop=True)
    print(f'  Total unique animals: {len(unique_animals)}')
    plot_animal_distribution(unique_animals, OUT_DIR)

    if use_quality:
        print('  Building quality lookup…')
        quality_lookup = build_quality_lookup(good_frames_df, common_frames)
        total_bad = sum(len(v) for v in quality_lookup.values())
        total_pos = len(quality_lookup) * len(common_frames)
        print(f'  Bad frames: {total_bad:,}/{total_pos:,} '
              f'({total_bad / max(total_pos, 1) * 100:.2f}%)')

    # ── Step 2: EFD extraction ────────────────────────────────────────────────
    print(f'\nStep 2: Extracting EFD coefficients (order={EFD_ORDER})…')
    efd_df = build_efd_timeseries(df_x, df_y, unique_animals, common_frames,
                                   EFD_ORDER, quality_lookup, N_JOBS)
    n_anim = efd_df[['ExperimentID', 'Individuals']].drop_duplicates().shape[0]
    print(f'  Extracted {len(efd_df):,} frames from {n_anim} animals  '
          f'(mean {len(efd_df) / max(n_anim, 1):.1f} frames/animal)')
    plot_frames_per_animal(efd_df, OUT_DIR, use_quality)

    # ── Step 3: Filter low-coverage animals ───────────────────────────────────
    print('\nStep 3: Filtering low-coverage animals…')
    efd_df = filter_low_coverage_animals(efd_df, len(common_frames), MIN_GOOD_FRAC)
    efd_df.to_parquet(os.path.join(OUT_DIR, 'efd_raw_coefficients.parquet'), index=False)
    print(f'  Filtered EFDs: {len(efd_df):,} frames')

    # ── Step 4: Harmonic amplitudes ───────────────────────────────────────────
    print('\nStep 4: Computing harmonic amplitudes…')
    amp_df, amp_array, amp_array_posture = compute_harmonic_amplitudes(
        efd_df, EFD_ORDER, N_HARMONICS_USE, meta_cols)
    amp_cols = [f'amp_{i + 2}' for i in range(N_HARMONICS_USE)]
    print(f'  Using harmonics 2–{N_HARMONICS_USE + 1}  |  amplitude df: {amp_df.shape}')
    plot_power_spectrum(amp_array, EFD_ORDER, N_HARMONICS_USE, OUT_DIR)
    amp_df.to_parquet(os.path.join(OUT_DIR, 'efd_amplitudes.parquet'), index=False)

    # ── Step 5: Balanced discovery set ───────────────────────────────────────
    print(f'\nStep 5: Building balanced discovery set (mode={DISCOVERY_BALANCE_MODE})…')
    disc_df = build_balanced_discovery(
        amp_df, has_specific, DISCOVERY_BALANCE_MODE,
        MIN_ANIMALS, FRAMES_PER_ANIMAL, random_state=42)
    print(f'\n  Discovery: {len(disc_df):,} frames  |  '
          f'{disc_df[["ExperimentID", "Individuals"]].drop_duplicates().shape[0]} animals')
    group_cols = (['Condition', 'Specific'] if DISCOVERY_BALANCE_MODE == 'per_subgroup' and has_specific
                  else ['Condition'])
    plot_discovery_distribution(disc_df, group_cols, FRAMES_PER_ANIMAL, OUT_DIR, DISCOVERY_BALANCE_MODE)
    disc_df.to_parquet(os.path.join(OUT_DIR, 'discovery_set.parquet'), index=False)

    # ── Step 6: PCA on discovery set ─────────────────────────────────────────
    print('\nStep 6: PCA on discovery set…')
    scaler, pca, n_pcs = fit_pca_discovery(disc_df, amp_cols, RANDOM_STATE)

    # Add PC scores to discovery df and filter outliers
    X_disc = pca.transform(scaler.transform(disc_df[amp_cols].values))
    for i in range(n_pcs):
        disc_df[f'PC{i + 1}'] = X_disc[:, i]

    pc_cols = [f'PC{i + 1}' for i in range(n_pcs)]
    filt_pc_cols = pc_cols[:min(N_PCS_FOR_OUTLIER, n_pcs)]
    disc_filt_df = filter_pca_outliers(disc_df, filt_pc_cols, OUTLIER_QUANTILE)
    disc_filt_df.to_parquet(os.path.join(OUT_DIR, 'pca_discovery_filtered.parquet'), index=False)
    print(f'  ✓ Saved: pca_discovery_filtered.parquet')

    # ── Step 6b: Project all good frames ─────────────────────────────────────
    print('\nStep 6b: Projecting all good frames into PCA space…')
    pca_all_df = project_pca(amp_df, amp_cols, scaler, pca, n_pcs, meta_cols)
    pca_all_df.to_parquet(os.path.join(OUT_DIR, 'pca_all_good_frames.parquet'), index=False)
    print(f'  ✓ Saved: pca_all_good_frames.parquet  (n={len(pca_all_df):,})')

    # ── Step 6c: Alpha-centering ──────────────────────────────────────────────
    print(f'\nStep 6c: Alpha-centering (alpha={ALPHA})…')
    animal_medians = compute_animal_medians(pca_all_df, pc_cols)   # medians from full dataset

    pca_all_norm  = apply_alpha_centering(pca_all_df,   pc_cols, animal_medians, ALPHA, out_prefix='NPC')
    pca_disc_norm = apply_alpha_centering(disc_filt_df, pc_cols, animal_medians, ALPHA, out_prefix='NPC')

    pca_all_norm.to_parquet( os.path.join(OUT_DIR, f'pca_all_good_frames_ALPHA{ALPHA:.2f}.parquet'),    index=False)
    pca_disc_norm.to_parquet(os.path.join(OUT_DIR, f'pca_discovery_filtered_ALPHA{ALPHA:.2f}.parquet'), index=False)
    print(f'  ✓ Saved alpha-normalised files (ALPHA={ALPHA})')

    plot_alpha_scan(disc_filt_df, pc_cols, animal_medians, ALPHA_SCAN, OUT_DIR)

    print(f'\nDone. Outputs → {OUT_DIR}')


if __name__ == '__main__':
    main()

