#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Embedding and Clustering
========================
Balanced discovery set → UMAP → cluster search (HDBSCAN + GMM in feature
and UMAP spaces) → GMM fixed-K in feature space → KNN label propagation
to the full dataset.

Steps:
    A) HDBSCAN in feature space
    B) GMM model selection (BIC/AIC) in feature space
    C) UMAP embedding (reproducible: spectral init, fixed n_epochs)
    D) Cluster search in UMAP space (HDBSCAN + GMM, K constrained 5–17)
    E) GMM fixed-K in feature space + UMAP visualisation
    F) KNN label propagation → full dataset summary
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import umap
import hdbscan
from datetime import datetime
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score
import warnings

warnings.filterwarnings('ignore')

RANDOM_STATE = 139


# ── Helpers ───────────────────────────────────────────────────────────────────

def ensure_dir(p: str) -> str:
    os.makedirs(p, exist_ok=True)
    return p


def count_clusters(labels: np.ndarray) -> tuple[int, float]:
    unique     = set(labels)
    n_clusters = len(unique) - (1 if -1 in unique else 0)
    noise_frac = float((labels == -1).mean()) if -1 in unique else 0.0
    return n_clusters, noise_frac


def save_scatter_by_condition(df: pd.DataFrame, xcol: str, ycol: str,
                               outpath: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    for cond in sorted(df['Condition'].unique()):
        s = df[df['Condition'] == cond]
        ax.scatter(s[xcol], s[ycol], s=4, alpha=0.25, label=cond)
    ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(markerscale=3, fontsize=8)
    plt.tight_layout(); plt.savefig(outpath, dpi=250); plt.close()


def save_scatter_cluster(df: pd.DataFrame, xcol: str, ycol: str, ccol: str,
                          outpath: str, title: str):
    fig, ax = plt.subplots(figsize=(8, 6))
    if (df[ccol] == -1).any():
        s = df[df[ccol] == -1]
        ax.scatter(s[xcol], s[ycol], s=4, alpha=0.20, c='lightgray', label='noise')
    clusts = sorted([c for c in df[ccol].unique() if c != -1])
    for c in clusts:
        s = df[df[ccol] == c]
        ax.scatter(s[xcol], s[ycol], s=4, alpha=0.35,
                   label=f'C{c}' if len(clusts) <= 15 else None)
    ax.set_xlabel(xcol); ax.set_ylabel(ycol); ax.set_title(title)
    ax.grid(True, alpha=0.3)
    if len(clusts) <= 15:
        ax.legend(markerscale=3, fontsize=8, bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(outpath, dpi=250, bbox_inches='tight')
    plt.close()


# ── A) HDBSCAN in feature space ───────────────────────────────────────────────

def hdbscan_feature_space(X: np.ndarray, df: pd.DataFrame,
                           configs: list, target_range: tuple,
                           out_dir: str) -> pd.DataFrame:
    results = []
    for cfg in configs:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=cfg['min_cluster_size'],
                                     min_samples=cfg['min_samples'],
                                     metric='euclidean',
                                     cluster_selection_method='eom')
        labels = clusterer.fit_predict(X)
        C, N   = count_clusters(labels)
        ok     = target_range[0] <= C <= target_range[1]
        print(f"{'✓' if ok else ' '} PCA-HDBSCAN {cfg['name']:14s} | C:{C:2d} | N:{N:.1%}")
        results.append({'method': 'HDBSCAN_feat', 'config': cfg['name'],
                         'n_clusters': C, 'noise': N})
        if ok:
            tmp = df.copy(); tmp['cluster'] = labels
            tmp.to_parquet(os.path.join(out_dir, f"HDBSCAN_FEAT_{cfg['name']}.parquet"), index=False)
    pd.DataFrame(results).to_csv(os.path.join(out_dir, 'A_hdbscan_feature_results.csv'), index=False)
    return pd.DataFrame(results)


# ── B) GMM model selection in feature space ───────────────────────────────────

def gmm_feature_space(X: np.ndarray, df: pd.DataFrame,
                       k_range: list, out_dir: str,
                       feature_mode: str) -> tuple[pd.DataFrame, int]:
    bic, aic = [], []
    for k in k_range:
        g = GaussianMixture(n_components=k, covariance_type='full',
                             random_state=RANDOM_STATE, reg_covar=1e-6, max_iter=300)
        g.fit(X)
        bic.append(g.bic(X)); aic.append(g.aic(X))

    best_k = k_range[int(np.argmin(bic))]
    print(f'  Best K (BIC): {best_k}  |  Best K (AIC): {k_range[int(np.argmin(aic))]}')

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, bic, marker='o', label='BIC')
    plt.plot(k_range, aic, marker='o', label='AIC')
    plt.xlabel('K'); plt.ylabel('Score (lower=better)')
    plt.title(f'GMM model selection ({feature_mode} space)')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, '02_gmm_bic_aic.png'), dpi=250)
    plt.close()

    gmm = GaussianMixture(n_components=best_k, covariance_type='full',
                           random_state=RANDOM_STATE, reg_covar=1e-6, max_iter=500)
    labels = gmm.fit_predict(X).astype(int)
    df_out = df.copy(); df_out['cluster_gmm_feat'] = labels
    df_out.to_parquet(os.path.join(out_dir, f'GMM_FEAT_K{best_k}.parquet'), index=False)
    print(f'  ✓ Saved: GMM_FEAT_K{best_k}.parquet')
    return df_out, best_k


# ── C) UMAP (reproducible) ────────────────────────────────────────────────────

def fit_umap(X: np.ndarray, n_neighbors: int, min_dist: float) -> np.ndarray:
    reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist,
                         n_components=2, metric='euclidean',
                         random_state=RANDOM_STATE,
                         low_memory=True, init='spectral', n_epochs=500)
    return reducer.fit_transform(X).astype(np.float32)


# ── D) Cluster search in UMAP space ──────────────────────────────────────────

def hdbscan_umap_space(X_umap: np.ndarray, df: pd.DataFrame,
                        configs: list, target_range: tuple,
                        uc_name: str, out_dir: str) -> pd.DataFrame:
    results = []
    for cfg in configs:
        clusterer = hdbscan.HDBSCAN(min_cluster_size=cfg['min_cluster_size'],
                                     min_samples=cfg['min_samples'],
                                     metric='euclidean',
                                     cluster_selection_method='eom')
        labels = clusterer.fit_predict(X_umap)
        C, N   = count_clusters(labels)
        ok     = target_range[0] <= C <= target_range[1]
        print(f"{'✓' if ok else ' '} UMAP-HDBSCAN {cfg['name']:14s} | C:{C:2d} | N:{N:.1%}")
        results.append({'method': 'HDBSCAN_umap', 'config': cfg['name'],
                         'n_clusters': C, 'noise': N, 'in_target': ok})
        if ok:
            tmp = df.copy(); tmp['cluster_umap_hdbscan'] = labels
            pq  = os.path.join(out_dir, f"UMAP_{uc_name}_HDBSCAN_{cfg['name']}.parquet")
            tmp.to_parquet(pq, index=False)
            save_scatter_cluster(tmp, 'umap1', 'umap2', 'cluster_umap_hdbscan',
                                  pq.replace('.parquet', '.png'),
                                  f"UMAP {uc_name} HDBSCAN {cfg['name']} | C={C} N={N:.1%}")
    pd.DataFrame(results).to_csv(
        os.path.join(out_dir, f'UMAP_{uc_name}_hdbscan_sweep.csv'), index=False)
    print(f'  ✓ Saved sweep: UMAP_{uc_name}_hdbscan_sweep.csv')
    return pd.DataFrame(results)


def gmm_umap_space(X_umap: np.ndarray, df: pd.DataFrame,
                    k_range: list, uc_name: str, out_dir: str) -> tuple[pd.DataFrame, int]:
    bic, aic = [], []
    for k in k_range:
        g = GaussianMixture(n_components=k, covariance_type='full',
                             random_state=RANDOM_STATE, reg_covar=1e-6,
                             max_iter=400, n_init=5, init_params='kmeans')
        g.fit(X_umap)
        bic.append(g.bic(X_umap)); aic.append(g.aic(X_umap))

    best_k = k_range[int(np.argmin(bic))]
    print(f'  Best K in UMAP space (BIC, restricted {k_range[0]}–{k_range[-1]}): {best_k}')

    plt.figure(figsize=(8, 5))
    plt.plot(k_range, bic, marker='o', label='BIC')
    plt.plot(k_range, aic, marker='o', label='AIC')
    plt.xlabel('K'); plt.ylabel('Score (lower=better)')
    plt.title(f'GMM selection in UMAP space ({uc_name}) restricted to {k_range[0]}–{k_range[-1]}')
    plt.grid(True, alpha=0.3); plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'UMAP_{uc_name}_GMM_bic_aic_5to17.png'), dpi=250)
    plt.close()

    gmm = GaussianMixture(n_components=best_k, covariance_type='full',
                           random_state=RANDOM_STATE, reg_covar=1e-6,
                           max_iter=600, n_init=10, init_params='kmeans')
    labels = gmm.fit_predict(X_umap).astype(int)
    df_out = df.copy(); df_out['cluster_umap_gmm'] = labels
    df_out.to_parquet(os.path.join(out_dir, f'UMAP_{uc_name}_GMM_K{best_k}.parquet'), index=False)
    save_scatter_cluster(df_out, 'umap1', 'umap2', 'cluster_umap_gmm',
                          os.path.join(out_dir, f'UMAP_{uc_name}_GMM_K{best_k}.png'),
                          f'UMAP {uc_name} + GMM K={best_k} (restricted {k_range[0]}–{k_range[-1]})')
    return df_out, best_k


# ── E) GMM fixed-K in feature space ──────────────────────────────────────────

def gmm_feature_fixed_k(X: np.ndarray, df: pd.DataFrame,
                          k: int, feature_mode: str, n_pcs: int,
                          out_dir: str) -> pd.DataFrame:
    gmm = GaussianMixture(n_components=k, covariance_type='full',
                           random_state=RANDOM_STATE, reg_covar=1e-6,
                           max_iter=600, n_init=10, init_params='kmeans')
    labels = gmm.fit_predict(X).astype(int)
    df_out = df.copy(); df_out['cluster'] = labels
    name   = f'GMM_FEAT_ONLY_{feature_mode}{n_pcs}_K{k}'
    df_out.to_parquet(os.path.join(out_dir, f'{name}.parquet'), index=False)
    print(f'  ✓ Saved: {name}.parquet')
    return df_out


# ── F) KNN propagation ────────────────────────────────────────────────────────

def knn_propagate(discovery_df: pd.DataFrame, full_df: pd.DataFrame,
                   feat_cols: list, label_col: str,
                   k_expected: int, n_neighbors: int, weights: str,
                   out_dir: str) -> pd.DataFrame:
    # Build unique frame IDs
    def _frame_id(df_):
        return (df_['ExperimentID'].astype(str) + '_' +
                df_['Individuals'].astype(str) + '_' +
                df_['Timepoint'].astype(str))

    discovery_df = discovery_df.copy(); discovery_df['frame_id'] = _frame_id(discovery_df)
    full_df      = full_df.copy();      full_df['frame_id']      = _frame_id(full_df)

    disc_ids               = set(discovery_df['frame_id'])
    full_df['in_discovery'] = full_df['frame_id'].isin(disc_ids)

    print(f'  Labeled (discovery): {full_df["in_discovery"].sum():,}  |  '
          f'to propagate: {(~full_df["in_discovery"]).sum():,}')
    remaining_df = full_df[~full_df['in_discovery']].copy()

    print('\n  Remaining frames by condition:')
    for cond in sorted(remaining_df['Condition'].unique()):
        print(f'    {cond}: {(remaining_df["Condition"] == cond).sum():,}')

    # Feature matrices
    y_unique = sorted(pd.unique(discovery_df[label_col]))
    expected = set(range(k_expected))
    observed = set(int(x) for x in y_unique)
    if observed != expected:
        print(f'  WARNING: observed cluster IDs {sorted(observed)} != expected 0..{k_expected-1}')

    X_train   = discovery_df[feat_cols].to_numpy(dtype=np.float32)
    y_train   = discovery_df[label_col].to_numpy(dtype=int)
    X_predict = remaining_df[feat_cols].to_numpy(dtype=np.float32)
    print(f'  Training: {len(X_train):,}  |  Predicting: {len(X_predict):,}  |  Features: {feat_cols}')

    # Fit + predict
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, weights=weights,
                                metric='euclidean', n_jobs=-1)
    knn.fit(X_train, y_train)

    y_pred     = knn.predict(X_predict)
    confidence = knn.predict_proba(X_predict).max(axis=1)

    remaining_df['cluster']            = y_pred.astype(int)
    remaining_df['cluster_confidence'] = confidence.astype(np.float32)

    print(f'\n  Predicted {len(y_pred):,} labels')
    print(f'  Confidence  mean={confidence.mean():.3f}  '
          f'median={np.median(confidence):.3f}  min={confidence.min():.3f}')
    low_conf = confidence < 0.5
    print(f'  Low-confidence (<0.5): {low_conf.sum():,} ({low_conf.mean() * 100:.1f}%)')

    # 5-fold CV on discovery
    cv_scores = cross_val_score(knn, X_train, y_train, cv=5, n_jobs=-1)
    print(f'  5-fold CV accuracy: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}')

    # Predicted distribution
    print('\n  Predicted cluster distribution:')
    for cid, cnt in pd.Series(y_pred).value_counts().sort_index().items():
        print(f'    C{int(cid):02d}: {cnt:,} ({100 * cnt / len(y_pred):.1f}%)')

    # Combine
    disc_final                   = discovery_df.copy()
    disc_final['cluster_confidence'] = 1.0
    disc_final['label_source']   = 'discovery'
    remaining_df['label_source'] = 'propagated'

    full_clustered = pd.concat([disc_final, remaining_df], ignore_index=True)
    full_clustered.drop(columns=['frame_id', 'in_discovery'], errors='ignore', inplace=True)

    # ── Summary tables ────────────────────────────────────────────────────────
    full_clustered['animal_id'] = (full_clustered['ExperimentID'].astype(str) + '_' +
                                    full_clustered['Individuals'].astype(str))
    animals_per = (full_clustered.groupby(['cluster', 'Condition'])['animal_id']
                   .nunique().reset_index(name='n_animals'))
    pivot = (animals_per
             .pivot_table(index='cluster', columns='Condition',
                           values='n_animals', fill_value=0, aggfunc='sum')
             .reset_index())

    tot_animals = full_clustered.groupby('cluster')['animal_id'].nunique()
    tot_frames  = full_clustered['cluster'].value_counts()
    pivot['n_animals_total'] = pivot['cluster'].map(tot_animals).astype(int)
    pivot['n_frames_total']  = pivot['cluster'].map(tot_frames).astype(int)
    pivot = pivot.set_index('cluster').reindex(range(k_expected)).fillna(0).reset_index()

    pivot.to_csv(os.path.join(out_dir, 'SUMMARY_animals_per_cluster_per_condition.csv'), index=False)

    cond_cols = [c for c in pivot.columns if c not in ['cluster', 'n_animals_total', 'n_frames_total']]
    prop      = pivot.copy()
    for c in cond_cols:
        prop[c] = (prop[c] / prop['n_animals_total'].replace(0, np.nan)).fillna(0)
    prop.to_csv(os.path.join(out_dir, 'SUMMARY_animals_proportions.csv'), index=False)
    print('\n', pivot)

    # ── Plots ─────────────────────────────────────────────────────────────────
    plt.figure(figsize=(8, 5))
    plt.hist(remaining_df['cluster_confidence'], bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('kNN prediction confidence'); plt.ylabel('count')
    plt.title('kNN confidence (propagated labels)')
    plt.grid(True, alpha=0.3, axis='y'); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'knn_confidence_hist.png'), dpi=300)
    plt.close()

    hm = prop.set_index('cluster')[cond_cols]
    plt.figure(figsize=(10, 6))
    sns.heatmap(hm, annot=True, fmt='.2f', cmap='RdYlBu_r',
                cbar_kws={'label': 'Proportion of animals'})
    plt.xlabel('Condition'); plt.ylabel('Cluster')
    plt.title('Animal composition per cluster (proportions)'); plt.tight_layout()
    plt.savefig(os.path.join(out_dir, 'animals_condition_heatmap.png'), dpi=300)
    plt.close()

    out_path = os.path.join(out_dir, 'full_dataset_clustered_knn.parquet')
    full_clustered.to_parquet(out_path, index=False)
    print(f'\n  ✓ Saved full clustered dataset: {out_path}  ({len(full_clustered):,} frames)')
    return full_clustered


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    # ── Paths ────────────────────────────────────────────────────────────────
    BASE_DIR   = '/work/oleg/Adult_data_new/Data/posture_clustering1'                               # ← edit
    DISC_PATH  = os.path.join(BASE_DIR, 'pca_discovery_filtered_ALPHA0.20.parquet')                 # ← edit
    FULL_PATH  = os.path.join(BASE_DIR, 'pca_all_good_frames_ALPHA0.20.parquet')                    # ← edit

    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    OUT_DIR   = os.path.join(BASE_DIR, f'clustering_{timestamp}')
    ensure_dir(OUT_DIR)

    # ── Parameters ───────────────────────────────────────────────────────────
    FEATURE_MODE     = 'NPC'          # 'NPC' or 'PC'
    N_PCS_USE        = 5
    TARGET_CLUSTERS  = (5, 17)
    GMM_K_RANGE_FEAT = list(range(3, 26))
    GMM_K_RANGE_UMAP = list(range(5, 18))
    GMM_K_FIXED      = 16             # for fixed-K GMM + KNN propagation
    UMAP_N_NEIGHBORS = 50
    UMAP_MIN_DIST    = 0.7
    KNN_N_NEIGHBORS  = 50
    KNN_WEIGHTS      = 'distance'

    feat_cols = [f'{FEATURE_MODE}{i}' for i in range(1, N_PCS_USE + 1)]
    uc_name   = f'n{UMAP_N_NEIGHBORS}_d{UMAP_MIN_DIST}'

    # ── Load ─────────────────────────────────────────────────────────────────
    print('Loading data…')
    pca_df  = pd.read_parquet(DISC_PATH)
    full_df = pd.read_parquet(FULL_PATH)
    print(f'  Discovery: {len(pca_df):,}  |  Full: {len(full_df):,}')
    print(f'  Conditions: {sorted(pca_df["Condition"].unique())}')

    # Canonical row order → reproducible UMAP
    sort_cols = [c for c in ['ExperimentID', 'Individuals', 'Timepoint'] if c in pca_df.columns]
    pca_df  = pca_df.sort_values(sort_cols,  kind='mergesort').reset_index(drop=True)
    full_df = full_df.sort_values(sort_cols, kind='mergesort').reset_index(drop=True)

    # Build feature matrix (float32, drop non-finite)
    X  = pca_df[feat_cols].to_numpy(dtype=np.float32)
    ok = np.isfinite(X).all(axis=1)
    pca_df = pca_df.loc[ok].reset_index(drop=True); X = X[ok]

    # Sanity scatter
    save_scatter_by_condition(
        pca_df, feat_cols[0], feat_cols[1],
        os.path.join(OUT_DIR, f'01_{FEATURE_MODE}_space_scatter.png'),
        f'{FEATURE_MODE}-space: {feat_cols[0]} vs {feat_cols[1]}')

    # ── A) HDBSCAN in feature space ───────────────────────────────────────────
    print('\n' + '='*60)
    print('A) HDBSCAN IN FEATURE SPACE')
    print('='*60)
    hdb_feat_configs = [
        {'min_cluster_size': 500,  'min_samples': 10, 'name': 'mcs500_ms10'},
        {'min_cluster_size': 700,  'min_samples': 8,  'name': 'mcs700_ms8'},
        {'min_cluster_size': 900,  'min_samples': 10, 'name': 'mcs900_ms10'},
        {'min_cluster_size': 1200, 'min_samples': 15, 'name': 'mcs1200_ms15'},
    ]
    hdbscan_feature_space(X, pca_df, hdb_feat_configs, TARGET_CLUSTERS, OUT_DIR)

    # ── B) GMM model selection in feature space ────────────────────────────────
    print('\n' + '='*60)
    print('B) GMM MODEL SELECTION (BIC/AIC) IN FEATURE SPACE')
    print('='*60)
    pca_gmm, best_k_feat = gmm_feature_space(X, pca_df, GMM_K_RANGE_FEAT, OUT_DIR, FEATURE_MODE)

    # ── C) UMAP ───────────────────────────────────────────────────────────────
    print('\n' + '='*60)
    print(f'C) UMAP ({uc_name})')
    print('='*60)
    emb = fit_umap(X, UMAP_N_NEIGHBORS, UMAP_MIN_DIST)
    pca_gmm['umap1'] = emb[:, 0]
    pca_gmm['umap2'] = emb[:, 1]

    embed_path = os.path.join(OUT_DIR, f'UMAP_{uc_name}_embedding.parquet')
    pca_gmm.to_parquet(embed_path, index=False)
    print(f'  ✓ Saved embedding: {embed_path}')

    save_scatter_by_condition(
        pca_gmm, 'umap1', 'umap2',
        os.path.join(OUT_DIR, f'03_umap_{uc_name}_GMMlabels.png'),
        f'UMAP {uc_name} – by condition')

    # ── D) Cluster search in UMAP space ───────────────────────────────────────
    print('\n' + '='*60)
    print('D) CLUSTER SEARCH IN UMAP SPACE (target 5–17)')
    print('='*60)
    X_umap = pca_gmm[['umap1', 'umap2']].to_numpy(dtype=np.float32)

    hdb_umap_configs = [
        {'min_cluster_size': 300,  'min_samples': 10, 'name': 'mcs300_ms10'},
        {'min_cluster_size': 500,  'min_samples': 10, 'name': 'mcs500_ms10'},
        {'min_cluster_size': 700,  'min_samples': 10, 'name': 'mcs700_ms10'},
        {'min_cluster_size': 900,  'min_samples': 10, 'name': 'mcs900_ms10'},
        {'min_cluster_size': 1200, 'min_samples': 15, 'name': 'mcs1200_ms15'},
        {'min_cluster_size': 1500, 'min_samples': 15, 'name': 'mcs1500_ms15'},
        {'min_cluster_size': 1800, 'min_samples': 20, 'name': 'mcs1800_ms20'},
        {'min_cluster_size': 2200, 'min_samples': 25, 'name': 'mcs2200_ms25'},
    ]
    hdbscan_umap_space(X_umap, pca_gmm, hdb_umap_configs, TARGET_CLUSTERS, uc_name, OUT_DIR)
    _, best_k_umap = gmm_umap_space(X_umap, pca_gmm, GMM_K_RANGE_UMAP, uc_name, OUT_DIR)

    # ── E) GMM fixed-K in feature space → UMAP visualisation ─────────────────
    print('\n' + '='*60)
    print(f'E) GMM FIXED K={GMM_K_FIXED} IN FEATURE SPACE + UMAP VIZ')
    print('='*60)
    disc_final = gmm_feature_fixed_k(X, pca_df, GMM_K_FIXED, FEATURE_MODE, N_PCS_USE, OUT_DIR)
    disc_final['umap1'] = emb[:, 0]
    disc_final['umap2'] = emb[:, 1]

    save_scatter_cluster(
        disc_final, 'umap1', 'umap2', 'cluster',
        os.path.join(OUT_DIR, f'UMAP_{uc_name}_GMM_FEATONLY_K{GMM_K_FIXED}.png'),
        f'UMAP {uc_name} (viz only) – GMM labels in feature space (K={GMM_K_FIXED})')

    # Also view in PCA(2) of the feature space
    X_p2 = PCA(n_components=2, random_state=RANDOM_STATE).fit_transform(X)
    disc_final['feat_pca1'] = X_p2[:, 0]
    disc_final['feat_pca2'] = X_p2[:, 1]
    save_scatter_cluster(
        disc_final, 'feat_pca1', 'feat_pca2', 'cluster',
        os.path.join(OUT_DIR, f'GMM_FEATSPACE_PCA2_K{GMM_K_FIXED}.png'),
        f'GMM clusters K={GMM_K_FIXED} in feature PCA(2)')

    disc_final.to_parquet(
        os.path.join(OUT_DIR, f'UMAP_{uc_name}_EMBEDDING_with_GMMlabels_K{GMM_K_FIXED}.parquet'),
        index=False)

    # ── F) KNN propagation to full dataset ────────────────────────────────────
    print('\n' + '='*60)
    print(f'F) KNN PROPAGATION (K={GMM_K_FIXED}) → FULL DATASET')
    print('='*60)
    knn_dir = ensure_dir(os.path.join(OUT_DIR, f'knn_propagation_K{GMM_K_FIXED}'))
    knn_propagate(disc_final, full_df.copy(), feat_cols, 'cluster',
                   GMM_K_FIXED, KNN_N_NEIGHBORS, KNN_WEIGHTS, knn_dir)

    print(f'\nDone. All outputs → {OUT_DIR}')


if __name__ == '__main__':
    main()
