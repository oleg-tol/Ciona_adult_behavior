# DeepLabCut Behavioural Analysis — *Ciona intestinalis* Adult

Pipeline for processing DLC tracking data and computing behaviour and shape metrics.

## Modules

| File | Description |
|------|-------------|
| `Loading_data.py` | Load and merge `*filtered.csv` DLC output files, resample, add condition labels |
| `Contraction_metrics.py` | Polygon area, oral/atrial siphon widths & areas, mantle area, contraction amplitude & speed |
| `Contraction_data_preprocessing.py` | Seasonal decomposition, gap recovery, smoothing, z-scoring |
| `State_detection_from_contraction_metrics.py` | Contraction peak detection, full/half classification, quiescence detection, unified state map |
| `Shape_metrics.py` | EllipseAspectRatio, Solidity, BodyAngle |
| `Eigen_cionas_efds_pca.py` | EFD extraction, harmonic amplitudes (Eigen Ciona), balanced discovery set, PCA, alpha-centering |
| `Embedding.py` | UMAP embedding, HDBSCAN/GMM cluster search, fixed-K GMM, KNN label propagation to full dataset |
| `HMM.py` | HMM feature preparation (NPC + derivatives), sticky GaussianHMM fitting, post-hoc smoothing, dwell summaries |

## Pipeline

```
DLC *filtered.csv
  → Loading_data.py                  →  x_final.csv, y_final.csv
  → Contraction_metrics.py       →  *_processed.csv  (6 metrics)
  → Contraction_data_preprocessing.py            →  *_zscored.csv
  → State_detection_from_contraction_metrics.py           →  contraction_events.csv, unified_state_map.csv
  → Shape_metrics.py       →  *_processed.csv  (3 metrics)
  → Eigen_cionas_efds_pca.py                   →  efd_amplitudes.parquet, pca_*_ALPHA0.20.parquet
  → Embedding.py      →  UMAP embedding, cluster parquets, knn full dataset
  → HMM.py                →  decoded_states_all.parquet, dwell summaries
```

## Key parameters

| Parameter | Value | Location |
|-----------|-------|----------|
| FPS / frame skip | 20 / 5 → 0.25 s/frame | `load_dlc.py` |
| Tile size | 245 µm | `contraction_metrics.py` |
| Decomposition period | 50 frames | `detrend_zscore.py` |
| Peak detection | prominence=0.7, distance=5, width=5 | `state_detection.py` |
| EFD order / harmonics used | 30 / 2–11 | `efd_pca.py` |
| Alpha-centering | α=0.2 | `efd_pca.py` |
| HMM states / downsample | K=8 / 5× | `hmm_states.py` |

## Requirements

```
pandas numpy scipy statsmodels scikit-learn shapely
pyefd joblib umap-learn hdbscan hmmlearn
matplotlib seaborn
```

Edit the path variables in `main()` of each file before running.
