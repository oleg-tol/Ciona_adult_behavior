# DeepLabCut Experimental Data Processing (Ciona adult)

This repository contains modules for processing experimental data and computing behavior/shape metrics.

## Modules

- **data_loading.py** — Loads and merges `*filtered.csv` files, removes likelihood columns, optionally resamples, and adds condition labels.
- **shape_metrics.py** — Computes polygon area, oral/atrial siphon widths & areas, and mantle area.
- **metrics_detrending.py** — Detrends metrics (seasonal decomposition), fills gaps, smooths, and z‑scores time series.
- **state_definition.py** — Detects contraction events, classifies them (full/half), finds quiescence, and builds a unified state map.
- **efd_wavelets_pca.py** — Computes per‑frame elliptic Fourier descriptors (EFD), per‑animal wavelet features, and static/dynamic PCs.
- **embedding_clustering.py** — Embeds static PCs with t‑SNE, clusters with DBSCAN, plots by state/condition, and reconstructs cluster medoid shapes.

## Requirements

- Python 3.x
- Pandas
- NumPy

*(For advanced steps you may also need: SciPy, statsmodels, scikit‑learn, shapely, tslearn, pyefd, PyWavelets, openTSNE, matplotlib, seaborn, joblib.)*

## Setup

1. Install the required packages.
2. Update file paths/arguments in the source files (or pass via CLI flags).
3. Run the desired module.

## Metric Calculation Module

The analysis functions span the core steps used in our experiments:

- **Shape metrics**: polygon areas; oral/atrial widths & areas; mantle area.
- **Detrending & normalization**: seasonal decomposition, interpolation/recovery of short gaps, smoothing, z‑scoring.
- **State metrics**: contraction peak detection and categorization, quiescence detection, unified state map, summary features.
- **Descriptors & PCs**: EFD time series, wavelet summaries, static and dynamic principal components.
- **Embedding & clustering**: t‑SNE embedding, DBSCAN clustering, cluster medoid reconstruction.

## Usage

Import the modules in your script or run them directly from the command line to produce the corresponding CSV outputs and (when applicable) figures.

# DeepLabCut Experimental Data Processing (Ciona adult)

This repository contains small, focused modules for processing experimental data and computing behavior/shape metrics in a simple, scriptable style.

## Modules

- **data_loading.py** — Loads and merges `*filtered.csv` files, removes likelihood columns, optionally resamples, and adds condition labels.
- **shape_metrics.py** — Computes polygon area, oral/atrial siphon widths & areas, and mantle area.
- **metrics_detrending.py** — Detrends metrics (seasonal decomposition), fills gaps, smooths, and z‑scores time series.
- **state_definition.py** — Detects contraction events, classifies them (full/half), finds quiescence, and builds a unified state map.
- **efd_wavelets_pca.py** — Computes per‑frame elliptic Fourier descriptors (EFD), per‑animal wavelet features, and static/dynamic PCs.
- **embedding_clustering.py** — Embeds static PCs with t‑SNE, clusters with DBSCAN, plots by state/condition, and reconstructs cluster medoid shapes.

## Requirements

- Python 3.x
- Pandas
- NumPy

*(For advanced steps you may also need: SciPy, statsmodels, scikit‑learn, shapely, tslearn, pyefd, PyWavelets, openTSNE, matplotlib, seaborn, joblib.)*

## Setup

1. Install the required packages.
2. Update file paths/arguments in the source files (or pass via CLI flags).
3. Run the desired module.

## Metric Calculation Module

The analysis functions span the core steps used in our experiments:

- **Shape metrics**: polygon areas; oral/atrial widths & areas; mantle area.
- **Detrending & normalization**: seasonal decomposition, interpolation/recovery of short gaps, smoothing, z‑scoring.
- **State metrics**: contraction peak detection and categorization, quiescence detection, unified state map, summary features.
- **Descriptors & PCs**: EFD time series, wavelet summaries, static and dynamic principal components.
- **Embedding & clustering**: t‑SNE embedding, DBSCAN clustering, cluster medoid reconstruction.

## Usage

Import the modules in your script or run them directly from the command line to produce the corresponding CSV outputs and (when applicable) figures.
