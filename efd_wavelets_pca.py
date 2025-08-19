"""
EFD, wavelets, PCA analysis from coordinates.
"""
import os
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from joblib import Parallel, delayed
from pyefd import elliptic_fourier_descriptors
import pywt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA

META = ['ExperimentID', 'Individuals', 'Condition']


def _numeric_time_map(df: pd.DataFrame) -> Dict[float, str]:
    m = {}
    for c in df.columns:
        if c in META:
            continue
        try:
            v = float(c)
            m[v] = c
        except Exception:
            continue
    return m


def _common_times(px: Dict[float, str], mpx: Dict[float, str], mpy: Dict[float, str], limit: float) -> List[float]:
    s = set(px) & set(mpx) & set(mpy)
    return sorted([t for t in s if t <= limit])


def _group_xy(df_x: pd.DataFrame, df_y: pd.DataFrame):
    gx = { (e, i): g for (e, i), g in df_x.groupby(['ExperimentID', 'Individuals']) }
    gy = { (e, i): g for (e, i), g in df_y.groupby(['ExperimentID', 'Individuals']) }
    return gx, gy


def _efd_row(coeffs: np.ndarray, exp: str, ind: str, cond: str, t: float) -> Dict[str, float]:
    d = {'ExperimentID': exp, 'Individuals': ind, 'Condition': cond, 'Timepoint': t}
    flat = coeffs.flatten()
    for k, v in enumerate(flat, 1):
        d[f'EFD_{k}'] = float(v)
    return d


def compute_efds_for_animal(exp: str, ind: str, cond: str,
                             times: List[float], map_poly: Dict[float, str], poly_row: pd.Series,
                             gx_row: pd.DataFrame, gy_row: pd.DataFrame, order: int) -> List[Dict[str, float]]:
    out = []
    for t in times:
        if pd.isna(poly_row[map_poly[t]]):
            continue
        try:
            xs = pd.to_numeric(gx_row[ t ], errors='coerce').values if t in gx_row.columns else pd.to_numeric(gx_row[str(t)], errors='coerce').values
            ys = pd.to_numeric(gy_row[ t ], errors='coerce').values if t in gy_row.columns else pd.to_numeric(gy_row[str(t)], errors='coerce').values
        except Exception:
            # fallback via original labels if numeric strings were used
            continue
        if len(xs) < 3 or len(xs) != len(ys):
            continue
        contour = np.column_stack([xs, -ys])
        try:
            coeffs = elliptic_fourier_descriptors(contour, order=order, normalize=False)
        except Exception:
            continue
        out.append(_efd_row(coeffs, exp, ind, cond, t))
    return out


def build_efd_timeseries(df_x: pd.DataFrame, df_y: pd.DataFrame, df_poly: pd.DataFrame,
                         order: int, time_limit: float, n_jobs: int = -1) -> pd.DataFrame:
    map_poly = _numeric_time_map(df_poly)
    map_x = _numeric_time_map(df_x)
    map_y = _numeric_time_map(df_y)
    common = _common_times(map_poly, map_x, map_y, time_limit)
    gx, gy = _group_xy(df_x, df_y)

    animals = df_poly[META].drop_duplicates()
    def _one(row):
        exp, ind, cond = row['ExperimentID'], row['Individuals'], row.get('Condition', np.nan)
        p = df_poly[(df_poly['ExperimentID'] == exp) & (df_poly['Individuals'] == ind)].iloc[0]
        return compute_efds_for_animal(exp, ind, cond, common, map_poly, p, gx.get((exp, ind), pd.DataFrame()), gy.get((exp, ind), pd.DataFrame()), order)

    chunks = Parallel(n_jobs=n_jobs, verbose=0)(delayed(_one)(row) for _, row in animals.iterrows())
    flat = [rec for lst in chunks for rec in lst]
    return pd.DataFrame(flat)


def amps_from_efds(df_efd: pd.DataFrame, order: int) -> pd.DataFrame:
    cols = [f'EFD_{k}' for k in range(1, 4*(order+1) + 1)]
    E = df_efd[cols].to_numpy(dtype=float)
    amps = []
    for h in range(order+1):
        block = E[:, 4*h:4*h+4]
        amps.append(np.linalg.norm(block, axis=1))
    A = np.vstack(amps).T
    out = df_efd[['ExperimentID','Individuals','Condition','Timepoint']].copy()
    for j in range(order+1):
        out[f'amp_{j+1}'] = A[:, j]
    return out


def static_pcs(amp_df: pd.DataFrame, truncate: int, n_pcs: int) -> pd.DataFrame:
    cols = [f'amp_{k}' for k in range(1, truncate+1)]
    X = amp_df[cols].to_numpy(dtype=float)
    Z = StandardScaler().fit_transform(X)
    p = PCA(n_components=n_pcs, random_state=0)
    S = p.fit_transform(Z)
    out = amp_df[['ExperimentID','Individuals','Condition','Timepoint']].copy()
    for i in range(n_pcs):
        out[f'PC{i+1}'] = S[:, i]
    return out


def dynamic_pcs(amp_df: pd.DataFrame, truncate: int, n_pcs: int) -> pd.DataFrame:
    cols = [f'amp_{k}' for k in range(1, truncate+1)]
    parts = []
    for (e, i), g in amp_df.groupby(['ExperimentID','Individuals']):
        g = g.sort_values('Timepoint')
        A = g[cols].to_numpy(dtype=float)
        dA = np.vstack([np.zeros((1, A.shape[1])), np.diff(A, axis=0)])
        Z = StandardScaler().fit_transform(np.hstack([A[:,:], dA]))
        S = PCA(n_components=n_pcs, random_state=0).fit_transform(Z)
        tmp = g[['ExperimentID','Individuals','Condition','Timepoint']].copy()
        for k in range(n_pcs):
            tmp[f'PC{k+1}'] = S[:, k]
        parts.append(tmp)
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame(columns=['ExperimentID','Individuals','Condition','Timepoint']+[f'PC{k+1}' for k in range(n_pcs)])


def wavelet_features(df_efd: pd.DataFrame, scales: np.ndarray) -> pd.DataFrame:
    efd_cols = [c for c in df_efd.columns if c.startswith('EFD_')]
    rows = []
    for (e, i, c), g in df_efd.groupby(['ExperimentID','Individuals','Condition']):
        g = g.sort_values('Timepoint')
        if len(g) < 2:
            continue
        feats = []
        t = g['Timepoint'].to_numpy(dtype=float)
        dt = np.diff(t).mean()
        for col in efd_cols:
            ts = g[col].to_numpy(dtype=float)
            coeffs, _ = pywt.cwt(ts, scales, 'morl', sampling_period=dt)
            feats.extend(np.mean(np.abs(coeffs), axis=1))
        rows.append({'ExperimentID': e, 'Individuals': i, 'Condition': c, 'Features': np.array(feats, dtype=float)})
    df = pd.DataFrame(rows)
    if df.empty:
        return df
    F = np.vstack(df['Features'].values)
    for j in range(F.shape[1]):
        df[f'wfeat_{j+1}'] = F[:, j]
    del df['Features']
    return df


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description='EFD, wavelets, and PCA (no plots).')
    p.add_argument('--x', required=True)
    p.add_argument('--y', required=True)
    p.add_argument('--poly', required=True)
    p.add_argument('--order', type=int, default=30)
    p.add_argument('--time-limit', type=float, default=3660)
    p.add_argument('--max-scale', type=int, default=50)
    p.add_argument('--truncate', type=int, default=10)
    p.add_argument('--n-static-pcs', type=int, default=6)
    p.add_argument('--n-dynamic-pcs', type=int, default=6)
    p.add_argument('--jobs', type=int, default=-1)
    p.add_argument('--out', default='./output_features')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    df_x = pd.read_csv(args.x)
    df_y = pd.read_csv(args.y)
    df_poly = pd.read_csv(args.poly)

    efd_ts = build_efd_timeseries(df_x, df_y, df_poly, order=args.order, time_limit=args.time_limit, n_jobs=args.jobs)
    efd_path = os.path.join(args.out, 'efd_timeseries.csv')
    efd_ts.to_csv(efd_path, index=False)

    amp_ts = amps_from_efds(efd_ts, order=args.order)
    amp_path = os.path.join(args.out, 'amp_timeseries.csv')
    amp_ts.to_csv(amp_path, index=False)

    stat = static_pcs(amp_ts, truncate=args.truncate, n_pcs=args.n_static_pcs)
    stat_path = os.path.join(args.out, 'static_pcs_timeseries.csv')
    stat.to_csv(stat_path, index=False)

    dyn = dynamic_pcs(amp_ts, truncate=args.truncate, n_pcs=args.n_dynamic_pcs)
    dyn_path = os.path.join(args.out, 'dynamic_pcs_timeseries.csv')
    dyn.to_csv(dyn_path, index=False)

    scales = np.arange(1, args.max_scale + 1)
    wav = wavelet_features(efd_ts, scales=scales)
    wav_path = os.path.join(args.out, 'wavelet_features.csv')
    wav.to_csv(wav_path, index=False)

    for k, v in [('efd_timeseries', efd_path), ('amp_timeseries', amp_path), ('static_pcs', stat_path), ('dynamic_pcs', dyn_path), ('wavelet_features', wav_path)]:
        print(f'{k}: {v}')


if __name__ == '__main__':
    main()

