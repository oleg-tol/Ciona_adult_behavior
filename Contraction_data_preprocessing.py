#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Detrend, interpolate, and z-score six shape metrics.
"""
import os
import numpy as np
import pandas as pd
from typing import List, Tuple
from statsmodels.tsa.seasonal import seasonal_decompose
from scipy.interpolate import interp1d
from tslearn.preprocessing import TimeSeriesScalerMeanVariance

COMMON = ['ExperimentID', 'Individuals', 'Condition']


def _time_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in COMMON]


def _common_time_cols(dfs: List[pd.DataFrame]) -> List[str]:
    sets = [set(_time_cols(d)) for d in dfs if not d.empty]
    if not sets:
        return []
    inter = set.intersection(*sets)
    return sorted(list(inter), key=lambda x: float(x) if str(x).replace('.', '', 1).isdigit() else str(x))


def decompose_series(s: pd.Series, model: str, period: int):
    x = s.dropna()
    if len(x) < 2 * period:
        return None, None, None
    d = seasonal_decompose(x, model=model, period=period)
    return d.trend, d.seasonal, d.resid


def decompose_and_extract(df: pd.DataFrame, cols: List[str], period: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    trends, seasons = [], []
    for key in df[['ExperimentID', 'Individuals', 'Condition']].drop_duplicates().itertuples(index=False, name=None):
        exp, ind, cond = key
        series = df[(df['ExperimentID'] == exp) & (df['Individuals'] == ind)].iloc[0][cols].astype(float)
        trend, season, _ = decompose_series(series, model='additive', period=period)
        if trend is None:
            continue
        trend  = (trend * -1).reindex(cols)
        season = season.reindex(cols)
        d_t = {'ExperimentID': exp, 'Individuals': ind, 'Condition': cond, **dict(zip(cols, trend))}
        d_s = {'ExperimentID': exp, 'Individuals': ind, 'Condition': cond, **dict(zip(cols, season))}
        trends.append(d_t)
        seasons.append(d_s)
    return pd.DataFrame(trends), pd.DataFrame(seasons)


def remove_sparse_rows(df: pd.DataFrame, threshold: float) -> pd.DataFrame:
    t    = _time_cols(df)
    need = int(threshold * len(t))
    keep = df[t].notna().sum(axis=1) >= need
    return df.loc[keep].reset_index(drop=True)


def _ids(df: pd.DataFrame):
    return set(zip(df['ExperimentID'], df['Individuals']))


def _filter_common_ids(df: pd.DataFrame, ids) -> pd.DataFrame:
    out = df.copy()
    out['__id__'] = list(zip(out['ExperimentID'], out['Individuals']))
    out = out[out['__id__'].isin(ids)].drop(columns='__id__').reset_index(drop=True)
    return out


def recover_lost_data(df: pd.DataFrame, lost_points: int) -> pd.DataFrame:
    out = df.copy()
    t0  = len(COMMON)
    for i, row in out.iterrows():
        vals = pd.to_numeric(row.iloc[t0:], errors='coerce').values
        idx  = np.arange(lost_points, len(vals))
        seg  = vals[lost_points:]
        if np.isfinite(seg).sum() > 1:
            f = interp1d(idx, seg, kind='linear', fill_value='extrapolate')
            out.iloc[i, t0:t0 + lost_points] = f(np.arange(0, lost_points))
    return out


def fill_gaps(df: pd.DataFrame, max_nan_fill: int) -> pd.DataFrame:
    out = df.copy()
    t0  = len(COMMON)
    for i, row in out.iterrows():
        vals = pd.to_numeric(row.iloc[t0:], errors='coerce').values
        mask = np.isfinite(vals)
        if not mask.any():
            continue
        last = np.where(mask)[0][-1]
        seg  = pd.Series(vals[: last + 1]).interpolate(limit=max_nan_fill).values
        out.iloc[i, t0:t0 + len(seg)] = seg
    return out


def mark_first_last(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    t   = _time_cols(df)
    for i, row in out.iterrows():
        s  = row[t]
        ok = s.notna()
        if ok.any():
            out.loc[i, ok.idxmax()]          = -1
            out.loc[i, ok[::-1].idxmax()]    = -1
    return out


def zscore_timeseries(df: pd.DataFrame) -> pd.DataFrame:
    t = _time_cols(df)
    X = df[t].apply(pd.to_numeric, errors='coerce')
    X = X.dropna(axis=1, how='all')
    Z = TimeSeriesScalerMeanVariance().fit_transform(X.values)[:, :, 0]
    out = pd.DataFrame(Z, index=df.index, columns=X.columns)
    return pd.concat([df[COMMON].reset_index(drop=True), out.reset_index(drop=True)], axis=1)


def _load_metrics(metrics_dir: str) -> List[pd.DataFrame]:
    paths = [
        'polygon_areas_processed.csv',
        'oral_siphon_widths_processed.csv',
        'atrial_siphon_widths_processed.csv',
        'oral_siphon_areas_processed.csv',
        'atrial_siphon_areas_processed.csv',
        'mantle_areas_processed.csv',
    ]
    dfs = []
    for p in paths:
        fp = os.path.join(metrics_dir, p)
        if not os.path.exists(fp):
            raise FileNotFoundError(fp)
        dfs.append(pd.read_csv(fp))
    return dfs


def main():
    METRICS_DIR  = './output_data/Ciona_adult_metrics'   # ← edit
    OUTDIR       = './output'
    PERIOD       = 50
    THRESHOLD    = 0.2
    LOST_POINTS  = 25
    MAX_NAN_FILL = 200

    polys, ow, aw, oa, aa, ma = _load_metrics(METRICS_DIR)

    cols = _common_time_cols([polys, ow, aw, oa, aa, ma])
    if not cols:
        raise ValueError('No common time columns across metrics.')

    poly_t, _ = decompose_and_extract(polys, cols, PERIOD)
    ow_t,   _ = decompose_and_extract(ow,   cols, PERIOD)
    aw_t,   _ = decompose_and_extract(aw,   cols, PERIOD)
    oa_t,   _ = decompose_and_extract(oa,   cols, PERIOD)
    aa_t,   _ = decompose_and_extract(aa,   cols, PERIOD)
    ma_t,   _ = decompose_and_extract(ma,   cols, PERIOD)

    poly_t = remove_sparse_rows(poly_t, THRESHOLD)
    ow_t   = remove_sparse_rows(ow_t,   THRESHOLD)
    aw_t   = remove_sparse_rows(aw_t,   THRESHOLD)
    oa_t   = remove_sparse_rows(oa_t,   THRESHOLD)
    aa_t   = remove_sparse_rows(aa_t,   THRESHOLD)
    ma_t   = remove_sparse_rows(ma_t,   THRESHOLD)

    ids = _ids(poly_t) & _ids(ow_t) & _ids(aw_t) & _ids(oa_t) & _ids(aa_t) & _ids(ma_t)

    poly_t = _filter_common_ids(poly_t, ids)
    ow_t   = _filter_common_ids(ow_t,   ids)
    aw_t   = _filter_common_ids(aw_t,   ids)
    oa_t   = _filter_common_ids(oa_t,   ids)
    aa_t   = _filter_common_ids(aa_t,   ids)
    ma_t   = _filter_common_ids(ma_t,   ids)

    poly_t = recover_lost_data(poly_t, LOST_POINTS)
    ow_t   = recover_lost_data(ow_t,   LOST_POINTS)
    aw_t   = recover_lost_data(aw_t,   LOST_POINTS)
    oa_t   = recover_lost_data(oa_t,   LOST_POINTS)
    aa_t   = recover_lost_data(aa_t,   LOST_POINTS)
    ma_t   = recover_lost_data(ma_t,   LOST_POINTS)

    poly_t = fill_gaps(poly_t, MAX_NAN_FILL)
    ow_t   = fill_gaps(ow_t,   MAX_NAN_FILL)
    aw_t   = fill_gaps(aw_t,   MAX_NAN_FILL)
    oa_t   = fill_gaps(oa_t,   MAX_NAN_FILL)
    aa_t   = fill_gaps(aa_t,   MAX_NAN_FILL)
    ma_t   = fill_gaps(ma_t,   MAX_NAN_FILL)

    poly_t = mark_first_last(poly_t)
    ow_t   = mark_first_last(ow_t)
    aw_t   = mark_first_last(aw_t)
    oa_t   = mark_first_last(oa_t)
    aa_t   = mark_first_last(aa_t)
    ma_t   = mark_first_last(ma_t)

    os.makedirs(OUTDIR, exist_ok=True)

    out_files = [
        ('polygon_areas_zscored.csv',          zscore_timeseries(poly_t)),
        ('oral_siphon_widths_zscored.csv',     zscore_timeseries(ow_t)),
        ('atrial_siphon_widths_zscored.csv',   zscore_timeseries(aw_t)),
        ('oral_siphon_areas_zscored.csv',      zscore_timeseries(oa_t)),
        ('atrial_siphon_areas_zscored.csv',    zscore_timeseries(aa_t)),
        ('mantle_areas_zscored.csv',           zscore_timeseries(ma_t)),
    ]

    for name, df in out_files:
        df.to_csv(os.path.join(OUTDIR, name), index=False, float_format='%.3f')
        print(os.path.join(OUTDIR, name))

    # sanity: recompute common time columns after z-scoring
    zs = [df for _, df in out_files]
    c  = _common_time_cols(zs)
    if not c:
        raise ValueError('No common time columns after z-scoring.')


if __name__ == '__main__':
    main()

