#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Compute polygon area, siphon widths/areas, mantle area, contraction amplitude and speed.
"""
import os
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.errors import GEOSException
from typing import List, Tuple

COMMON = ['ExperimentID', 'Individuals', 'Bodyparts', 'Condition']

TILE_SIZE_UM     = 245.0   # 1 normalised unit = this many µm
SECONDS_PER_SAMPLE = 0.25  # 1 / effective_fps  (edit to match your recording)


def _time_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in COMMON]


def normalize_series(s: pd.Series) -> pd.Series:
    m = s.dropna().median()
    return s / m if pd.notna(m) and m != 0 else s


def process_metric(df: pd.DataFrame, lower: float, upper: float, smooth: int) -> pd.DataFrame:
    t = _time_cols(df)
    out = df.copy()
    out[t] = out[t].apply(normalize_series, axis=1)
    out[t] = out[t].applymap(lambda x: x if (pd.notna(x) and lower <= x <= upper) else np.nan)
    out[t] = out[t].apply(lambda s: s.interpolate(limit=20, limit_area='inside'), axis=1)
    out[t] = out[t].rolling(window=smooth, axis=1, min_periods=1).mean()
    return out


def _shoelace(xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
    ok = (~np.isnan(xs) & ~np.isnan(ys)).all(axis=0)
    xr = np.roll(xs, -1, axis=0)
    yr = np.roll(ys, -1, axis=0)
    a = 0.5 * np.abs(np.sum(xs * yr - ys * xr, axis=0))
    a[~ok] = np.nan
    return a


def _siphon_area(px: pd.DataFrame, py: pd.DataFrame, parts: List[str]) -> pd.Series:
    if not set(parts).issubset(px.index):
        return pd.Series(np.nan, index=px.columns)
    xs = px.loc[parts].to_numpy()
    ys = py.loc[parts].to_numpy()
    return pd.Series(_shoelace(xs, ys), index=px.columns)


def _mantle_area(px: pd.DataFrame, py: pd.DataFrame, exclude: List[str]) -> pd.Series:
    px2 = px.drop(index=exclude, errors='ignore')
    py2 = py.drop(index=exclude, errors='ignore')
    if px2.shape[0] < 3:
        return pd.Series(np.nan, index=px.columns)
    return pd.Series(_shoelace(px2.to_numpy(), py2.to_numpy()), index=px.columns)


def _extract_coords(df: pd.DataFrame, part: str) -> pd.Series:
    t = _time_cols(df)
    m = df.loc[df['Bodyparts'] == part]
    if m.empty:
        return pd.Series(np.nan, index=t)
    s = pd.to_numeric(m[t].iloc[0], errors='coerce')
    s.index = t
    return s


def polygon_areas(df_x: pd.DataFrame, df_y: pd.DataFrame) -> pd.DataFrame:
    t = _time_cols(df_x)
    rows = []
    for (exp, ind), gx in df_x.groupby(['ExperimentID', 'Individuals']):
        gy = df_y[(df_y['ExperimentID'] == exp) & (df_y['Individuals'] == ind)]
        d = {'ExperimentID': exp, 'Individuals': ind}
        for col in t:
            x = pd.to_numeric(gx[col], errors='coerce')
            y = pd.to_numeric(gy[col], errors='coerce')
            verts = list(zip(x.values, y.values))
            if len(set(verts)) >= 3:
                try:
                    poly = Polygon(verts + [verts[0]])
                    d[col] = poly.area if poly.is_valid else np.nan
                except GEOSException:
                    d[col] = np.nan
            else:
                d[col] = np.nan
        rows.append(d)
    return pd.DataFrame(rows)


def siphon_widths(df_x: pd.DataFrame, df_y: pd.DataFrame, p1: str, p2: str) -> pd.DataFrame:
    t = _time_cols(df_x)
    out = []
    for (exp, ind), gx in df_x.groupby(['ExperimentID', 'Individuals']):
        gy = df_y[(df_y['ExperimentID'] == exp) & (df_y['Individuals'] == ind)]
        x1 = _extract_coords(gx, p1); y1 = _extract_coords(gy, p1)
        x2 = _extract_coords(gx, p2); y2 = _extract_coords(gy, p2)
        w = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        out.append({'ExperimentID': exp, 'Individuals': ind, **dict(zip(t, w.values))})
    return pd.DataFrame(out)


def siphon_and_mantle_areas(df_x: pd.DataFrame, df_y: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    oral_parts   = ['Dos', 'Vos', 'Doc', 'Voc']
    atrial_parts = ['Das', 'Vas', 'Dac', 'Vac']
    exclude      = ['Dos', 'Vos', 'Das', 'Vas']
    oral, atrial, mantle = [], [], []

    for (exp, ind), gx in df_x.groupby(['ExperimentID', 'Individuals']):
        gy = df_y[(df_y['ExperimentID'] == exp) & (df_y['Individuals'] == ind)]
        px = gx.set_index('Bodyparts')[_time_cols(gx)].apply(pd.to_numeric, errors='coerce')
        py = gy.set_index('Bodyparts')[_time_cols(gy)].apply(pd.to_numeric, errors='coerce')
        oral.append(  {'ExperimentID': exp, 'Individuals': ind, **_siphon_area(px, py, oral_parts).to_dict()})
        atrial.append({'ExperimentID': exp, 'Individuals': ind, **_siphon_area(px, py, atrial_parts).to_dict()})
        mantle.append({'ExperimentID': exp, 'Individuals': ind, **_mantle_area(px, py, exclude).to_dict()})

    return pd.DataFrame(oral), pd.DataFrame(atrial), pd.DataFrame(mantle)


def _add_condition(df_metrics: pd.DataFrame, df_x: pd.DataFrame) -> pd.DataFrame:
    cond = (
        df_x[['ExperimentID', 'Individuals', 'Condition']]
        .drop_duplicates()
        .set_index(['ExperimentID', 'Individuals'])['Condition']
        .to_dict()
    )
    out = df_metrics.copy()
    out['Condition'] = out.apply(lambda r: cond.get((r['ExperimentID'], r['Individuals']), np.nan), axis=1)
    first = ['ExperimentID', 'Individuals', 'Condition']
    rest  = [c for c in out.columns if c not in first]
    return out[first + rest]


def extract_amp_speed(events_df: pd.DataFrame, proc_dfs: dict,
                      tile_size_um: float = TILE_SIZE_UM,
                      seconds_per_sample: float = SECONDS_PER_SAMPLE) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Compute contraction amplitude (µm) and contraction/relaxation speed (µm/s)
    for each detected event, then aggregate per individual.

    Parameters
    ----------
    events_df         : categorized contraction events DataFrame
    proc_dfs          : dict of {label: processed_metric_df}, e.g.
                        {'Mantle Area': df, 'Oral Siphon Width': df, ...}
    tile_size_um      : physical scale – 1 normalised unit = tile_size_um µm
    seconds_per_sample: 1 / effective_fps

    Returns
    -------
    amp_event_df : one row per event with Amplitude_um, Speed_contract_um_s, Speed_relax_um_s
    amp_agg_df   : per-individual mean of the above, split by contraction type (All/Full/Half)
    """
    amp_speed_features = ['Amplitude_um', 'Speed_contract_um_s', 'Speed_relax_um_s']

    if events_df is None or events_df.empty:
        return pd.DataFrame(), pd.DataFrame()

    def _time_cols_generic(df):
        return [c for c in df.columns
                if str(c).startswith('t') and str(c)[1:].isdigit()]

    # Pre-index each processed df for fast row lookup
    stores = {}
    for label, dfm in proc_dfs.items():
        tc = _time_cols_generic(dfm)
        if not tc:
            continue
        tmp  = dfm.set_index(['ExperimentID', 'Individuals'])
        mat  = tmp[tc].apply(pd.to_numeric, errors='coerce').to_numpy(dtype=float)
        keys = list(tmp.index)
        stores[label] = (mat, {k: i for i, k in enumerate(keys)}, len(tc))

    rows  = []
    scale = float(tile_size_um)
    sp    = seconds_per_sample

    for _, ev in events_df.iterrows():
        key       = (ev['ExperimentID'], ev['Individuals'])
        ctype     = ev.get('Category', 'all')
        ev_metric = str(ev['Metric'])

        match_label = None
        for label in stores:
            if label.lower() in ev_metric.lower() or ev_metric.lower() in label.lower():
                match_label = label
                break
        if match_label is None:
            continue

        mat, k2r, n_tcols = stores[match_label]
        ridx = k2r.get(key)
        if ridx is None:
            continue

        s_idx = int(ev['Start_index'])
        e_idx = min(int(ev['End_index']), n_tcols - 1)
        p_idx = int(ev['Peak_index'])
        if s_idx >= n_tcols or s_idx > e_idx:
            continue

        seg   = mat[ridx, s_idx : e_idx + 1] * scale
        p_rel = max(0, min(p_idx - s_idx, len(seg) - 1))
        if len(seg) < 2 or np.all(np.isnan(seg)):
            continue

        amp        = float(np.nanmax(seg) - np.nanmin(seg))
        t_contract = p_rel * sp
        speed_c    = (float((seg[p_rel] - seg[0]) / t_contract)
                      if t_contract > 0 and np.isfinite(seg[p_rel]) and np.isfinite(seg[0])
                      else np.nan)
        t_relax    = (len(seg) - 1 - p_rel) * sp
        speed_r    = (float((seg[p_rel] - seg[-1]) / t_relax)
                      if t_relax > 0 and np.isfinite(seg[p_rel]) and np.isfinite(seg[-1])
                      else np.nan)

        rows.append({
            'ExperimentID':        ev['ExperimentID'],
            'Individuals':         ev['Individuals'],
            'Condition':           ev['Condition'],
            'Specific':            ev.get('Specific'),
            'Metric':              ev['Metric'],
            'Category':            ctype,
            'Amplitude_um':        amp,
            'Speed_contract_um_s': speed_c,
            'Speed_relax_um_s':    speed_r,
        })

    amp_event_df = pd.DataFrame(rows)
    if amp_event_df.empty:
        return amp_event_df, pd.DataFrame()

    agg_rows = []
    for subset_label, subset_df in [
        ('All',  amp_event_df),
        ('Full', amp_event_df[amp_event_df['Category'] == 'full']),
        ('Half', amp_event_df[amp_event_df['Category'] == 'half']),
    ]:
        if subset_df.empty:
            continue
        agg = subset_df.groupby(
            ['ExperimentID', 'Individuals', 'Condition', 'Specific', 'Metric'],
            as_index=False, dropna=False
        )[amp_speed_features].mean()
        agg['Contraction_type'] = subset_label
        agg_rows.append(agg)

    amp_agg_df = pd.concat(agg_rows, ignore_index=True) if agg_rows else pd.DataFrame()
    return amp_event_df, amp_agg_df


def run(df_x: pd.DataFrame, df_y: pd.DataFrame, smooth: int, outdir: str) -> List[str]:
    os.makedirs(outdir, exist_ok=True)

    poly      = polygon_areas(df_x, df_y)
    oral_w    = siphon_widths(df_x, df_y, 'Dos', 'Vos')
    atrial_w  = siphon_widths(df_x, df_y, 'Das', 'Vas')
    oral_a, atrial_a, mantle_a = siphon_and_mantle_areas(df_x, df_y)

    poly_p    = process_metric(poly,     0, 1.1, smooth)
    oralw_p   = process_metric(oral_w,   0, 1.4, smooth)
    atrialw_p = process_metric(atrial_w, 0, 1.8, smooth)
    orala_p   = process_metric(oral_a,   0, 1.4, smooth)
    atriala_p = process_metric(atrial_a, 0, 1.8, smooth)
    mantle_p  = process_metric(mantle_a, 0, 1.1, smooth)

    out_files = []
    for name, df in [
        ('polygon_areas_processed.csv',        poly_p),
        ('oral_siphon_widths_processed.csv',   oralw_p),
        ('atrial_siphon_widths_processed.csv', atrialw_p),
        ('oral_siphon_areas_processed.csv',    orala_p),
        ('atrial_siphon_areas_processed.csv',  atriala_p),
        ('mantle_areas_processed.csv',         mantle_p),
    ]:
        path = os.path.join(outdir, name)
        _add_condition(df, df_x).to_csv(path, index=False)
        out_files.append(path)
    return out_files


def main():
    X_PATH = './output_data/Ciona_adult_plotting/x_final.csv'   # ← edit
    Y_PATH = './output_data/Ciona_adult_plotting/y_final.csv'   # ← edit
    OUTDIR = './output_data/Ciona_adult_metrics'
    SMOOTH = 5

    df_x = pd.read_csv(X_PATH)
    df_y = pd.read_csv(Y_PATH)

    for f in run(df_x, df_y, SMOOTH, OUTDIR):
        print(f)


if __name__ == '__main__':
    main()