"""
Compute polygon area, siphon widths/areas, and mantle area from x/y CSVs.
"""
import os
import numpy as np
import pandas as pd
from shapely.geometry import Polygon
from shapely.errors import GEOSException
from typing import List, Tuple

COMMON = ['ExperimentID', 'Individuals', 'Bodyparts', 'Condition']


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
    t = _time_cols(df_x)
    oral_parts = ['Dos', 'Vos', 'Doc', 'Voc']
    atrial_parts = ['Das', 'Vas', 'Dac', 'Vac']
    exclude = ['Dos', 'Vos', 'Das', 'Vas']
    oral, atrial, mantle = [], [], []

    gx_groups = df_x.groupby(['ExperimentID', 'Individuals'])
    gy_groups = df_y.groupby(['ExperimentID', 'Individuals'])

    for (exp, ind), gx in gx_groups:
        gy = gy_groups.get_group((exp, ind))
        px = gx.set_index('Bodyparts')[_time_cols(gx)].apply(pd.to_numeric, errors='coerce')
        py = gy.set_index('Bodyparts')[_time_cols(gy)].apply(pd.to_numeric, errors='coerce')
        a_oral = _siphon_area(px, py, oral_parts)
        a_atr = _siphon_area(px, py, atrial_parts)
        a_man = _mantle_area(px, py, exclude)
        oral.append({'ExperimentID': exp, 'Individuals': ind, **a_oral.to_dict()})
        atrial.append({'ExperimentID': exp, 'Individuals': ind, **a_atr.to_dict()})
        mantle.append({'ExperimentID': exp, 'Individuals': ind, **a_man.to_dict()})

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
    rest = [c for c in out.columns if c not in first]
    return out[first + rest]


def run(df_x: pd.DataFrame, df_y: pd.DataFrame, smooth: int, outdir: str) -> List[str]:
    os.makedirs(outdir, exist_ok=True)

    poly = polygon_areas(df_x, df_y)
    oral_w = siphon_widths(df_x, df_y, 'Dos', 'Vos')
    atrial_w = siphon_widths(df_x, df_y, 'Das', 'Vas')
    oral_a, atrial_a, mantle_a = siphon_and_mantle_areas(df_x, df_y)

    poly_p    = process_metric(poly,     0, 1.1, smooth)
    oralw_p   = process_metric(oral_w,   0, 1.4, smooth)
    atrialw_p = process_metric(atrial_w, 0, 1.8, smooth)
    orala_p   = process_metric(oral_a,   0, 1.4, smooth)
    atriala_p = process_metric(atrial_a, 0, 1.8, smooth)
    mantle_p  = process_metric(mantle_a, 0, 1.1, smooth)

    out_files = []
    for name, df in [
        ('polygon_areas_processed.csv', poly_p),
        ('oral_siphon_widths_processed.csv', oralw_p),
        ('atrial_siphon_widths_processed.csv', atrialw_p),
        ('oral_siphon_areas_processed.csv', orala_p),
        ('atrial_siphon_areas_processed.csv', atriala_p),
        ('mantle_areas_processed.csv', mantle_p),
    ]:
        path = os.path.join(outdir, name)
        _add_condition(df, df_x).to_csv(path, index=False)
        out_files.append(path)
    return out_files


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description='Compute shape metrics from x/y CSVs.')
    p.add_argument('--x', required=True)
    p.add_argument('--y', required=True)
    p.add_argument('--out', default='./output_data/Ciona_adult_metrics')
    p.add_argument('--smooth', type=int, default=5)
    args = p.parse_args()

    df_x = pd.read_csv(args.x)
    df_y = pd.read_csv(args.y)

    for f in run(df_x, df_y, args.smooth, args.out):
        print(f)


if __name__ == '__main__':
    main()

