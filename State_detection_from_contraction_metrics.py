#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
State detection from z-scored contraction metrics.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from typing import List, Tuple

COMMON = ['ExperimentID', 'Individuals', 'Condition', 'Specific']


# ── Signal helpers ────────────────────────────────────────────────────────────

def smooth_signal(sig: np.ndarray, window: int, order: int) -> np.ndarray:
    sig = np.asarray(sig, dtype=float)
    pad = window
    ext = np.pad(sig, pad, mode='edge')
    sm  = savgol_filter(ext, window, order, mode='nearest')
    return sm[pad:-pad]


def mark_boundary_signal(sig: np.ndarray) -> np.ndarray:
    """Set first and last valid frame to -1 (boundary artefact marker)."""
    out   = np.asarray(sig, dtype=float).copy()
    valid = np.where(~np.isnan(out))[0]
    if valid.size > 0:
        out[valid[0]]  = -1
        out[valid[-1]] = -1
    return out


def offset_positive(sig: np.ndarray) -> np.ndarray:
    mn = np.nanmin(sig)
    return sig - mn if mn < 0 else sig + 1


def detect_contractions(sig: np.ndarray, T: int,
                         prom: float, dist: int, width: int,
                         smooth_win: int, poly: int):
    """Returns peaks, lefts, rights, prominences, binary mask, smoothed signal."""
    s = pd.Series(sig).interpolate(method='linear', limit_direction='both').values
    s = offset_positive(s)
    s = smooth_signal(s, smooth_win, poly)

    peaks, props = find_peaks(s, prominence=prom, distance=dist, width=width)
    proms  = props.get('prominences', np.array([]))
    lefts  = np.floor(props.get('left_ips',  np.array([]))).astype(int)
    rights = np.ceil( props.get('right_ips', np.array([]))).astype(int)

    binary = np.zeros(T, dtype=float)
    for l, r in zip(lefts, rights):
        binary[max(int(l), 0) : min(int(r), T - 1) + 1] = 1

    return peaks, lefts, rights, proms, binary, s


def detect_quiescence(binary: np.ndarray, smoothed: np.ndarray,
                       threshold: float, tol: float) -> List[Tuple[int, int]]:
    """Return list of (start, end) quiescent intervals."""
    events, start = [], None
    for i, val in enumerate(binary):
        in_q = (val == 0) and (smoothed[i] < threshold + tol)
        if in_q and start is None:
            start = i
        elif (not in_q) and start is not None:
            events.append((start, i - 1))
            start = None
    if start is not None:
        events.append((start, len(binary) - 1))
    return events


def categorize_events(df: pd.DataFrame, thresh: float) -> pd.DataFrame:
    out = []
    if df.empty:
        return pd.DataFrame()
    for (e, i, m), grp in df.groupby(['ExperimentID', 'Individuals', 'Metric'], sort=False):
        grp = grp.copy()
        max_h           = grp['Peak_value'].max()
        grp['Category'] = np.where(grp['Peak_value'] >= thresh * max_h, 'full', 'half')
        grp['Duration'] = grp['End_index'].astype(int) - grp['Start_index'].astype(int) + 1
        out.append(grp)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


# ── Time-column helpers ───────────────────────────────────────────────────────

def _tcols(df_or_row) -> List[str]:
    if isinstance(df_or_row, pd.Series):
        cols = df_or_row.index
    else:
        cols = df_or_row.columns
    skip = set(COMMON)
    return sorted([c for c in cols if c not in skip and
                   str(c).startswith('t') and str(c)[1:].isdigit()],
                  key=lambda c: int(str(c)[1:]))


def _prep_signal(row: pd.Series, tc: List[str]) -> np.ndarray:
    s = pd.to_numeric(row[tc], errors='coerce')
    s = (pd.Series(s.values, index=tc)
           .interpolate(method='linear', limit_area='inside')
           .bfill().ffill())
    return s.values.astype(float)


def _get_row(idx_df: pd.DataFrame, key) -> pd.Series:
    r = idx_df.loc[key]
    return r.iloc[0] if isinstance(r, pd.DataFrame) else r


# ── State classification ──────────────────────────────────────────────────────

def state_intervals_for_metric(categorized_df: pd.DataFrame,
                                exp_id, ind, metric: str, n: int) -> list:
    """Build merged state intervals for one (individual, metric)."""
    evs = categorized_df[
        (categorized_df['ExperimentID'] == exp_id) &
        (categorized_df['Individuals']  == ind)    &
        (categorized_df['Metric']       == metric)
    ]
    event_list = sorted(
        [(int(r.Start_index), int(r.End_index) + 1, str(r.Category).lower())
         for _, r in evs.iterrows()],
        key=lambda x: x[0]
    )
    bounds = sorted({0, n} | {s for s, e, _ in event_list} | {e for _, e, _ in event_list})
    spans  = []
    for i in range(len(bounds) - 1):
        s, e    = bounds[i], bounds[i + 1]
        overlap = [ev for ev in event_list if not (ev[1] <= s or ev[0] >= e)]
        if not overlap:
            state = 'Quiescence'
        elif all(ev[2] == 'full' for ev in overlap):
            state = 'Full Contraction'
        else:
            state = 'Half Contraction'
        spans.append((s, e, state))
    merged = []
    if spans:
        cs, ce, cst = spans[0]
        for s, e, st in spans[1:]:
            if st == cst:
                ce = e
            else:
                merged.append((cs, ce, cst)); cs, ce, cst = s, e, st
        merged.append((cs, ce, cst))
    return merged


def unify_state(categorized_df: pd.DataFrame, exp_id, ind, n: int):
    """Vote across three metrics to assign a unified state per frame."""
    def state_at(t, ints):
        for s, e, st in ints:
            if s <= t < e:
                return st
        return 'Quiescence'

    m_ints = state_intervals_for_metric(categorized_df, exp_id, ind, 'Mantle Area',         n)
    o_ints = state_intervals_for_metric(categorized_df, exp_id, ind, 'Oral Siphon Width',   n)
    a_ints = state_intervals_for_metric(categorized_df, exp_id, ind, 'Atrial Siphon Width', n)

    unified = []
    for t in range(n):
        sm, so, sa = state_at(t, m_ints), state_at(t, o_ints), state_at(t, a_ints)
        if sm == so == sa == 'Full Contraction':
            unified.append('Full Contraction')
        elif any(s != 'Quiescence' for s in (sm, so, sa)):
            unified.append('Half Contraction')
        else:
            unified.append('Quiescence')

    merged = []
    if unified:
        cs, cst = 0, unified[0]
        for i in range(1, n):
            if unified[i] != cst:
                merged.append((cs, i, cst)); cs, cst = i, unified[i]
        merged.append((cs, n, cst))
    return merged, unified


# ── Main processing function ──────────────────────────────────────────────────

def process(zdir: str, prom: float, dist: int, width: int,
            smooth_win: int, poly: int, cat_thresh: float, tol: float,
            sec_per_sample: float, out_data: str, out_plots: str,
            plot_all: bool) -> dict:

    os.makedirs(out_data,  exist_ok=True)
    os.makedirs(out_plots, exist_ok=True)

    mantle = pd.read_csv(os.path.join(zdir, 'mantle_areas_zscored.csv'))
    oral   = pd.read_csv(os.path.join(zdir, 'oral_siphon_widths_zscored.csv'))
    atrial = pd.read_csv(os.path.join(zdir, 'atrial_siphon_widths_zscored.csv'))

    mantle_idx  = mantle.set_index(['ExperimentID', 'Individuals'], drop=False)
    oral_idx    = oral.set_index(  ['ExperimentID', 'Individuals'], drop=False)
    atrial_idx  = atrial.set_index(['ExperimentID', 'Individuals'], drop=False)

    common_keys = (set(mantle_idx.index) & set(oral_idx.index) & set(atrial_idx.index))
    print(f"  Detecting events for {len(common_keys)} individuals…")

    mantle_events, oral_events, atrial_events = [], [], []
    all_quiescence = []
    mantle_smoothed_list, oral_smoothed_list, atrial_smoothed_list = [], [], []

    for (exp_id, ind) in sorted(common_keys, key=lambda x: (str(x[0]), str(x[1]))):
        m_row = _get_row(mantle_idx,  (exp_id, ind))
        o_row = _get_row(oral_idx,    (exp_id, ind))
        a_row = _get_row(atrial_idx,  (exp_id, ind))

        condition = m_row.get('Condition', None)
        specific  = m_row.get('Specific',  None)

        tcols = sorted(
            set(_tcols(m_row)) & set(_tcols(o_row)) & set(_tcols(a_row)),
            key=lambda c: int(str(c)[1:])
        )
        T = len(tcols)
        if T == 0:
            continue

        signals = {
            'Mantle Area':         _prep_signal(m_row, tcols),
            'Oral Siphon Width':   _prep_signal(o_row, tcols),
            'Atrial Siphon Width': _prep_signal(a_row, tcols),
        }

        smoothed_store = {}
        for metric, raw_sig in signals.items():
            sm  = smooth_signal(raw_sig, smooth_win, poly)
            sig = mark_boundary_signal(sm)
            smoothed_store[metric] = sig

            peaks, lefts, rights, proms, binary, _ = detect_contractions(
                sig, T, prom, dist, width, smooth_win, poly)
            threshold   = np.nanmedian(sig)
            q_intervals = detect_quiescence(binary, sig, threshold, tol)

            ev_list = (mantle_events if metric == 'Mantle Area' else
                       oral_events   if metric == 'Oral Siphon Width' else
                       atrial_events)

            for p, l, r, pr in zip(peaks, lefts, rights, proms):
                ev_list.append({
                    'ExperimentID': exp_id, 'Individuals': ind,
                    'Condition':    condition, 'Specific': specific,
                    'Start':        tcols[int(l)], 'Peak': tcols[int(p)], 'End': tcols[int(r)],
                    'Start_index':  int(l), 'Peak_index': int(p), 'End_index': int(r),
                    'Prominence':   float(pr), 'Peak_value': float(sig[int(p)]),
                    'Metric':       metric,
                })

            for s_i, e_i in q_intervals:
                all_quiescence.append({
                    'ExperimentID': exp_id, 'Individuals': ind,
                    'Condition':    condition, 'Specific': specific,
                    'Metric':       metric,
                    'Start':        int(s_i), 'End': int(e_i),
                    'Duration_samples': int(e_i - s_i + 1),
                    'Duration_seconds': float((e_i - s_i + 1) * sec_per_sample),
                    'SignalThreshold':  float(threshold),
                })

        base = {'ExperimentID': exp_id, 'Individuals': ind,
                'Condition': condition, 'Specific': specific}
        mantle_smoothed_list.append( {**base, **dict(zip(tcols, smoothed_store['Mantle Area']))})
        oral_smoothed_list.append(   {**base, **dict(zip(tcols, smoothed_store['Oral Siphon Width']))})
        atrial_smoothed_list.append( {**base, **dict(zip(tcols, smoothed_store['Atrial Siphon Width']))})

    mantle_sm  = pd.DataFrame(mantle_smoothed_list)
    oral_sm    = pd.DataFrame(oral_smoothed_list)
    atrial_sm  = pd.DataFrame(atrial_smoothed_list)
    quiescence_df = pd.DataFrame(all_quiescence)

    all_events_df = pd.concat([
        pd.DataFrame(mantle_events),
        pd.DataFrame(oral_events),
        pd.DataFrame(atrial_events),
    ], ignore_index=True)

    categorized_df = categorize_events(all_events_df, cat_thresh)

    print(f"  ✓ Contractions detected: {len(categorized_df)}")
    if not categorized_df.empty:
        print(f"      full: {(categorized_df['Category']=='full').sum()}  "
              f"half: {(categorized_df['Category']=='half').sum()}")
    print(f"  ✓ Quiescence periods: {len(quiescence_df)}")

    # Save events and smoothed signals
    categorized_df.to_csv(os.path.join(out_data, 'contraction_events.csv'), index=False)
    quiescence_df.to_csv( os.path.join(out_data, 'quiescence_events.csv'),  index=False)
    mantle_sm.to_csv(     os.path.join(out_data, 'mantle_smoothed.csv'),    index=False, float_format='%.4f')
    oral_sm.to_csv(       os.path.join(out_data, 'oral_smoothed.csv'),      index=False, float_format='%.4f')
    atrial_sm.to_csv(     os.path.join(out_data, 'atrial_smoothed.csv'),    index=False, float_format='%.4f')

    # ── Unified state map ─────────────────────────────────────────────────────
    print("  Building unified state map…")
    mantle_sm_idx = mantle_sm.set_index(['ExperimentID', 'Individuals'], drop=False)
    oral_sm_set   = set(zip(oral_sm['ExperimentID'], oral_sm['Individuals']))
    state_map_rows = []

    for (exp_id, ind) in mantle_sm_idx.index.unique():
        if (exp_id, ind) not in oral_sm_set:
            continue
        m_row = _get_row(mantle_sm_idx, (exp_id, ind))
        condition = m_row.get('Condition', None)
        specific  = m_row.get('Specific',  None)
        tcols = _tcols(m_row)
        n     = len(tcols)
        if n == 0:
            continue
        _, unified_series = unify_state(categorized_df, exp_id, ind, n)
        row = {'Condition': condition, 'ExperimentID': exp_id,
               'Individuals': ind, 'Specific': specific}
        row.update({f't{i}': unified_series[i] for i in range(n)})
        state_map_rows.append(row)

    state_map_df = pd.DataFrame(state_map_rows)
    state_map_df.to_csv(os.path.join(out_plots, 'unified_behavior_states_map.csv'), index=False)
    print(f"  ✓ {len(state_map_df)} individuals → unified_behavior_states_map.csv")

    # ── Per-animal plots ──────────────────────────────────────────────────────
    cmap = {
        'Full Contraction': plt.cm.gnuplot2(0.3),
        'Half Contraction': plt.cm.gnuplot2(0.6),
        'Quiescence':       plt.cm.gnuplot2(0.9),
    }

    def _plot_one(e: str, i: str):
        mrow = mantle_sm[(mantle_sm['ExperimentID'] == e) & (mantle_sm['Individuals'] == i)]
        orow = oral_sm[  (oral_sm['ExperimentID']   == e) & (oral_sm['Individuals']   == i)]
        arow = atrial_sm[(atrial_sm['ExperimentID'] == e) & (atrial_sm['Individuals'] == i)]
        if mrow.empty or orow.empty or arow.empty:
            return
        m_sig = pd.to_numeric(mrow.iloc[0][_tcols(mantle_sm)], errors='coerce').dropna().values + 2
        o_sig = pd.to_numeric(orow.iloc[0][_tcols(oral_sm)],   errors='coerce').dropna().values + 2
        a_sig = pd.to_numeric(arow.iloc[0][_tcols(atrial_sm)], errors='coerce').dropna().values + 2
        n = min(len(m_sig), len(o_sig), len(a_sig))
        m_sig, o_sig, a_sig = m_sig[:n], o_sig[:n], a_sig[:n]
        t = np.arange(n) * (sec_per_sample / 60.0)

        m_ints = state_intervals_for_metric(categorized_df, e, i, 'Mantle Area',         n)
        o_ints = state_intervals_for_metric(categorized_df, e, i, 'Oral Siphon Width',   n)
        a_ints = state_intervals_for_metric(categorized_df, e, i, 'Atrial Siphon Width', n)
        u_ints, _ = unify_state(categorized_df, e, i, n)

        def _plot_metric(ax, sig, ints, title, color):
            ax.plot(t, sig, color=color)
            for s, u, st in ints:
                ax.fill_between(t[s:u], 0, sig[s:u],
                                color=cmap[{'full': 'Full Contraction', 'half': 'Half Contraction'}.get(st, st)],
                                alpha=0.6)
                ax.axvline(t[s],     color='k', linestyle=':', lw=1)
                ax.axvline(t[u - 1], color='k', linestyle=':', lw=1)
            ax.set_xlim(0, t[-1]); ax.set_ylabel('Z-score'); ax.set_title(title)

        plt.figure(figsize=(12, 12))
        plt.suptitle(f"Behavior States: {e} – {i}")
        ax0 = plt.subplot(4, 1, 1); ax0.axis('off')
        for s, u, st in u_ints:
            ax0.axvspan(t[s], t[u - 1] if u > 0 else t[0], color=cmap[st], alpha=0.8)
        ax0.set_xlim(0, t[-1]); ax0.set_title('Unified State')
        ax1 = plt.subplot(4, 1, 2); _plot_metric(ax1, m_sig, m_ints, 'Mantle Area',         'red')
        ax2 = plt.subplot(4, 1, 3); _plot_metric(ax2, o_sig, o_ints, 'Oral Siphon Width',   'blue')
        ax3 = plt.subplot(4, 1, 4); _plot_metric(ax3, a_sig, a_ints, 'Atrial Siphon Width', 'green')
        ax3.set_xlabel('Time (minutes)')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        base = f"Behavior_States_{e}_{i}"
        for ext in ['pdf', 'svg', 'png']:
            plt.savefig(os.path.join(out_plots, f"{base}.{ext}"))
        plt.close()

    if plot_all:
        for e, i in sorted(common_keys):
            _plot_one(e, i)

    return {
        'mantle_smoothed':  os.path.join(out_data,  'mantle_smoothed.csv'),
        'oral_smoothed':    os.path.join(out_data,  'oral_smoothed.csv'),
        'atrial_smoothed':  os.path.join(out_data,  'atrial_smoothed.csv'),
        'state_map':        os.path.join(out_plots, 'unified_behavior_states_map.csv'),
    }


def main():
    ZSCORED_DIR    = './output/zscored'   # ← edit
    OUT_DATA       = './output_data'      # ← edit
    OUT_PLOTS      = './output_plots'     # ← edit
    PROM           = 0.7
    DISTANCE       = 5
    WIDTH          = 5
    SMOOTH_WIN     = 20
    POLY_ORDER     = 1
    CAT_THRESH     = 0.6
    TOL            = 0.6
    SEC_PER_SAMPLE = 0.25
    PLOT_ALL       = False

    paths = process(
        zdir           = ZSCORED_DIR,
        prom           = PROM,
        dist           = DISTANCE,
        width          = WIDTH,
        smooth_win     = SMOOTH_WIN,
        poly           = POLY_ORDER,
        cat_thresh     = CAT_THRESH,
        tol            = TOL,
        sec_per_sample = SEC_PER_SAMPLE,
        out_data       = OUT_DATA,
        out_plots      = OUT_PLOTS,
        plot_all       = PLOT_ALL,
    )
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()