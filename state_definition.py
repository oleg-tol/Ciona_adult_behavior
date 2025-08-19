"""
State detection from z-scored shape metrics.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter, find_peaks
from typing import Dict, List, Tuple

COMMON = ['ExperimentID', 'Individuals', 'Condition']


def _odd(n: int) -> int:
    return n if n % 2 == 1 else n + 1


def smooth_signal(x: np.ndarray, window: int, poly: int) -> np.ndarray:
    w = _odd(max(3, window))
    pad = w
    ext = np.pad(x, pad, mode='edge')
    y = savgol_filter(ext, w, poly, mode='nearest')
    return y[pad:-pad]


def mark_first_last_valid_points(x: np.ndarray) -> np.ndarray:
    y = x.copy()
    idx = np.where(~np.isnan(y))[0]
    if idx.size:
        y[idx[0]] = -1
        y[idx[-1]] = -1
    return y


def offset_signal(x: np.ndarray) -> np.ndarray:
    m = np.nanmin(x)
    return x - m if m < 0 else x + 1


def find_contraction_peaks_and_binary(x: np.ndarray, prom: float, dist: int, width: int, T: int,
                                      smooth_win: int, poly: int, apply_offset: bool = True):
    s = pd.Series(x).interpolate(method='linear', limit_direction='both').values
    if apply_offset:
        s = offset_signal(s)
    s = smooth_signal(s, smooth_win, poly)
    peaks, props = find_peaks(s, prominence=prom, distance=dist, width=width)
    lefts = np.floor(props['left_ips']).astype(int)
    rights = np.ceil(props['right_ips']).astype(int)
    proms = props['prominences']
    binar = np.zeros(T)
    for a, b in zip(lefts, rights):
        a = max(a, 0); b = min(b, T - 1)
        binar[a:b + 1] = 1
    return peaks, lefts, rights, proms, binar, s


def categorize_events(df: pd.DataFrame, thresh: float) -> pd.DataFrame:
    out = []
    for (e, i, m), g in df.groupby(['ExperimentID', 'Individuals', 'Metric'], sort=False):
        mh = g['Peak_value'].max()
        gg = g.copy()
        gg['Category'] = np.where(gg['Peak_value'] >= thresh * mh, 'full', 'half')
        if 'Duration' not in gg:
            gg['Duration'] = gg['End_index'].astype(int) - gg['Start_index'].astype(int) + 1
        out.append(gg)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=df.columns.tolist() + ['Category', 'Duration'])


def _time_cols(df: pd.DataFrame) -> List[str]:
    return [c for c in df.columns if c not in COMMON]


def _load_three(zdir: str) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    m = pd.read_csv(os.path.join(zdir, 'mantle_areas_zscored.csv'))
    o = pd.read_csv(os.path.join(zdir, 'oral_siphon_widths_zscored.csv'))
    a = pd.read_csv(os.path.join(zdir, 'atrial_siphon_widths_zscored.csv'))
    return m, o, a

#main
def process(zdir: str, prom: float, dist: int, width: int, smooth_win: int, poly: int,
            cat_thresh: float, tol: float, sec_per_sample: float,
            out_data: str, out_plots: str, plot_all: bool) -> Dict[str, str]:
    os.makedirs(out_data, exist_ok=True)
    os.makedirs(out_plots, exist_ok=True)

    mantle, oral, atrial = _load_three(zdir)

    keys_m = set(zip(mantle['ExperimentID'], mantle['Individuals']))
    keys_o = set(zip(oral['ExperimentID'], oral['Individuals']))
    keys_a = set(zip(atrial['ExperimentID'], atrial['Individuals']))
    keys = keys_m & keys_o & keys_a

    mantle_s, oral_s, atrial_s = [], [], []
    mantle_ev, oral_ev, atrial_ev = [], [], []

    for exp, ind in sorted(keys):
        mr = mantle[(mantle['ExperimentID'] == exp) & (mantle['Individuals'] == ind)]
        orr = oral[(oral['ExperimentID'] == exp) & (oral['Individuals'] == ind)]
        ar = atrial[(atrial['ExperimentID'] == exp) & (atrial['Individuals'] == ind)]
        if mr.empty or orr.empty or ar.empty:
            continue
        cond = mr['Condition'].iloc[0] if 'Condition' in mr.columns else None

        m_cols = _time_cols(mr); o_cols = _time_cols(orr); a_cols = _time_cols(ar)
        t_common = sorted(set(m_cols) | set(o_cols) | set(a_cols), key=lambda x: float(x))
        T = len(t_common)

        raw = {
            'Mantle Area': pd.to_numeric(mr.iloc[0][m_cols], errors='coerce'),
            'Oral Siphon Width': pd.to_numeric(orr.iloc[0][o_cols], errors='coerce'),
            'Atrial Siphon Width': pd.to_numeric(ar.iloc[0][a_cols], errors='coerce'),
        }

        aligned = {}
        for k, s in raw.items():
            s = s.reindex(t_common).interpolate(method='linear', limit_direction='forward', limit_area='inside').bfill()
            aligned[k] = s.values

        proc = {}
        for metric, arr in aligned.items():
            sm = smooth_signal(arr, smooth_win, poly)
            pm = mark_first_last_valid_points(sm)
            peaks, lefts, rights, proms, binar, used = find_contraction_peaks_and_binary(
                pm, prom, dist, width, T, smooth_win, poly
            )
            proc[metric] = used
            for p, l, r, pr in zip(peaks, lefts, rights, proms):
                ev = {
                    'ExperimentID': exp, 'Individuals': ind, 'Condition': cond,
                    'Start': t_common[l], 'Peak': t_common[p], 'End': t_common[r],
                    'Start_index': int(l), 'Peak_index': int(p), 'End_index': int(r),
                    'Prominence': float(pr), 'Peak_value': float(used[p]), 'Metric': metric,
                }
                (mantle_ev if metric == 'Mantle Area' else oral_ev if metric == 'Oral Siphon Width' else atrial_ev).append(ev)

        mantle_s.append({'ExperimentID': exp, 'Individuals': ind, 'Condition': cond, **dict(zip(t_common, proc['Mantle Area']))})
        oral_s.append({'ExperimentID': exp, 'Individuals': ind, 'Condition': cond, **dict(zip(t_common, proc['Oral Siphon Width']))})
        atrial_s.append({'ExperimentID': exp, 'Individuals': ind, 'Condition': cond, **dict(zip(t_common, proc['Atrial Siphon Width']))})

    mantle_sm = pd.DataFrame(mantle_s)
    oral_sm = pd.DataFrame(oral_s)
    atrial_sm = pd.DataFrame(atrial_s)
    mantle_ev_df = pd.DataFrame(mantle_ev)
    oral_ev_df = pd.DataFrame(oral_ev)
    atrial_ev_df = pd.DataFrame(atrial_ev)

    mantle_sm.to_csv(os.path.join(out_data, 'mantle_smoothed.csv'), index=False)
    oral_sm.to_csv(os.path.join(out_data, 'oral_smoothed.csv'), index=False)
    atrial_sm.to_csv(os.path.join(out_data, 'atrial_smoothed.csv'), index=False)
    mantle_ev_df.to_csv(os.path.join(out_data, 'mantle_contraction_events.csv'), index=False)
    oral_ev_df.to_csv(os.path.join(out_data, 'oral_contraction_events.csv'), index=False)
    atrial_ev_df.to_csv(os.path.join(out_data, 'atrial_contraction_events.csv'), index=False)

    all_events = pd.concat([
        mantle_ev_df.assign(Metric='Mantle Area'),
        oral_ev_df.assign(Metric='Oral Siphon Width'),
        atrial_ev_df.assign(Metric='Atrial Siphon Width'),
    ], ignore_index=True)
    categorized = categorize_events(all_events, cat_thresh)

    def _features(df: pd.DataFrame) -> pd.DataFrame:
        feats = []
        for (e, i, m), g in df.groupby(['ExperimentID', 'Individuals', 'Metric']):
            peaks_sec = g['Peak'].astype(float).values * sec_per_sample
            avg_ip = np.mean(np.diff(peaks_sec)) if len(peaks_sec) > 1 else np.nan
            avg_dur = g['Duration'].mean() * sec_per_sample
            # total time from max end index
            tot_sec = (g['End_index'].max() + 1) * sec_per_sample
            freq = (len(g) / (tot_sec / 60.0)) * 10 if tot_sec > 0 else np.nan
            cond = g['Condition'].iloc[0] if 'Condition' in g.columns else None
            feats.append({'ExperimentID': e, 'Individuals': i, 'Condition': cond, 'Metric': m,
                          'Avg_IP': avg_ip, 'Avg_Duration': avg_dur, 'Frequency_per_10mins': freq})
        return pd.DataFrame(feats)

    feats_df = _features(categorized)
    feats_df.to_csv(os.path.join(out_data, 'contraction_event_features.csv'), index=False)

    # quiescence
    def _binary(sig: np.ndarray) -> np.ndarray:
        _, _, _, _, b, _ = find_contraction_peaks_and_binary(sig, prom, dist, width, len(sig), smooth_win, poly)
        return b

    q_events, q_stats = [], []
    for _, row in mantle_sm.iterrows():
        e, i = row['ExperimentID'], row['Individuals']
        cond = row.get('Condition', None)
        m_sig = pd.to_numeric(row[_time_cols(mantle_sm)], errors='coerce').dropna().values
        orow = oral_sm[(oral_sm['ExperimentID'] == e) & (oral_sm['Individuals'] == i)]
        arow = atrial_sm[(atrial_sm['ExperimentID'] == e) & (atrial_sm['Individuals'] == i)]
        if orow.empty or arow.empty:
            continue
        o_sig = pd.to_numeric(orow.iloc[0][_time_cols(oral_sm)], errors='coerce').dropna().values
        a_sig = pd.to_numeric(arow.iloc[0][_time_cols(atrial_sm)], errors='coerce').dropna().values

        m_thr, o_thr, a_thr = np.nanmedian(m_sig), np.nanmedian(o_sig), np.nanmedian(a_sig)
        m_bin, o_bin, a_bin = _binary(m_sig), _binary(o_sig), _binary(a_sig)

        def _find_quiescence(binarr: np.ndarray, sig: np.ndarray, thr: float) -> List[Tuple[int,int]]:
            ev = []
            start = None
            for k in range(len(sig)):
                if binarr[k] == 0 and sig[k] < thr + tol and start is None:
                    start = k
                elif (binarr[k] == 1 or sig[k] >= thr + tol) and start is not None:
                    ev.append((start, k - 1)); start = None
            if start is not None:
                ev.append((start, len(sig) - 1))
            return ev

        metrics = {
            'Mantle Area': (m_sig, m_thr, m_bin),
            'Oral Siphon Width': (o_sig, o_thr, o_bin),
            'Atrial Siphon Width': (a_sig, a_thr, a_bin),
        }

        for name, (sig, thr, binarr) in metrics.items():
            evs = _find_quiescence(binarr, sig, thr)
            for s, t in evs:
                q_events.append({'ExperimentID': e, 'Individuals': i, 'Condition': cond, 'Metric': name,
                                 'Start': s, 'End': t, 'Duration_samples': t - s + 1,
                                 'Duration_seconds': (t - s + 1) * sec_per_sample, 'SignalThreshold': thr})
            total_sec = len(sig) * sec_per_sample
            total_min = total_sec / 60.0
            if evs:
                durs = [t - s + 1 for s, t in evs]
                avg_dur = float(np.mean(durs) * sec_per_sample)
                if len(evs) > 1:
                    ipi = float(np.mean([evs[k][0] - evs[k-1][1] for k in range(1, len(evs))]) * sec_per_sample)
                else:
                    ipi = np.nan
                freq = (len(evs) / total_min) * 10 if total_min > 0 else np.nan
                cnt = len(evs)
            else:
                avg_dur = 0.0; ipi = np.nan; freq = 0.0; cnt = 0
            q_stats.append({'ExperimentID': e, 'Individuals': i, 'Condition': cond, 'Metric': name,
                            'Frequency_per_10min': freq, 'Avg_IPI': ipi, 'Avg_Duration': avg_dur, 'Event_Count': cnt})

    q_events_df = pd.DataFrame(q_events)
    q_stats_df = pd.DataFrame(q_stats)

    q_events_df.to_csv(os.path.join(out_data, 'quiescence_events_detail.csv'), index=False)
    q_stats_df.to_csv(os.path.join(out_data, 'quiescence_aggregate_stats.csv'), index=False)

    # unified states
    cmap = {
        'Full Contraction': plt.cm.gnuplot2(0.3),
        'Half Contraction': plt.cm.gnuplot2(0.6),
        'Quiescence': plt.cm.gnuplot2(0.9),
    }

    def _get_merged_for_metric(e: str, i: str, metric: str, n: int):
        src = categorized[categorized['ExperimentID'].eq(e) & categorized['Individuals'].eq(i) & categorized['Metric'].eq(metric)]
        evs = []
        for _, r in src.iterrows():
            s, t, cat = int(r['Start_index']), int(r['End_index']) + 1, r['Category'].strip().lower()
            evs.append((s, t, cat))
        evs.sort(key=lambda z: z[0])
        bounds = {0, n}
        for s, t, _ in evs:
            bounds.add(s); bounds.add(t)
        bounds = sorted(bounds)
        spans = []
        for a, b in zip(bounds[:-1], bounds[1:]):
            ov = [ev for ev in evs if not (ev[1] <= a or ev[0] >= b)]
            if not ov:
                st = 'Quiescence'
            elif all(z[2] == 'full' for z in ov):
                st = 'Full Contraction'
            else:
                st = 'Half Contraction'
            spans.append((a, b, st))
        merged = []
        if spans:
            cs, ce, cst = spans[0]
            for a, b, st in spans[1:]:
                if st == cst:
                    ce = b
                else:
                    merged.append((cs, ce, cst)); cs, ce, cst = a, b, st
            merged.append((cs, ce, cst))
        return merged

    def _truncate(ints, n: int):
        out = []
        for s, t, st in ints:
            if t <= 0 or s >= n:
                continue
            a, b = max(0, s), min(n, t)
            if a < b:
                out.append((a, b, st))
        return out

    def _unify(e: str, i: str, n: int):
        m = _truncate(_get_merged_for_metric(e, i, 'Mantle Area', n), n)
        o = _truncate(_get_merged_for_metric(e, i, 'Oral Siphon Width', n), n)
        a = _truncate(_get_merged_for_metric(e, i, 'Atrial Siphon Width', n), n)
        seq = []
        for t in range(n):
            sm = 'Full Contraction' if any(s <= t < u and st == 'full' for s, u, st in m) else 'Quiescence'
            so = 'Full Contraction' if any(s <= t < u and st == 'full' for s, u, st in o) else 'Quiescence'
            sa = 'Full Contraction' if any(s <= t < u and st == 'full' for s, u, st in a) else 'Quiescence'
            if sm == so == sa == 'Full Contraction':
                seq.append('Full Contraction')
            elif any(z != 'Quiescence' for z in [sm, so, sa]):
                seq.append('Half Contraction')
            else:
                seq.append('Quiescence')
        merged = []
        if seq:
            cs, st = 0, seq[0]
            for t in range(1, n):
                if seq[t] != st:
                    merged.append((cs, t, st)); cs, st = t, seq[t]
            merged.append((cs, n, st))
        return merged

    # plots and map
    def _plot_one(e: str, i: str):
        mrow = mantle_sm[(mantle_sm['ExperimentID'] == e) & (mantle_sm['Individuals'] == i)]
        orow = oral_sm[(oral_sm['ExperimentID'] == e) & (oral_sm['Individuals'] == i)]
        arow = atrial_sm[(atrial_sm['ExperimentID'] == e) & (atrial_sm['Individuals'] == i)]
        if mrow.empty or orow.empty or arow.empty:
            return
        m_sig = pd.to_numeric(mrow.iloc[0][_time_cols(mantle_sm)], errors='coerce').dropna().values + 2
        o_sig = pd.to_numeric(orow.iloc[0][_time_cols(oral_sm)], errors='coerce').dropna().values + 2
        a_sig = pd.to_numeric(arow.iloc[0][_time_cols(atrial_sm)], errors='coerce').dropna().values + 2
        n = min(len(m_sig), len(o_sig), len(a_sig))
        m_sig, o_sig, a_sig = m_sig[:n], o_sig[:n], a_sig[:n]
        t = np.arange(n) * (sec_per_sample / 60.0)
        m_ints = _truncate(_get_merged_for_metric(e, i, 'Mantle Area', n), n)
        o_ints = _truncate(_get_merged_for_metric(e, i, 'Oral Siphon Width', n), n)
        a_ints = _truncate(_get_merged_for_metric(e, i, 'Atrial Siphon Width', n), n)
        u_ints = _unify(e, i, n)

        def _plot_metric(ax, sig, ints, title, color):
            ax.plot(t, sig, color=color)
            for s, u, st in ints:
                ax.fill_between(t[s:u], 0, sig[s:u], color=cmap[{'full':'Full Contraction','half':'Half Contraction'}.get(st, st)], alpha=0.6)
                ax.axvline(t[s], color='k', linestyle=':', lw=1)
                ax.axvline(t[u-1], color='k', linestyle=':', lw=1)
            ax.set_xlim(0, t[-1])
            ax.set_ylabel('Z-score')
            ax.set_title(title)

        plt.figure(figsize=(12, 12))
        plt.suptitle(f"Behavior States: {e} – {i}")
        ax0 = plt.subplot(4, 1, 1); ax0.axis('off')
        for s, u, st in u_ints:
            ax0.axvspan(t[s], t[u-1] if u>0 else t[0], color=cmap[st], alpha=0.8)
        ax0.set_xlim(0, t[-1]); ax0.set_title('Unified State')
        ax1 = plt.subplot(4, 1, 2); _plot_metric(ax1, m_sig, m_ints, 'Mantle Area', 'red')
        ax2 = plt.subplot(4, 1, 3); _plot_metric(ax2, o_sig, o_ints, 'Oral Siphon Width', 'blue')
        ax3 = plt.subplot(4, 1, 4); _plot_metric(ax3, a_sig, a_ints, 'Atrial Siphon Width', 'green')
        ax3.set_xlabel('Time (minutes)')
        plt.tight_layout(rect=[0, 0, 1, 0.95])
        base = f"Behavior_States_{e}_{i}"
        for ext in ['pdf', 'svg', 'png']:
            plt.savefig(os.path.join(out_plots, f"{base}.{ext}"))
        plt.close()

    # state map
    rows = []
    for e, i in keys:
        mr = mantle_sm[(mantle_sm['ExperimentID'] == e) & (mantle_sm['Individuals'] == i)]
        orr = oral_sm[(oral_sm['ExperimentID'] == e) & (oral_sm['Individuals'] == i)]
        ar = atrial_sm[(atrial_sm['ExperimentID'] == e) & (atrial_sm['Individuals'] == i)]
        if mr.empty or orr.empty or ar.empty:
            continue
        n = min(len(_time_cols(mr)), len(_time_cols(orr)), len(_time_cols(ar)))
        ints = _unify(e, i, n)
        series = ['Quiescence'] * n
        for s, u, st in ints:
            for t in range(s, min(u, n)):
                series[t] = st
        cond = mr['Condition'].iloc[0] if 'Condition' in mr.columns else None
        row = {'ExperimentID': e, 'Individuals': i, 'Condition': cond}
        row.update({f't{t}': series[t] for t in range(n)})
        rows.append(row)
        if plot_all:
            _plot_one(e, i)

    state_map = pd.DataFrame(rows)
    state_map.to_csv(os.path.join(out_plots, 'unified_behavior_states_map.csv'), index=False)

    # categorized outputs per metric
    man_out = categorized[categorized['Metric'] == 'Mantle Area'].drop(columns=['Metric'])
    ora_out = categorized[categorized['Metric'] == 'Oral Siphon Width'].drop(columns=['Metric'])
    atr_out = categorized[categorized['Metric'] == 'Atrial Siphon Width'].drop(columns=['Metric'])
    man_out.to_csv(os.path.join(out_data, 'mantle_events_with_duration.csv'), index=False)
    ora_out.to_csv(os.path.join(out_data, 'oral_events_with_duration.csv'), index=False)
    atr_out.to_csv(os.path.join(out_data, 'atrial_events_with_duration.csv'), index=False)

    return {
        'mantle_smoothed': os.path.join(out_data, 'mantle_smoothed.csv'),
        'oral_smoothed': os.path.join(out_data, 'oral_smoothed.csv'),
        'atrial_smoothed': os.path.join(out_data, 'atrial_smoothed.csv'),
        'state_map': os.path.join(out_plots, 'unified_behavior_states_map.csv'),
    }


def main() -> None:
    import argparse
    p = argparse.ArgumentParser(description='State detection from z-scored metrics')
    p.add_argument('--zscored-dir', required=True)
    p.add_argument('--prom', type=float, default=0.7)
    p.add_argument('--distance', type=int, default=5)
    p.add_argument('--width', type=int, default=5)
    p.add_argument('--smooth-win', type=int, default=20)
    p.add_argument('--poly-order', type=int, default=1)
    p.add_argument('--cat-thresh', type=float, default=0.6)
    p.add_argument('--tol', type=float, default=0.6)
    p.add_argument('--sec-per-sample', type=float, default=0.25)
    p.add_argument('--out-data', default='./output_data')
    p.add_argument('--out-plots', default='./output_plots')
    p.add_argument('--plot-all', action='store_true')
    args = p.parse_args()

    paths = process(
        zdir=args.zscored_dir,
        prom=args.prom,
        dist=args.distance,
        width=args.width,
        smooth_win=args.smooth_win,
        poly=args.poly_order,
        cat_thresh=args.cat_thresh,
        tol=args.tol,
        sec_per_sample=args.sec_per_sample,
        out_data=args.out_data,
        out_plots=args.out_plots,
        plot_all=args.plot_all,
    )
    for k, v in paths.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()

