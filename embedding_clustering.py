"""
Embedding and clustering of static PCs.
"""
import os
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import DBSCAN
from openTSNE import TSNEEmbedding, affinity, initialization
from pyefd import reconstruct_contour

META = ['ExperimentID','Individuals','Condition','Timepoint']


def ensure_dir(p):
    os.makedirs(p, exist_ok=True); return p


def amp_from_efds(efd_df: pd.DataFrame, order: int, truncate: int):
    cols = [f'EFD_{i+1}' for i in range(4*order)]
    efd = efd_df.dropna(subset=cols).reset_index(drop=True)
    E = efd[cols].to_numpy(float)
    s = np.linalg.norm(E[:, :4], axis=1, keepdims=True)
    En = E / s
    En *= np.sign(En[:, [1]])  # mirror-align by B1
    amps = []
    for h in range(order):
        amps.append(np.linalg.norm(En[:, 4*h:4*h+4], axis=1))
    A = np.vstack(amps).T[:, :truncate]
    df = efd[['ExperimentID','Individuals','Condition','Timepoint']].copy()
    for j in range(A.shape[1]):
        df[f'amp_{j+1}'] = A[:, j]
    # keep aligned coeffs for medoids
    return df, A, En


def static_pcs(df_amp: pd.DataFrame, A: np.ndarray, n_pcs: int):
    Z = StandardScaler().fit_transform(A)
    p = PCA(n_components=n_pcs, random_state=0)
    S = p.fit_transform(Z)
    out = df_amp.copy(); out['row_id'] = np.arange(len(out))
    for i in range(n_pcs):
        out[f'sPC{i+1}'] = S[:, i]
    return out


def add_states(df_ts: pd.DataFrame, state_map_csv: str):
    st = (pd.read_csv(state_map_csv)
            .set_index(['ExperimentID','Individuals','Condition'])
            .stack().reset_index())
    st.columns = ['ExperimentID','Individuals','Condition','Timepoint_str','State']
    st['Timepoint'] = st['Timepoint_str'].str.lstrip('t').astype(int)
    return df_ts.merge(st[['ExperimentID','Individuals','Condition','Timepoint','State']],
                       on=['ExperimentID','Individuals','Condition','Timepoint'], how='inner')


def embed_cluster(df: pd.DataFrame, pc_cols, tag: str, out_dir: str,
                   perplexity: int, eps: float, min_samples: int):
    X = df[pc_cols].to_numpy(float)
    aff = affinity.PerplexityBasedNN(X, perplexity=perplexity, metric='euclidean', n_jobs=-1, random_state=42, method='exact')
    init = initialization.pca(X, random_state=42)
    emb = TSNEEmbedding(init, aff, negative_gradient_method='fft', n_jobs=-1)
    emb = emb.optimize(n_iter=250, exaggeration=30, momentum=0.5).optimize(n_iter=750, exaggeration=1, momentum=0.8)

    sub = ensure_dir(os.path.join(out_dir, tag))
    with open(os.path.join(sub, f'tsne_{tag}.pkl'), 'wb') as f:
        pickle.dump(np.asarray(emb), f, protocol=4)

    df[f'TSNE1_{tag}'], df[f'TSNE2_{tag}'] = emb[:,0], emb[:,1]

    db = DBSCAN(eps=eps, min_samples=min_samples)
    labels = db.fit_predict(np.asarray(emb))
    df[f'db_{tag}'] = labels

    labs = np.unique(labels)
    pal = sns.color_palette('tab20', max(1, len(labs)))
    lut = {lab: pal[i % len(pal)] for i, lab in enumerate(labs)}
    df[f'col_{tag}'] = [lut[l] for l in labels]
    return df, lut


def plot_by_state(df, tag, conds, state_colors, out_dir):
    sub = ensure_dir(os.path.join(out_dir, tag))
    fig, axs = plt.subplots(1, len(conds), figsize=(5*len(conds),5), sharex=True, sharey=True)
    axs = np.atleast_1d(axs)
    for ax, c in zip(axs, conds):
        d = df[df['Condition'] == c]
        for st, col in state_colors.items():
            m = d['State'] == st
            ax.scatter(d.loc[m, f'TSNE1_{tag}'], d.loc[m, f'TSNE2_{tag}'], s=10, alpha=0.7, c=[col], label=st, rasterized=True)
        ax.set_title(f'{tag} — {c}'); ax.set_xlabel('TSNE1'); ax.set_ylabel('TSNE2'); ax.grid(True, ls='--', alpha=0.3)
        ax.legend(frameon=False)
    fig.tight_layout()
    for ext in ('png','svg','pdf'):
        fig.savefig(os.path.join(sub, f'tsne_by_state_{tag}.{ext}'), dpi=300)
    plt.close(fig)


def plot_by_cluster(df, tag, conds, lut, out_dir):
    sub = ensure_dir(os.path.join(out_dir, tag))
    fig, axs = plt.subplots(1, len(conds), figsize=(5*len(conds),5), sharex=True, sharey=True)
    axs = np.atleast_1d(axs)
    for ax, c in zip(axs, conds):
        d = df[df['Condition'] == c]
        cols = [lut.get(l, (0.7,0.7,0.7)) for l in d[f'db_{tag}']]
        ax.scatter(d[f'TSNE1_{tag}'], d[f'TSNE2_{tag}'], s=10, alpha=0.8, c=cols, rasterized=True)
        ax.set_title(f'{tag} DBSCAN — {c}'); ax.set_xlabel('TSNE1'); ax.set_ylabel('TSNE2'); ax.grid(True, ls='--', alpha=0.3)
    fig.tight_layout()
    for ext in ('png','svg','pdf'):
        fig.savefig(os.path.join(sub, f'dbscan_by_condition_{tag}.{ext}'), dpi=300)
    plt.close(fig)


def reconstruct_shape(flat_coeffs, n=300):
    return reconstruct_contour(flat_coeffs.reshape(-1,4), locus=(0,0), num_points=n)


def medoids_and_shapes(df_emb, tag, pcs_cols, En, out_dir, lut):
    sub = ensure_dir(os.path.join(out_dir, tag))
    labels = df_emb[f'db_{tag}'].values
    labs = [l for l in np.unique(labels) if l != -1]
    Xpc = df_emb[pcs_cols].to_numpy(float)
    idxs = {}
    for lbl in labs:
        m = labels == lbl
        Xc = Xpc[m]
        cen = Xc.mean(axis=0)
        pos = np.argmin(np.linalg.norm(Xc - cen, axis=1))
        idxs[lbl] = np.where(m)[0][pos]

    fig, ax = plt.subplots(1, max(1,len(labs)), figsize=(2.5*max(1,len(labs)), 2.5), squeeze=False, dpi=200)
    ax = ax[0]
    for k, lbl in enumerate(labs):
        c = reconstruct_shape(En[df_emb.iloc[idxs[lbl]]['row_id']])
        if not np.allclose(c[0], c[-1]):
            c = np.vstack([c, c[0]])
        a = ax[k] if len(labs)>1 else ax
        color = lut.get(lbl, (0.7,0.7,0.7))
        a.fill(c[:,0], c[:,1], facecolor=color, alpha=0.3, edgecolor=None)
        a.plot(c[:,0], c[:,1], lw=1.5, color=color)
        a.set_title(f'cluster {lbl}', fontsize=8); a.axis('equal'); a.axis('off')
    fig.tight_layout()
    for ext in ('png','svg','pdf'):
        fig.savefig(os.path.join(sub, f'medoid_shapes_dbscan.{ext}'), dpi=300)
    plt.close(fig)

    rows = []
    for lbl, idx in idxs.items():
        r = df_emb.iloc[idx].copy().to_dict(); r['cluster_label'] = int(lbl); rows.append(r)
    pd.DataFrame(rows).to_csv(os.path.join(sub, 'medoid_summary_dbscan.csv'), index=False)


def main():
    import argparse
    p = argparse.ArgumentParser(description='Static embedding + clustering + plots')
    p.add_argument('--efd', required=True)
    p.add_argument('--state-map', required=True)
    p.add_argument('--order', type=int, default=30)
    p.add_argument('--truncate', type=int, default=10)
    p.add_argument('--n-static-pcs', type=int, default=5)
    p.add_argument('--perplexity', type=int, default=150)
    p.add_argument('--eps', type=float, default=6)
    p.add_argument('--min-samples', type=int, default=50)
    p.add_argument('--out', default='./output_embeddings')
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)

    efd = pd.read_csv(args.efd)
    df_amp, A, En = amp_from_efds(efd, order=args.order, truncate=args.truncate)
    df_static = static_pcs(df_amp, A, n_pcs=args.n_static_pcs)
    df_static = add_states(df_static, args.state_map)

    conds = sorted(df_static['Condition'].dropna().unique().tolist())
    tag = 'static'
    pc_cols = [f'sPC{i+1}' for i in range(args.n_static_pcs)]

    df_emb, lut = embed_cluster(df_static.copy(), pc_cols, tag, args.out, args.perplexity, args.eps, args.min_samples)

    # colors by behavioral state (gnuplot2 stops)
    state_colors = {
        'Quiescence': plt.cm.gnuplot2(0.9),
        'Half Contraction': plt.cm.gnuplot2(0.6),
        'Full Contraction': plt.cm.gnuplot2(0.3),
    }
    plot_by_state(df_emb, tag, conds, state_colors, args.out)
    plot_by_cluster(df_emb, tag, conds, lut, args.out)

    medoids_and_shapes(df_emb, tag, pc_cols, En, args.out, lut)

    df_static.to_csv(os.path.join(args.out, 'static_pcs.csv'), index=False)
    df_emb.to_csv(os.path.join(args.out, 'static_tsne_dbscan.csv'), index=False)
    print('Saved to', args.out)


if __name__ == '__main__':
    main()

