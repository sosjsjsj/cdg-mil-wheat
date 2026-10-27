"""Band occlusion: drop in test R² when a sliding window of bands is replaced
by the training-set mean input (windows of 2, 4 and 10 bands).

    python code_xai/spectral.py --checkpoint results/models/model_rep1_Protein_msc_KAN.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code_training'))
import config
import data
import models

N_BANDS = config.N_BANDS
WL      = np.linspace(400, 1000, N_BANDS)
WINDOWS = {10: 2, 20: 4, 50: 10}          # nm -> bands
THRESH  = 0.5

TARGET_COLORS = {'Protein':      ('#1a7a2e', '#a8d5b5'),
                 'Moisture':     ('#1a4f9e', '#a3c0f0'),
                 'Carbohydrate': ('#b8860b', '#fde98a')}
RED_LINE, RED_FILL, FILL_A = '#cc0000', '#f4a0a0', 0.35


def load_inputs(ckpt, target):
    """Test inputs, test labels and the mean training input."""
    tr, _, _, te = data.split_bags(data.build_bag_index())
    ref  = ckpt.get('msc_ref')
    if ref is None and ckpt['prep'] == 'msc':
        ref = data.compute_msc_ref(tr)
    kw = dict(spec_mean=ckpt['spec_mean'], spec_std=ckpt['spec_std'],
              y_mean=ckpt['y_mean'], y_std=ckpt['y_std'],
              method=ckpt['prep'], msc_ref=ref, is_train=False, use_aug=False)
    dte = data.BagDataset1D(te, target, **kw)
    dtr = data.BagDataset1D(tr, target, **kw)
    xs  = [dte[i][0] for i in range(len(dte))]
    y   = np.array([float(dte[i][1]) for i in range(len(dte))]) * ckpt['y_std'] + ckpt['y_mean']
    mu  = torch.cat([dtr[i][0] for i in range(len(dtr))]).mean(0)
    return xs, y, mu


def r2(model, xs, y, ckpt):
    with torch.no_grad():
        p = np.array([float(model(x)) for x in xs]) * ckpt['y_std'] + ckpt['y_mean']
    return r2_score(y, p)


def occlusion(model, xs, y, mu, ckpt, w):
    """Drop in test R² for each window of w bands, shifted by one band."""
    base   = r2(model, xs, y, ckpt)
    starts = np.arange(0, N_BANDS - w + 1)
    drops  = []
    for s in starts:
        occ = []
        for x in xs:
            z = x.clone()
            z[:, s:s + w] = mu[s:s + w]
            occ.append(z)
        drops.append(base - r2(model, occ, y, ckpt))
    centers = np.array([WL[s:s + w].mean() for s in starts])
    return starts, centers, np.array(drops), base


def premix(color, alpha):
    return tuple(np.array(mcolors.to_rgb(color)) * alpha + (1 - alpha))


def crossings(x, y, th):
    xs, ys = [x[0]], [y[0]]
    for i in range(1, len(y)):
        if (y[i - 1] < th) != (y[i] < th):
            f = (th - y[i - 1]) / (y[i] - y[i - 1])
            xs.append(x[i - 1] + f * (x[i] - x[i - 1]))
            ys.append(th)
        xs.append(x[i])
        ys.append(y[i])
    return np.array(xs), np.array(ys)


def plot_importance(ax, x, sal, target, legend=True, thresh=THRESH, labels=None):
    """Normalised ΔR²; values at or above thresh are drawn in red."""
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    line_c, fill_c = TARGET_COLORS[target]
    fill_pm, red_pm = premix(fill_c, FILL_A), premix(RED_FILL, FILL_A)
    xx, yy = crossings(x, sal, thresh)
    above = yy >= thresh
    ax.fill_between(xx, yy, 0, color=fill_pm, linewidth=0)
    ax.fill_between(xx, yy, 0, where=above, color=red_pm, linewidth=0)
    ax.plot(xx, yy, color=line_c, lw=1.3, zorder=2)
    ax.plot(xx, np.where(above, yy, np.nan), color=RED_LINE, lw=1.3, zorder=3)
    ax.axhline(thresh, color='gray', lw=0.8, linestyle='--', alpha=0.6)
    ax.set_xlim(400, 1000)
    ax.set_ylim(0, 1.05)
    ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
    ax.tick_params(labelsize=14)
    ax.set_xlabel('Wavelength (nm)', fontsize=16)
    ax.set_ylabel('Normalised ΔR²', fontsize=16)
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    if legend:
        low, high = labels or (f'Normalised ΔR² < {thresh:g}', f'Normalised ΔR² ≥ {thresh:g}')
        handles = [mpatches.Patch(facecolor=fill_pm, edgecolor=line_c, linewidth=1.2, label=low),
                   mpatches.Patch(facecolor=red_pm, edgecolor=RED_LINE, linewidth=1.2, label=high)]
        ax.legend(handles=handles, loc='lower right', bbox_to_anchor=(1.0, 1.0), fontsize=12,
                  frameon=True, edgecolor='gray')


def main():
    ap = argparse.ArgumentParser(description='CDG-MIL spectral XAI (band occlusion)')
    ap.add_argument('--checkpoint', required=True, help='trained 1-D model (.pt)')
    ap.add_argument('--top', type=float, default=10,
                    help='mark the top N %% of windows as principal bands (default 10); '
                         '0 marks normalised ΔR² >= 0.5 instead')
    ap.add_argument('--dpi', type=int, default=300)
    args = ap.parse_args()

    torch.set_num_threads(4)
    ckpt     = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    target   = ckpt['target']
    backbone = ckpt['backbone']
    if backbone == 'CNN3D':
        raise SystemExit('Band occlusion runs on 1-D backbones; use spatial.py for CNN3D.')

    model = models.build_model(backbone, models.get_head())
    model.load_state_dict(ckpt['state_dict'])
    model.eval()

    out = config.XAI_DIR / 'spectral' / target
    out.mkdir(parents=True, exist_ok=True)
    xs, y, mu = load_inputs(ckpt, target)

    curves = {}
    for nm, w in WINDOWS.items():
        starts, centers, drops, base = occlusion(model, xs, y, mu, ckpt, w)
        sal = (drops - drops.min()) / (drops.max() - drops.min() + 1e-12)
        curves[nm] = (centers, sal)
        pd.DataFrame({'start_band': starts, 'end_band': starts + w - 1,
                      'center_nm': centers.round(2), 'dR2': drops.round(5),
                      'saliency_minmax': sal.round(4)}).to_csv(
            out / f'occlusion_{nm}nm_sliding.csv', index=False, encoding='utf-8-sig')

        cut = {}
        if args.top:
            th  = float(np.quantile(sal, 1 - args.top / 100))
            cut = dict(thresh=th, labels=(f'Bottom {100 - args.top:g}%', f'Top {args.top:g}%'))
        fig, ax = plt.subplots(figsize=(12, 3.6))
        plot_importance(ax, centers, sal, target, **cut)
        plt.tight_layout()
        plt.savefig(out / f'occlusion_{nm}nm_sliding.png', dpi=args.dpi)
        plt.close()
        print(f'[{target}] {nm:>2} nm window: baseline R² {base:.4f}, '
              f'peak {centers[drops.argmax()]:.0f} nm (ΔR² {drops.max():.4f})', flush=True)

    fig, axes = plt.subplots(1, len(WINDOWS), figsize=(18, 3.6))
    for ax, (nm, (centers, sal)) in zip(axes, curves.items()):
        cut = {}
        if args.top:
            cut = dict(thresh=float(np.quantile(sal, 1 - args.top / 100)))
        plot_importance(ax, centers, sal, target, legend=False, **cut)
        ax.set_title(f'{target} | {nm} nm window', fontsize=13)
    plt.tight_layout()
    plt.savefig(out / 'compare_window_widths.png', dpi=args.dpi)
    plt.close()
    print(f'written to {out}')


if __name__ == '__main__':
    main()
