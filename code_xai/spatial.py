"""3D Grad-CAM of the CNN3D model for the three highest-attention kernels of
the highest and lowest sample per treatment. Requires dataset/hyperspectral_wheat_3d/.

    python code_xai/spatial.py --checkpoint results/models/model_rep1_Protein_snv_CNN3D.pt
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'code_training'))
import config
import data
import models

DEVICE = torch.device('cpu')
TOP_K  = 3
CMAP   = plt.get_cmap('RdYlGn_r')

def _normalize(arr):
    mn, mx = arr.min(), arr.max()
    return (arr - mn) / (mx - mn + 1e-8)


def _resize2d(arr, shape=(50, 50)):
    img = Image.fromarray((_normalize(arr) * 255).astype(np.uint8))
    img = img.resize((shape[1], shape[0]), Image.BILINEAR)
    return np.array(img).astype(np.float32) / 255.0


def _resize1d(arr, length):
    return np.interp(np.linspace(0, len(arr) - 1, length), np.arange(len(arr)), arr)


class GradCAM3D:
    """Grad-CAM on the last Conv3d, back-propagated from the bag-level prediction."""
    def __init__(self, model):
        self.model, self._feats = model, []
        layer = None
        for m in model.backbone.cnn.modules():
            if isinstance(m, nn.Conv3d):
                layer = m
        if layer is None:
            raise ValueError('Conv3d not found in model.backbone.cnn')
        layer.register_forward_hook(self._keep)

    def _keep(self, mod, inp, out):
        if torch.is_grad_enabled():
            out.retain_grad()
            self._feats.append(out)

    def compute_bag(self, bag, kernels):
        """bag: (N, 50, 50, bands) -> {kernel index: {'spatial', 'spectral'}}"""
        self.model.eval()
        self._feats = []
        pred = self.model(bag.to(DEVICE)).squeeze()
        self.model.zero_grad()
        pred.backward()
        feat = torch.cat(self._feats, dim=0)
        grad = torch.cat([f.grad for f in self._feats], dim=0)
        if feat.shape[0] != bag.shape[0]:
            raise RuntimeError(f'captured {feat.shape[0]} kernel maps for a bag of {bag.shape[0]}')
        n_bands = bag.shape[-1]
        out = {}
        for k in kernels:
            weights = grad[k].mean(dim=(1, 2, 3), keepdim=True)
            cam = torch.abs((weights * feat[k]).sum(dim=0)).detach().cpu().numpy()
            out[k] = {'spatial':  _normalize(_resize2d(cam.mean(axis=0), (50, 50))),
                      'spectral': _normalize(_resize1d(cam.mean(axis=(1, 2)), n_bands))}
        self._feats = []
        return out


def save_colorbar(path):
    fig, ax = plt.subplots(figsize=(1.0, 4.0))
    cb = fig.colorbar(plt.cm.ScalarMappable(cmap=CMAP), cax=ax, ticks=[0, 0.5, 1])
    cb.ax.set_yticklabels(['Low', 'Mid', 'High'])
    cb.set_label('GradCAM Activation')
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()


def select_bags(df_te, target):
    """Highest and lowest sample of the target within each treatment."""
    picks = []
    for code in config.TREATMENTS:
        s = df_te[df_te.treatment == code]
        for label, row in [('High', s.loc[s[target].idxmax()]), ('Low', s.loc[s[target].idxmin()])]:
            picks.append((f'{label}_{row.genotype}_{code}', row.key))
    return picks


def explain_bag(model, gcam, df_bag, target, kw, out, label):
    ds = data.BagDataset3D(df_bag, target, is_train=False, use_aug=False, **kw)
    X, _, _ = next(iter(DataLoader(ds, batch_size=1, collate_fn=data.mil_collate_fn)))
    n = X.shape[0]
    attn    = model.get_attention_weights(X.to(DEVICE)).cpu().numpy().reshape(-1)
    top_idx = np.argsort(attn)[::-1][:min(TOP_K, n)]
    fpaths  = list(df_bag.iloc[0].fpath_list)
    maps    = gcam.compute_bag(X, top_idx)

    out.mkdir(parents=True, exist_ok=True)
    spectra = []
    for rank, ki in enumerate(top_idx, start=1):
        r  = maps[ki]
        bg = np.load(fpaths[ki]).astype(np.float32).sum(axis=2) == 0
        sp = r['spatial'].copy()
        sp[bg] = 0.0
        rgb = (CMAP(sp)[:, :, :3] * 255).astype(np.uint8)
        rgb[bg] = 0
        Image.fromarray(rgb).save(out / f'top{rank:02d}_kernel{ki}_spatial.png')
        spectra.append(r['spectral'])

    avg = np.mean(spectra, axis=0)
    pd.DataFrame({'band': np.arange(len(avg)), 'importance': avg}).to_csv(
        out / 'spectral_avg.csv', index=False)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(avg, color='tomato', lw=1.5)
    ax.fill_between(range(len(avg)), avg, alpha=0.25, color='tomato')
    ax.set_xlabel('Band index', fontsize=11)
    ax.set_ylabel('Importance', fontsize=11)
    ax.set_title(f'Spectral Importance (GradCAM avg)\n{target} | CDG-MIL | {label}',
                 fontsize=11, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.savefig(out / 'spectral_avg.png', dpi=150)
    plt.close()
    print(f'    {label}: {n} kernels, top {len(top_idx)}', flush=True)


def main():
    ap = argparse.ArgumentParser(description='CDG-MIL spatial XAI (Grad-CAM, CNN3D)')
    ap.add_argument('--checkpoint', required=True, help='trained CNN3D model (.pt)')
    ap.add_argument('--bags', nargs='+',
                    help='bag keys such as 29-2_CT (default: highest/lowest per treatment in the test set)')
    args = ap.parse_args()

    ckpt   = torch.load(args.checkpoint, map_location=DEVICE, weights_only=False)
    target = ckpt['target']
    if ckpt['backbone'] != 'CNN3D':
        raise SystemExit('Grad-CAM needs a CNN3D checkpoint.')

    model = models.build_model('CNN3D', models.get_head())
    model.load_state_dict(ckpt['state_dict'])
    model.to(DEVICE).eval()
    gcam = GradCAM3D(model)
    kw = dict(spec_mean=ckpt['spec_mean'], spec_std=ckpt['spec_std'],
              y_mean=ckpt['y_mean'], y_std=ckpt['y_std'],
              method=ckpt['prep'], msc_ref=ckpt.get('msc_ref'))

    bag_df = data.build_bag_index()
    df_te  = bag_df[bag_df.replicate == 2].reset_index(drop=True)
    bags   = [(k, k) for k in args.bags] if args.bags else select_bags(df_te, target)
    root   = config.XAI_DIR / 'spatial'
    print(f'[spatial] {target}: CNN3D {ckpt["prep"]} rep{ckpt.get("repeat")}', flush=True)

    for label, key in bags:
        df_bag = bag_df[bag_df.key == key].reset_index(drop=True)
        if df_bag.empty:
            print(f'    [skip] bag not found: {key}')
            continue
        explain_bag(model, gcam, df_bag, target, kw, root / target / label, label)

    save_colorbar(root / 'colorbar.png')
    print(f'written to {root / target}')


if __name__ == '__main__':
    main()
