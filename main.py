"""Train CDG-MIL (replicate 1, train/validation 8:2) and evaluate on replicate 2.

    python main.py --target Protein --backbone KAN --prep msc
    python main.py --checkpoint results/models/model_rep1_Protein_msc_KAN.pt
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
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent / 'code_training'))
import config
import data
import evaluate
import models
import train

BACKBONES = ['CNN1D', 'CNN3D', 'Transformer', 'DCNN', 'Mamba', 'KAN']


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def make_dataset(bag_df, target, backbone, stats, is_train):
    Dataset = data.BagDataset3D if backbone == 'CNN3D' else data.BagDataset1D
    return Dataset(bag_df, target,
                   spec_mean=stats['spec_mean'], spec_std=stats['spec_std'],
                   y_mean=stats['y_mean'], y_std=stats['y_std'],
                   method=stats['prep'], msc_ref=stats['msc_ref'],
                   is_train=is_train,
                   use_aug=config.USE_AUG if is_train else False)


def predict(model, dataset, y_mean, y_std):
    model.eval()
    keys, y_true, y_pred = [], [], []
    with torch.no_grad():
        for X, y, key in DataLoader(dataset, batch_size=1, collate_fn=data.mil_collate_fn):
            y_pred.append(float(model(X.to(config.DEVICE)).cpu()) * y_std + y_mean)
            y_true.append(float(y) * y_std + y_mean)
            keys.append(key)
    return keys, np.array(y_true), np.array(y_pred)


def save_checkpoint(model, tag, stats, backbone, target, rep):
    torch.save({
        'state_dict': model.state_dict(),
        'spec_mean':  stats['spec_mean'], 'spec_std': stats['spec_std'],
        'y_mean':     stats['y_mean'],    'y_std':    stats['y_std'],
        'backbone':   backbone,           'target':   target,
        'prep':       stats['prep'],      'repeat':   rep,
        'msc_ref':    stats['msc_ref'],
    }, config.MODELS_DIR / f'model_{tag}.pt')


def save_scatter(y_true, y_pred, metrics, tag):
    _, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, alpha=0.6, s=60, color='steelblue')
    lo  = min(y_true.min(), y_pred.min())
    hi  = max(y_true.max(), y_pred.max())
    pad = (hi - lo) * 0.04
    ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], 'r--', lw=1.5)
    ax.text(0.05, 0.92, f"R²={metrics['R2']:.3f}\nRMSE={metrics['RMSE']:.3f}",
            transform=ax.transAxes, fontsize=11,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    ax.set_xlabel('Actual'); ax.set_ylabel('Predicted')
    ax.set_title(tag, fontsize=9, fontweight='bold')
    ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(config.FIGURES_DIR / f'scatter_{tag}.png', dpi=150)
    plt.close()


def save_history(history, tag):
    df = pd.DataFrame(history)
    df.to_csv(config.HISTORY_DIR / f'history_{tag}.csv', index=False)
    best = df.loc[df['is_best']].iloc[-1]
    _, ax = plt.subplots(figsize=(8, 4))
    ax.plot(df['epoch'], df['train_RMSE'], label='Train RMSE', color='steelblue', lw=1.5)
    ax.plot(df['epoch'], df['val_RMSE'],   label='Validation RMSE', color='tomato', lw=1.5)
    ax.axvline(best['epoch'], color='gray', linestyle='--', lw=1, alpha=0.7)
    ax.scatter([best['epoch']], [best['val_RMSE']], color='tomato', zorder=5, s=60,
               label=f"Best ep{int(best['epoch'])} ({best['val_RMSE']:.3f})")
    ax.set_xlabel('Epoch'); ax.set_ylabel('RMSE')
    ax.set_title(tag, fontsize=9, fontweight='bold')
    ax.legend(fontsize=9); ax.grid(alpha=0.3, linestyle='--')
    ax.spines['top'].set_visible(False); ax.spines['right'].set_visible(False)
    plt.tight_layout()
    plt.savefig(config.HISTORY_DIR / f'history_{tag}.png', dpi=150)
    plt.close()


def save_results(keys, y_true, y_pred, tag, info):
    metrics = evaluate.compute_metrics(y_true, y_pred)
    pd.DataFrame({'key': keys, 'y_true': y_true, 'y_pred': y_pred}).to_csv(
        config.PRED_DIR / f'pred_{tag}.csv', index=False)
    save_scatter(y_true, y_pred, metrics, tag)

    path = config.METRICS_DIR / 'metrics.csv'
    row  = pd.DataFrame([{**info, **metrics}])
    row.to_csv(path, mode='a', header=not path.exists(), index=False)
    print(f'[{tag}] ' + '  '.join(f'{k}={v}' for k, v in metrics.items()))
    return metrics


def run_train(target, backbone, prep, seed, rep):
    set_seed(seed)
    rep1, df_tr, df_va, df_te = data.split_bags(data.build_bag_index())

    spec_mean, spec_std = data.compute_train_stats(rep1)
    stats = dict(spec_mean=spec_mean, spec_std=spec_std,
                 y_mean=float(rep1[target].mean()),
                 y_std=float(rep1[target].std()) or 1.0,
                 prep=prep,
                 msc_ref=data.compute_msc_ref(rep1) if prep == 'msc' else None)

    train_ds = make_dataset(df_tr, target, backbone, stats, is_train=True)
    val_ds   = make_dataset(df_va, target, backbone, stats, is_train=False)
    test_ds  = make_dataset(df_te, target, backbone, stats, is_train=False)

    model = models.build_model(backbone, models.get_head())
    model, history = train.train_mil(model, train_ds, val_ds, stats['y_mean'], stats['y_std'])

    tag = f'rep{rep}_{target}_{prep}_{backbone}'
    save_checkpoint(model, tag, stats, backbone, target, rep)
    save_history(history, tag)
    keys, y_true, y_pred = predict(model, test_ds, stats['y_mean'], stats['y_std'])
    return save_results(keys, y_true, y_pred, tag,
                        dict(mode='train', target=target, prep=prep, repeat=rep, backbone=backbone))


def summarise(target, backbone, prep, runs):
    df  = pd.DataFrame(runs)
    row = dict(target=target, backbone=backbone, prep=prep, n_repeats=len(runs))
    for k in df.columns:
        row[f'{k}_mean'] = round(float(df[k].mean()), 4)
        row[f'{k}_sd']   = round(float(df[k].std(ddof=1)), 4) if len(df) > 1 else 0.0
    path = config.METRICS_DIR / 'summary.csv'
    pd.DataFrame([row]).to_csv(path, mode='a', header=not path.exists(), index=False)
    print(f'[{target} | {backbone} | {prep}] over {len(runs)} repetitions: ' +
          '  '.join(f"{k}={row[f'{k}_mean']}±{row[f'{k}_sd']}" for k in df.columns))


def run_checkpoint(path):
    ckpt     = torch.load(path, map_location=config.DEVICE, weights_only=False)
    backbone = ckpt['backbone']
    target   = ckpt['target']
    stats    = {k: ckpt[k] for k in ['spec_mean', 'spec_std', 'y_mean', 'y_std', 'prep', 'msc_ref']}

    *_, df_te = data.split_bags(data.build_bag_index())
    test_ds  = make_dataset(df_te, target, backbone, stats, is_train=False)

    model = models.build_model(backbone, models.get_head()).to(config.DEVICE)
    model.load_state_dict(ckpt['state_dict'])

    tag = f"eval_{Path(path).stem.removeprefix('model_')}"
    keys, y_true, y_pred = predict(model, test_ds, stats['y_mean'], stats['y_std'])
    save_results(keys, y_true, y_pred, tag,
                 dict(mode='checkpoint', target=target, prep=stats['prep'],
                      repeat=ckpt.get('repeat'), backbone=backbone))


def main():
    parser = argparse.ArgumentParser(description='CDG-MIL wheat quality prediction')
    parser.add_argument('--target',   choices=config.TARGETS, default='Protein')
    parser.add_argument('--backbone', choices=BACKBONES, default='CNN1D')
    parser.add_argument('--prep',     choices=config.PREPROCESSINGS, default='raw')
    parser.add_argument('--seeds',    type=int, nargs='+',
                        help=f'seeds to use (default: {config.N_SEEDS} random seeds)')
    parser.add_argument('--checkpoint', help='evaluate a saved .pt instead of training')
    args = parser.parse_args()

    if args.checkpoint:
        run_checkpoint(args.checkpoint)
    else:
        seeds = args.seeds or np.random.default_rng().integers(0, 2**31 - 1, config.N_SEEDS).tolist()
        print(f'seeds: {seeds}')
        runs = [run_train(args.target, args.backbone, args.prep, s, i)
                for i, s in enumerate(seeds, start=1)]
        summarise(args.target, args.backbone, args.prep, runs)


if __name__ == '__main__':
    main()
