"""Training loop."""
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from data import mil_collate_fn
from config import EPOCHS, LR, WEIGHT_DECAY, GRAD_CLIP, ACCUM_STEPS, DEVICE


def train_mil(model: nn.Module,
              train_dataset,
              val_dataset,
              y_mean: float,
              y_std: float):
    """Train on bag-level loaders; return the model with the lowest validation RMSE
    and the per-epoch history."""
    model        = model.to(DEVICE)
    train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True,
                              collate_fn=mil_collate_fn,
                              num_workers=4, persistent_workers=True,
                              pin_memory=torch.cuda.is_available())
    val_loader   = DataLoader(val_dataset,   batch_size=1, shuffle=False,
                              collate_fn=mil_collate_fn,
                              num_workers=4, persistent_workers=True,
                              pin_memory=torch.cuda.is_available())

    optimizer = torch.optim.AdamW(model.parameters(),
                                  lr=LR, weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=20, min_lr=1e-6)
    criterion = nn.HuberLoss(delta=1.0)

    best_rmse  = np.inf
    best_state = None
    history    = []

    for ep in range(EPOCHS):
        model.train()
        optimizer.zero_grad()
        tr_preds, tr_trues = [], []

        for step, (X, y, _) in enumerate(train_loader):
            X, y = X.to(DEVICE), y.to(DEVICE)
            pred = model(X)
            loss = criterion(pred.unsqueeze(0), y) / ACCUM_STEPS
            loss.backward()

            if (step + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
                optimizer.step()
                optimizer.zero_grad()

            tr_preds.append(float(pred.detach().cpu()) * y_std + y_mean)
            tr_trues.append(float(y.cpu())             * y_std + y_mean)

        torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
        optimizer.step()
        optimizer.zero_grad()

        tr_rmse = float(np.sqrt(np.mean(
            (np.array(tr_preds) - np.array(tr_trues)) ** 2)))

        model.eval()
        va_preds, va_trues = [], []
        with torch.no_grad():
            for X, y, _ in val_loader:
                pred = model(X.to(DEVICE))
                va_preds.append(float(pred.cpu()) * y_std + y_mean)
                va_trues.append(float(y)          * y_std + y_mean)

        va_rmse = float(np.sqrt(np.mean(
            (np.array(va_preds) - np.array(va_trues)) ** 2)))

        scheduler.step(tr_rmse)
        lr = optimizer.param_groups[0]['lr']

        is_best = va_rmse < best_rmse
        if is_best:
            best_rmse  = va_rmse
            best_state = {k: v.cpu().clone()
                          for k, v in model.state_dict().items()}

        history.append({
            'epoch':      ep + 1,
            'train_RMSE': round(tr_rmse, 4),
            'val_RMSE':   round(va_rmse, 4),
            'lr':         lr,
            'is_best':    is_best,
        })
        print(f'  ep {ep+1:3d}/{EPOCHS}  '
              f'train={tr_rmse:.4f}  val={va_rmse:.4f}  '
              f'lr={lr:.1e}{"  <- best" if is_best else ""}')

    model.load_state_dict(best_state)
    return model, history
