"""Regression metrics."""
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error


def compute_metrics(y_true, y_pred):
    r2   = r2_score(y_true, y_pred)
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    mae  = float(mean_absolute_error(y_true, y_pred))
    rpd  = float(np.std(y_true, ddof=1) / (rmse + 1e-10))
    bias = float(np.mean(y_pred - y_true))
    return {
        'R2':   round(r2,   4),
        'RMSE': round(rmse, 4),
        'MAE':  round(mae,  4),
        'RPD':  round(rpd,  4),
        'Bias': round(bias, 4),
    }
