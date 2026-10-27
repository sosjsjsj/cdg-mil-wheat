"""Paths and hyperparameters."""
from pathlib import Path
import torch

ROOT_DIR    = Path(__file__).resolve().parents[1]
DATA_DIR    = str(ROOT_DIR / 'dataset' / 'hyperspectral_wheat_3d')
DATA_1D_CSV = str(ROOT_DIR / 'dataset' / 'hyperspectral_wheat_1d.csv')
RESULTS_DIR = ROOT_DIR / 'results'

METRICS_DIR = RESULTS_DIR / 'metrics'
PRED_DIR    = RESULTS_DIR / 'predictions'
MODELS_DIR  = RESULTS_DIR / 'models'
FIGURES_DIR = RESULTS_DIR / 'figures'
HISTORY_DIR = RESULTS_DIR / 'history'
XAI_DIR     = RESULTS_DIR / 'xai'

for d in [METRICS_DIR, PRED_DIR, MODELS_DIR, FIGURES_DIR, HISTORY_DIR, XAI_DIR]:
    d.mkdir(parents=True, exist_ok=True)

TARGETS        = ['Protein', 'Moisture', 'Carbohydrate']
TREATMENTS     = ['CT', 'LN', 'HN']
N_BANDS        = 120
SEED           = 42                 # train/validation split
N_SEEDS        = 3                  # repetitions with random seeds
VAL_RATIO      = 0.2
PREPROCESSINGS = ['raw', 'snv', 'msc', 'sg']

FEAT_DIM     = 64
N_AUG        = 5
SUB_BAG_SIZE = 15
EPOCHS       = 150
LR           = 1e-4
WEIGHT_DECAY = 1e-3
GRAD_CLIP    = 1.0
ACCUM_STEPS  = 8

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

USE_AUG = True
