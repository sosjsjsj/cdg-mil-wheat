"""Bag construction, spectral preprocessing and datasets."""
import os
import re
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from scipy.signal import savgol_filter
from config import DATA_DIR, DATA_1D_CSV, TREATMENTS, N_BANDS, SEED, VAL_RATIO, N_AUG, SUB_BAG_SIZE


def _snv(x: np.ndarray) -> np.ndarray:
    m = x.mean(axis=1, keepdims=True)
    s = x.std(axis=1, keepdims=True)
    s[s < 1e-8] = 1e-8
    return (x - m) / s


def _msc(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    out = np.zeros_like(x)
    for i in range(x.shape[0]):
        coef   = np.polyfit(ref, x[i], 1)
        out[i] = (x[i] - coef[1]) / (coef[0] + 1e-8)
    return out


def _sg(x: np.ndarray) -> np.ndarray:
    return savgol_filter(x, window_length=11, polyorder=2, axis=1)


def preprocess_spectra(x: np.ndarray, method: str,
                       msc_ref: np.ndarray = None) -> np.ndarray:
    """x: (N, N_BANDS). method: 'raw', 'snv', 'msc' or 'sg'."""
    if method == 'raw':   return x.copy()
    if method == 'snv':   return _snv(x)
    if method == 'msc':   return _msc(x, msc_ref)
    if method == 'sg':    return _sg(x)
    raise ValueError(f'Unknown preprocessing method: {method}')


# Without the 3-D cubes, kernel spectra are read from the 1-D CSV. Rows are named
# by their position within the bag.
_SPECTRA_BY_FILE = None


def _spectra_by_file():
    global _SPECTRA_BY_FILE
    if _SPECTRA_BY_FILE is None:
        df = pd.read_csv(DATA_1D_CSV)
        bands = [f'b{i:03d}' for i in range(N_BANDS)]
        values = df[bands].to_numpy(dtype=np.float32)
        position = df.groupby(['genotype', 'replicate', 'treatment']).cumcount()
        names = [f"VNIR_{int(v)}-{int(r)}_{t}_kernel_{int(k):03d}.npy"
                 for v, r, t, k in zip(df['genotype'], df['replicate'],
                                       df['treatment'], position)]
        _SPECTRA_BY_FILE = dict(zip(names, values))
    return _SPECTRA_BY_FILE


def load_pixels(fpath):
    """(n_pixels, N_BANDS) from the cube, or the (1, N_BANDS) mean spectrum from the CSV."""
    if os.path.exists(fpath):
        return np.load(fpath).astype(np.float32).reshape(-1, N_BANDS)
    spectrum = _spectra_by_file().get(os.path.basename(fpath))
    if spectrum is None:
        raise FileNotFoundError(f'No cube and no CSV spectrum for {fpath}')
    return spectrum.reshape(1, N_BANDS)


def compute_msc_ref(bag_df: pd.DataFrame) -> np.ndarray:
    """Mean spectrum of all kernels in the training bags."""
    spectra = []
    for _, row in bag_df.iterrows():
        for fpath in row['fpath_list']:
            arr   = load_pixels(fpath)
            valid = arr[arr.sum(axis=1) > 0]
            if len(valid) > 0:
                spectra.append(valid.mean(axis=0))
    if not spectra:
        return np.ones(N_BANDS, dtype=np.float32)
    return np.array(spectra, dtype=np.float32).mean(axis=0)


def compute_train_stats(bag_df: pd.DataFrame):
    """Band-wise mean and std of the raw kernel spectra."""
    spectra = []
    for _, row in bag_df.iterrows():
        for fpath in row['fpath_list']:
            arr   = load_pixels(fpath)
            valid = arr[arr.sum(axis=1) > 0]
            if len(valid) > 0:
                spectra.append(valid.mean(axis=0))
    if not spectra:
        return np.zeros(N_BANDS, dtype=np.float32), np.ones(N_BANDS, dtype=np.float32)
    spectra = np.array(spectra, dtype=np.float32)
    mean    = spectra.mean(axis=0)
    std     = spectra.std(axis=0)
    std[std < 1e-8] = 1e-8
    return mean, std


def build_bag_index() -> pd.DataFrame:
    """One row per bag: genotype, replicate, treatment, fpath_list, bag_size, key and labels."""
    df         = pd.read_csv(DATA_1D_CSV)
    bag_labels = (
        df[['genotype', 'replicate', 'treatment', 'Protein', 'Moisture', 'Carbohydrate']]
        .drop_duplicates()
    )

    label_lookup: dict = {}
    for _, row in bag_labels.iterrows():
        key                = f"{int(row['genotype'])}-{int(row['replicate'])}_{row['treatment']}"
        label_lookup[key]  = row.to_dict()

    bag_dict: dict = {}
    for code in TREATMENTS:
        folder_path = os.path.join(DATA_DIR, code)
        if os.path.exists(folder_path):
            fnames = sorted(os.listdir(folder_path))
        else:
            fnames = sorted(n for n in _spectra_by_file() if f'_{code}_' in n)
        for fname in fnames:
            m = re.match(r'VNIR_(\d+)-(\d+)_(\w+)_kernel_(\d+)\.npy', fname)
            if not m:
                continue
            genotype, replicate = int(m.group(1)), int(m.group(2))
            key                 = f'{genotype}-{replicate}_{code}'
            if key not in label_lookup:
                continue
            fpath = os.path.join(folder_path, fname)
            if key not in bag_dict:
                bag_dict[key] = {
                    'genotype':      genotype,
                    'replicate':     replicate,
                    'treatment':     code,
                    'fpath_list':    [],
                    'Protein':       label_lookup[key]['Protein'],
                    'Moisture':      label_lookup[key]['Moisture'],
                    'Carbohydrate':  label_lookup[key]['Carbohydrate'],
                }
            bag_dict[key]['fpath_list'].append(fpath)

    rows = []
    for key, info in bag_dict.items():
        rows.append({
            'genotype':     info['genotype'],
            'replicate':    info['replicate'],
            'treatment':    info['treatment'],
            'fpath_list':   info['fpath_list'],
            'bag_size':     len(info['fpath_list']),
            'key':          key,
            'Protein':      info['Protein'],
            'Moisture':     info['Moisture'],
            'Carbohydrate': info['Carbohydrate'],
        })
    return pd.DataFrame(rows)


def split_bags(bag_df: pd.DataFrame):
    """Replicate 1 -> fixed 8:2 train/validation split, replicate 2 -> test."""
    rep1 = bag_df[bag_df['replicate'] == 1].sort_values('key').reset_index(drop=True)
    test = bag_df[bag_df['replicate'] == 2].reset_index(drop=True)
    perm  = np.random.default_rng(SEED).permutation(len(rep1))
    n_val = int(round(len(rep1) * VAL_RATIO))
    val   = rep1.iloc[perm[:n_val]].reset_index(drop=True)
    train = rep1.iloc[perm[n_val:]].reset_index(drop=True)
    return rep1, train, val, test


class BagDataset1D(Dataset):
    """Returns (X, y, key) with X: (N, N_BANDS) kernel mean spectra.

    During training with use_aug, each bag yields N_AUG sub-bags of SUB_BAG_SIZE kernels.
    """

    def __init__(self, bag_df: pd.DataFrame, target: str,
                 spec_mean: np.ndarray, spec_std: np.ndarray,
                 y_mean: float, y_std: float,
                 method: str = 'raw', msc_ref: np.ndarray = None,
                 is_train: bool = True, use_aug: bool = True):
        self.bags      = bag_df.reset_index(drop=True)
        self.target    = target
        self.spec_mean = torch.FloatTensor(spec_mean)
        self.spec_std  = torch.FloatTensor(spec_std)
        self.y_mean    = float(y_mean)
        self.y_std     = float(y_std)
        self.method    = method
        self.msc_ref   = msc_ref
        self.is_train  = is_train
        self.use_aug   = use_aug

    def __len__(self) -> int:
        if self.is_train and self.use_aug:
            return len(self.bags) * N_AUG
        return len(self.bags)

    def __getitem__(self, idx: int):
        bag_idx = idx % len(self.bags)
        row     = self.bags.iloc[bag_idx]
        fpaths  = list(row['fpath_list'])

        if self.is_train and self.use_aug:
            k      = min(SUB_BAG_SIZE, len(fpaths))
            fpaths = np.random.choice(fpaths, k, replace=False).tolist()

        spectra = []
        for fpath in fpaths:
            arr   = load_pixels(fpath)
            valid = arr[arr.sum(axis=1) > 0]
            if len(valid) == 0:
                valid = arr[:1]
            spectra.append(valid.mean(axis=0))

        X = np.array(spectra, dtype=np.float32)
        X = preprocess_spectra(X, self.method, self.msc_ref).astype(np.float32)
        X = (torch.FloatTensor(X) - self.spec_mean) / self.spec_std
        y = torch.FloatTensor([(row[self.target] - self.y_mean) / self.y_std])
        return X, y, row['key']


class BagDataset3D(Dataset):
    """Returns (X, y, key) with X: (N, 50, 50, N_BANDS) kernel cubes."""

    def __init__(self, bag_df: pd.DataFrame, target: str,
                 spec_mean: np.ndarray, spec_std: np.ndarray,
                 y_mean: float, y_std: float,
                 method: str = 'raw', msc_ref: np.ndarray = None,
                 is_train: bool = True, use_aug: bool = True):
        self.bags      = bag_df.reset_index(drop=True)
        self.target    = target
        self.spec_mean = spec_mean.astype(np.float32)
        self.spec_std  = spec_std.astype(np.float32)
        self.y_mean    = float(y_mean)
        self.y_std     = float(y_std)
        self.method    = method
        self.msc_ref   = msc_ref
        self.is_train  = is_train
        self.use_aug   = use_aug

    def __len__(self) -> int:
        if self.is_train and self.use_aug:
            return len(self.bags) * N_AUG
        return len(self.bags)

    def __getitem__(self, idx: int):
        bag_idx = idx % len(self.bags)
        row     = self.bags.iloc[bag_idx]
        fpaths  = list(row['fpath_list'])

        if self.is_train and self.use_aug:
            k      = min(SUB_BAG_SIZE, len(fpaths))
            fpaths = np.random.choice(fpaths, k, replace=False).tolist()

        cubes = []
        for fpath in fpaths:
            cube = np.load(fpath).astype(np.float32)
            if cube.shape != (50, 50, N_BANDS):
                cube = cube.reshape(50, 50, N_BANDS)
            if self.method != 'raw':
                pixels = cube.reshape(-1, N_BANDS)
                valid = pixels.sum(axis=1) > 0
                pixels[valid] = preprocess_spectra(pixels[valid], self.method, self.msc_ref)
            cube = (cube - self.spec_mean) / self.spec_std
            cubes.append(cube)

        X = np.stack(cubes, axis=0)
        X = torch.FloatTensor(X)
        y = torch.FloatTensor([(row[self.target] - self.y_mean) / self.y_std])
        return X, y, row['key']


def mil_collate_fn(batch):
    """Unwrap a batch of one bag."""
    X, y, key = batch[0]
    return X, y, key
