"""Average the masked pixels of every kernel cube into one 120-band spectrum.

Input:  dataset/hyperspectral_wheat_3d/{CT,LN,HN}/*.npy
Output: dataset/hyperspectral_wheat_1d.csv (one row per kernel)
"""

import os
import re
import numpy as np
import pandas as pd

ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, "dataset", "hyperspectral_wheat_3d")
OUTPUT_CSV = os.path.join(ROOT_DIR, "dataset", "hyperspectral_wheat_1d.csv")

TREATMENTS = ["CT", "LN", "HN"]

rows = []
n_bands = 120

for treatment_code in TREATMENTS:
    folder_path = os.path.join(DATA_DIR, treatment_code)
    files = sorted(os.listdir(folder_path))

    for fname in files:
        if not fname.endswith(".npy"):
            continue

        m = re.match(r"VNIR_(\d+)-(\d+)_(\w+)_kernel_(\d+)\.npy", fname)
        if not m:
            continue

        arr = np.load(os.path.join(folder_path, fname))
        pixels = arr.reshape(-1, n_bands)

        # Pixels outside the kernel mask are zero in every band.
        valid_pixels = pixels[pixels.sum(axis=1) > 0]
        if valid_pixels.shape[0] == 0:
            continue

        mean_spectrum = valid_pixels.mean(axis=0)

        row = {
            "genotype":  int(m.group(1)),
            "replicate": int(m.group(2)),
            "treatment": treatment_code,
            "kernel_id": int(m.group(4)),
        }
        for b in range(n_bands):
            row[f"b{b:03d}"] = mean_spectrum[b]

        rows.append(row)

df_out = pd.DataFrame(rows)
df_out["treatment"] = pd.Categorical(df_out["treatment"], categories=TREATMENTS)
df_out = df_out.sort_values(["genotype", "replicate", "treatment", "kernel_id"]).reset_index(drop=True)
df_out = df_out.drop(columns="kernel_id")
df_out.to_csv(OUTPUT_CSV, index=False)

n_bags = df_out.groupby(["genotype", "replicate", "treatment"]).ngroups
print(f"Saved {len(df_out):,} kernels in {n_bags} bags to {OUTPUT_CSV}")
print(df_out.groupby(["treatment", "replicate"]).size().to_string())
