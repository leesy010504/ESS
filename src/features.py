"""Cell selection, data split and feature sets for early cycle-life prediction.

Feature design follows the DAY 1 report (모델 설계 전략 §1). All features are
computed from cycles <= 100 by ``preprocess.py``; this module only chooses which
cells and columns each model sees.
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
CELLS = ROOT / "data" / "processed" / "cells.csv"

TRAIN, TEST, SECONDARY = "Batch 1", "Batch 2", "Batch 3"
TARGET = "cycle_life"

# Batch 3 cells dropped from the paper's secondary test set (public loading script removes
# b3c2, 23, 32, 37, 42, 43 as 0-based indices = 1-based ids 3, 24, 33, 38, 43, 44).
# 3_24 and 3_33 are already unlabeled here.
BATCH3_OUTLIERS = ["3_03", "3_24", "3_33", "3_38", "3_43", "3_44"]

# Single-feature baseline: ln Var[ΔQ100-10(V)] (Spearman ρ -0.888 overall, consistent per batch).
# dq_min is excluded because it duplicates it (ρ = -0.996).
BASE = ["ln_dq_variance"]

# Extension candidates, one feature per redundant pair (qd_10/qd_100, ir_10/ir_100,
# charge_current_10/100). A group is adopted only if it lowers validation error.
GROUPS = {
    "fade": ["early_qd_slope_clean"],          # 10-100 cycle Qd slope ("추가 실험 후보")
    "capacity": ["qd_100"],                    # 100th-cycle Qd ("추가 실험 후보")
    "resistance": ["ir_100"],
    "temperature": ["early_tavg_mean"],
    "charging": ["c1", "switch_soc", "c2", "charge_current_10", "early_charge_time_clean"],
}
ALL_FEATURES = BASE + [f for group in GROUPS.values() for f in group]


def load_cells(path=CELLS):
    """Cells with a usable label: cycle_life present and last Qd near the 0.88 Ah EOL.

    Excludes 10 unlabeled Batch 2 cells (VarCharge/SLOWCYCLE protocols) and 10 Batch 1
    cells whose label equals the last logged cycle + 1 while Qd is still > 0.885 Ah
    (right-censored: testing stopped before EOL).
    """
    cells = pd.read_csv(path)
    usable = cells[TARGET].notna() & cells["endpoint_near_eol"]
    cells = cells[usable].reset_index(drop=True)
    cells["newstructure"] = cells["policy"].str.contains("newstructure")
    return cells


def split(cells):
    """Assignment split: train on Batch 1, test on Batch 2, optional secondary test on Batch 3."""
    return {name: cells[cells["batch"] == batch].reset_index(drop=True)
            for name, batch in (("train", TRAIN), ("test", TEST), ("secondary", SECONDARY))}


def holdout_split(train, seed=0, n_bins=5):
    """Policy-level Batch 1 split into (dev, holdout, holdout_policies).

    Policies are sorted by mean cycle life (ties by name), cut into ``n_bins`` equal bins,
    and one policy per bin goes to the hold-out. No policy is shared between dev and hold-out.
    """
    life = train.groupby("policy")[TARGET].mean().reset_index().sort_values([TARGET, "policy"])
    policies = life["policy"].to_numpy()
    rng = np.random.default_rng(seed)
    picked, bins = [], {}
    for b, chunk in enumerate(np.array_split(policies, n_bins)):
        choice = chunk[rng.integers(len(chunk))]
        picked.append(choice)
        bins.update({p: b for p in chunk})
    holdout = train[train["policy"].isin(picked)].reset_index(drop=True)
    dev = train[~train["policy"].isin(picked)].reset_index(drop=True)
    for frame in (dev, holdout):
        frame["life_bin"] = frame["policy"].map(bins)
    return dev, holdout, picked
