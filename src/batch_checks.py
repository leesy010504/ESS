"""Batch-level data checks that back the Batch 3 caveats in the assignment.

1) Qdlin start point: is the ΔQ100-10(V) feature distorted by batch-specific curve starts?
2) Batch 3 outliers: do the cells removed by Severson et al. look bad on logger metrics?
3) Cycle duration: wall-clock minutes per cycle (cycles 2-100), to quote batch-level protocol differences.
Run from the project root: ``python src/batch_checks.py``. Writes ``results/qdlin_batch_check.csv``,
``results/batch3_outlier_check.csv``, ``results/batch_cycle_duration.csv`` and ``figures/b3_qdlin_check.png``.
"""

from pathlib import Path
import sys

import h5py
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import BATCH3_OUTLIERS  # noqa: E402
from preprocess import FILES, RAW, values  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS, FIGURES = ROOT / "results", ROOT / "figures"
BATCH_COLORS = {"Batch 1": "#2a78d6", "Batch 2": "#eb6834", "Batch 3": "#1baf7a"}


def qdlin_rows(file, source, batch):
    rows, curves = [], {}
    voltage = values(file[source["Vdlin"][0, 0]])
    for index in range(len(source["cycle_life"])):
        cell_id = f"{batch[-1]}_{index + 1:02d}"
        cycle = values(file[source["summary"][index, 0]]["cycle"]).astype(int)
        qv = file[source["cycles"][index, 0]]["Qdlin"]
        got = {}
        for selected in (10, 100):
            position = np.flatnonzero(cycle == selected)
            if len(position) and position[0] < len(qv):
                got[selected] = values(file[qv[int(position[0]), 0]])
        if len(got) < 2:
            continue
        dq = got[100] - got[10]
        curves[cell_id] = got
        rows.append({"batch": batch, "cell_id": cell_id, "n_vdlin": len(voltage),
                     "vdlin_first": voltage[0], "vdlin_last": voltage[-1],
                     "qdlin10_first": got[10][0], "qdlin100_first": got[100][0],
                     "qdlin10_last": got[10][-1], "qdlin100_last": got[100][-1],
                     "dq_first": dq[0], "dq_last": dq[-1], "dq_var": np.var(dq)})
    return rows, curves


def cycle_duration_rows(file, source, batch):
    rows = []
    for index in range(len(source["cycle_life"])):
        cycle = values(file[source["summary"][index, 0]]["cycle"]).astype(int)
        cycles = file[source["cycles"][index, 0]]["t"]
        minutes = []
        for position in np.flatnonzero((cycle >= 2) & (cycle <= 100)):
            if position >= len(cycles):
                continue
            t = values(file[cycles[int(position), 0]])
            if len(t) and np.isfinite(t[-1]) and 0 < t[-1] <= 300:
                minutes.append(float(t[-1]))
        if minutes:
            rows.append({"batch": batch, "cell_id": f"{batch[-1]}_{index + 1:02d}",
                         "n_cycles_used": len(minutes), "cycle_minutes_median": float(np.median(minutes))})
    return rows


def outlier_rows(file, source, batch, life_all):
    rows = []
    for index in range(len(source["cycle_life"])):
        summary = file[source["summary"][index, 0]]
        cycle = values(summary["cycle"]).astype(int)
        qd, ir, tavg, ct = (values(summary[k]) for k in ("QDischarge", "IR", "Tavg", "chargetime"))
        n = min(len(a) for a in (cycle, qd, ir, tavg, ct))
        cycle, qd, ir, tavg, ct = (a[:n] for a in (cycle, qd, ir, tavg, ct))
        life = life_all[index]
        limit = life if np.isfinite(life) else cycle[-1]
        used = cycle <= limit
        rows.append({
            "cell_id": f"{batch[-1]}_{index + 1:02d}", "has_label": bool(np.isfinite(life)),
            "cycle_life": life, "n_cycles_logged": int(n),
            "missing_cycles_10_100": int(91 - np.isin(np.arange(10, 101), cycle).sum()),
            "qd_glitch_count": int(((qd > 1.3) | (qd <= 0))[used].sum()),
            "ir_zero_or_nan_frac": float((~(ir > 0))[used].mean()),
            "tavg_max": float(np.nanmax(tavg[used])),
            "tavg_jump_max": float(np.nanmax(np.abs(np.diff(tavg[used])))),
            "chargetime_glitch_count": int((ct[used] > 60).sum()),
            "qd_jump_max": float(np.nanmax(np.abs(np.diff(qd[used & (qd > 0)])))),
        })
    return pd.DataFrame(rows)


def main():
    RESULTS.mkdir(exist_ok=True)
    FIGURES.mkdir(exist_ok=True)
    q_rows, t_rows, curves, outliers = [], [], {}, None
    for batch, filename in FILES.items():
        with h5py.File(RAW / filename) as file:
            source = file["batch"]
            rows, got = qdlin_rows(file, source, batch)
            q_rows += rows
            curves.update(got)
            t_rows += cycle_duration_rows(file, source, batch)
            if batch == "Batch 3":
                life = np.array([values(file[source["cycle_life"][i, 0]])[0]
                                 for i in range(len(source["cycle_life"]))])
                outliers = outlier_rows(file, source, batch, life)
    qd = pd.DataFrame(q_rows)
    qd.to_csv(RESULTS / "qdlin_batch_check.csv", index=False)
    summary = qd.groupby("batch")[["qdlin10_first", "qdlin100_first", "dq_first", "dq_last"]].agg(
        ["median", lambda s: s.quantile(.75) - s.quantile(.25)])
    summary.columns = [f"{a}_{'median' if b == 'median' else 'iqr'}" for a, b in summary.columns]
    print("Vdlin first/last/n per batch:\n", qd.groupby("batch")[["vdlin_first", "vdlin_last", "n_vdlin"]]
          .agg(["min", "max"]).to_string())
    print("Qdlin start / ΔQ start per batch:\n", summary.T.to_string())

    duration = pd.DataFrame(t_rows)
    duration.round(3).to_csv(RESULTS / "batch_cycle_duration.csv", index=False)
    per_batch = duration.groupby("batch").cycle_minutes_median
    print("Cycle duration (min) across cells, per batch:\n",
          pd.DataFrame({"median": per_batch.median(), "iqr": per_batch.quantile(.75) - per_batch.quantile(.25),
                        "cells": per_batch.count()}).round(1).to_string())

    flags = ["missing_cycles_10_100", "qd_glitch_count", "ir_zero_or_nan_frac", "tavg_jump_max",
             "chargetime_glitch_count", "qd_jump_max"]
    out_ids = set(BATCH3_OUTLIERS)
    outliers["paper_excluded"] = outliers["cell_id"].isin(out_ids)
    for col in flags:
        median = outliers[col].median()
        mad = (outliers[col] - median).abs().median()
        outliers[f"flag_{col}"] = outliers[col] > median + 5 * max(mad, 1e-9 if col != "ir_zero_or_nan_frac" else 0.01)
    outliers["n_flags"] = outliers[[f"flag_{c}" for c in flags]].sum(axis=1)
    outliers.round(4).to_csv(RESULTS / "batch3_outlier_check.csv", index=False)
    print(outliers[outliers.paper_excluded | (outliers.n_flags > 0)][
        ["cell_id", "has_label", "cycle_life", "n_flags", *flags]].to_string())

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8))
    voltage = np.linspace(3.5, 2.0, 1000)
    for batch, cell in (("Batch 1", "1_01"), ("Batch 2", "2_01"), ("Batch 3", "3_01")):
        color = BATCH_COLORS[batch]
        for selected, style in ((10, "-"), (100, "--")):
            axes[0].plot(voltage, curves[cell][selected], style, color=color, label=f"{batch} {cell} c{selected}")
    axes[0].set(xlabel="Voltage (V)", ylabel="Qdlin (Ah)", title="Qdlin curves, one cell per batch")
    axes[0].legend(fontsize=6)
    for ax, col, title in ((axes[1], "qdlin10_first", "Qdlin cycle 10, first point"),
                           (axes[2], "dq_first", "ΔQ100-10 at first point")):
        data = [qd[qd.batch == b][col] for b in FILES]
        boxes = ax.boxplot(data, tick_labels=list(FILES), patch_artist=True)
        for box, b in zip(boxes["boxes"], FILES):
            box.set(facecolor=BATCH_COLORS[b], alpha=0.6)
        ax.set(ylabel="Ah", title=title)
    fig.tight_layout()
    fig.savefig(FIGURES / "b3_qdlin_check.png", dpi=130)
    print("Saved:", RESULTS, FIGURES)


if __name__ == "__main__":
    main()
