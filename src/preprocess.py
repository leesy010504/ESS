"""Extract the three requested battery batches without loading whole MAT files.

Run from the project root: ``python src/preprocess.py``. Writes ``data/processed/``
(cells.csv, qd_trajectories.csv, delta_q_curves.csv). Every model feature uses
cycles <= 100 only; full-life fade/knee metrics are kept for EDA description.

DAY 1 EDA columns are kept exactly as submitted. Two of them were found to contain
logger glitches during modeling, so ``*_clean`` versions are added for the model.
"""

from pathlib import Path
import re

import h5py
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data"
OUT = RAW / "processed"
FILES = {
    "Batch 1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    "Batch 2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",
    "Batch 3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
POLICY = re.compile(r"(\d+(?:\.\d+)?)C\((\d+)%\)-(\d+(?:\.\d+)?)C")


def values(dataset):
    return np.asarray(dataset[()]).reshape(-1)


def string(dataset):
    return "".join(chr(int(x)) for x in values(dataset) if x).strip()


def at_cycle(cycles, values_, cycle):
    found = np.flatnonzero(cycles == cycle)
    return float(values_[found[0]]) if len(found) else np.nan


def fade_metrics(cycles, qd, life):
    """Descriptive, full-life metrics; never use them as early model features."""
    good = np.isfinite(qd) & (qd > 0) & (cycles >= 2) & (cycles <= life)
    x, y = cycles[good], qd[good]
    if len(x) < 50 or x[-1] < 0.95 * life:
        return np.nan, np.nan, np.nan
    q20, q60, q95 = np.interp(np.array([0.20, 0.60, 0.95]) * life, x, y)
    early = (q20 - q60) / (0.40 * life)
    late = (q60 - q95) / (0.35 * life)
    # Two-line fit is an exploratory breakpoint, not a physical knee estimate.
    x, y = x[x >= 10], y[x >= 10]
    best_sse, knee = np.inf, np.nan
    for fraction in np.linspace(0.30, 0.90, 61):
        split = fraction * life
        left = x <= split
        if left.sum() < 15 or (~left).sum() < 15:
            continue
        a = np.polyfit(x[left], y[left], 1)
        b = np.polyfit(x[~left], y[~left], 1)
        sse = np.square(y[left] - np.polyval(a, x[left])).sum()
        sse += np.square(y[~left] - np.polyval(b, x[~left])).sum()
        if sse < best_sse:
            best_sse, knee = sse, fraction
    return float(early), float(late), float(knee)


def main():
    cells, trajectories, curves = [], [], []
    for batch, filename in FILES.items():
        path = RAW / filename
        if not path.is_file():
            raise FileNotFoundError(path)
        with h5py.File(path) as file:
            source = file["batch"]
            for index in range(len(source["cycle_life"])):
                cell_id = f"{batch[-1]}_{index + 1:02d}"
                summary = file[source["summary"][index, 0]]
                cycle_group = file[source["cycles"][index, 0]]
                cycle = values(summary["cycle"]).astype(int)
                qd = values(summary["QDischarge"])
                ir = values(summary["IR"])
                tavg = values(summary["Tavg"])
                charge_time = values(summary["chargetime"])
                lengths = [len(a) for a in (cycle, qd, ir, tavg, charge_time)]
                if len(set(lengths)) != 1:
                    raise ValueError(f"{batch} cell {index + 1}: summary lengths differ: {lengths}")
                n = lengths[0]
                cycle, qd, ir, tavg, charge_time = (
                    a[:n] for a in (cycle, qd, ir, tavg, charge_time)
                )
                raw_life = float(values(file[source["cycle_life"][index, 0]])[0])
                life = int(raw_life) if np.isfinite(raw_life) else np.nan
                limit = life if np.isfinite(life) else cycle[-1]
                # QC tolerance near the nominal 0.88 Ah EOL boundary; not proof of a true EOL event.
                endpoint_near_eol = bool(qd[-1] <= 0.885)
                policy = string(file[source["policy_readable"][index, 0]])
                match = POLICY.search(policy)
                c1, switch_soc, c2 = (
                    map(float, match.groups()) if match else (np.nan, np.nan, np.nan)
                )
                positive = np.isfinite(qd) & (qd > 0)
                usable = positive & (cycle <= limit)
                early = usable & (cycle >= 10) & (cycle <= 100)
                slope = float(np.polyfit(cycle[early], qd[early], 1)[0]) if early.sum() >= 10 else np.nan
                # Logger glitches: Qd far above the 1.1 Ah nominal (e.g. 1_19 at cycle 40) and
                # charge times of 419-3,934 min (normal 9-14 min; experiment pauses).
                clean_q = early & (qd < 1.3)
                slope_clean = (float(np.polyfit(cycle[clean_q], qd[clean_q], 1)[0])
                               if clean_q.sum() >= 10 else np.nan)
                clean_t = early & (charge_time > 0) & (charge_time <= 60)
                charge_time_clean = float(np.mean(charge_time[clean_t])) if clean_t.any() else np.nan
                early_fade, late_fade, knee = (
                    fade_metrics(cycle, qd, life) if np.isfinite(life) and endpoint_near_eol
                    else (np.nan, np.nan, np.nan)
                )
                qv = cycle_group["Qdlin"]
                current_refs = cycle_group["I"]
                voltage = values(file[source["Vdlin"][index, 0]])
                curve_by_cycle, current_means = {}, {}
                for selected in (10, 100):
                    position = np.flatnonzero(cycle == selected)
                    if len(position) and position[0] < len(qv):
                        j = int(position[0])
                        curve_by_cycle[selected] = values(file[qv[j, 0]])
                        current = values(file[current_refs[j, 0]])
                        charging = current[np.isfinite(current) & (current > 0)]
                        current_means[selected] = float(np.mean(charging)) if len(charging) else np.nan
                    else:
                        current_means[selected] = np.nan
                dq_variance, dq_min = np.nan, np.nan
                if len(curve_by_cycle) == 2 and all(len(v) == len(voltage) for v in curve_by_cycle.values()):
                    delta = curve_by_cycle[100] - curve_by_cycle[10]
                    dq_variance = float(np.var(delta))
                    dq_min = float(np.min(delta))
                    curves.extend(zip([batch] * len(delta), [cell_id] * len(delta), voltage, delta))
                cells.append({
                    "batch": batch, "cell_id": cell_id, "source_index": index + 1,
                    "cycle_life": life, "observed_last_cycle": int(cycle[-1]),
                    "observed_fraction": float(cycle[-1] / life) if np.isfinite(life) else np.nan,
                    "last_qd": float(qd[-1]), "endpoint_near_eol": endpoint_near_eol,
                    "policy": policy,
                    "c1": c1, "switch_soc": switch_soc, "c2": c2,
                    "qd_10": at_cycle(cycle, qd, 10), "qd_100": at_cycle(cycle, qd, 100),
                    "early_qd_slope": slope,
                    "ir_10": at_cycle(cycle, ir, 10), "ir_100": at_cycle(cycle, ir, 100),
                    "early_tavg_mean": float(np.mean(tavg[early])) if early.any() else np.nan,
                    "early_charge_time_mean": float(np.mean(charge_time[early])) if early.any() else np.nan,
                    "charge_current_10": current_means[10],
                    "charge_current_100": current_means[100],
                    "ln_dq_variance": float(np.log(dq_variance)) if dq_variance > 0 else np.nan,
                    "dq_min": dq_min,
                    "early_fade": early_fade, "late_fade": late_fade,
                    "knee_fraction": knee,
                    "early_qd_slope_clean": slope_clean,
                    "early_charge_time_clean": charge_time_clean,
                })
                trajectories.extend(zip([batch] * int(usable.sum()),
                                        [cell_id] * int(usable.sum()),
                                        cycle[usable], qd[usable]))
            print(batch, "raw cells:", len(source["cycle_life"]))
    OUT.mkdir(parents=True, exist_ok=True)
    cell_frame = pd.DataFrame(cells)
    cell_frame.to_csv(OUT / "cells.csv", index=False)
    pd.DataFrame(trajectories, columns=["batch", "cell_id", "cycle", "qd"]).to_csv(
        OUT / "qd_trajectories.csv", index=False
    )
    pd.DataFrame(curves, columns=["batch", "cell_id", "voltage", "delta_q"]).to_csv(
        OUT / "delta_q_curves.csv", index=False
    )
    print(cell_frame.groupby("batch").agg(
        raw_cells=("cell_id", "size"),
        labeled=("cycle_life", "count"),
        near_eol=("endpoint_near_eol", "sum"),
    ).to_string())
    print("Saved:", OUT)


if __name__ == "__main__":
    main()
