"""Train on Batch 1, validate model choice on Batch 3, test once on Batch 2.

The assignment fixes training (Batch 1) and test (Batch 2) data and the paper GAP.
Candidate models, the feature adoption rule and the target comparison follow the DAY 1
report (모델 설계 전략 §3), including "if batch-wise validation error does not drop,
choose the simpler model" — Batch 3 is used as that batch-level validation set.
Run from the project root after ``python src/preprocess.py``:
    python src/train.py
Writes every table under ``results/``. Batch 2 is predicted only after the final model
has been fixed.
"""

from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import ALL_FEATURES, BASE, GROUPS, TARGET, load_cells, split  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 0
CV = RepeatedKFold(n_splits=4, n_repeats=25, random_state=SEED)
TARGETS = ("raw", "log")

# Severson et al. (2019) Table 1 (RMSE in cycles, mean percent error in %).
# 'variance' is the same model form as our baseline: linear in log Var[ΔQ100-10(V)].
PAPER = pd.DataFrame([
    ("variance", 103, 138, 196, 14.1, 14.7, 11.4),
    ("discharge", 76, 91, 173, 9.8, 13.0, 8.6),
    ("full", 51, 118, 214, 5.6, 14.1, 10.7),
], columns=["paper_model", "paper_rmse_train", "paper_rmse_primary", "paper_rmse_secondary",
            "paper_mape_train", "paper_mape_primary", "paper_mape_secondary"])


def regressor(model, target):
    """Imputation and scaling live inside the pipeline, so they are fit on training folds only.

    With target='log' the model learns log(cycle_life); predictions are mapped back to
    cycles before any error is computed.
    """
    pipe = make_pipeline(SimpleImputer(strategy="median"), StandardScaler(), model)
    if target == "raw":
        return pipe
    return TransformedTargetRegressor(regressor=pipe, func=np.log, inverse_func=np.exp)


def make_model(name):
    return {
        "Baseline (train median)": lambda: DummyRegressor(strategy="median"),
        "Linear (ln Var ΔQ)": LinearRegression,
        "Ridge (extended)": lambda: RidgeCV(alphas=np.logspace(-3, 3, 25)),
        "Random Forest": lambda: RandomForestRegressor(
            n_estimators=500, max_depth=3, min_samples_leaf=3, random_state=SEED),
    }[name]()


COMPLEXITY = {"Baseline (train median)": 0, "Linear (ln Var ΔQ)": 1,
              "Ridge (extended)": 2, "Random Forest": 3}


def metrics(y, pred):
    err = pred - y
    return {"n": len(y), "rmse": float(np.sqrt(np.mean(err ** 2))),
            "mae": float(np.mean(np.abs(err))),
            "mape": float(np.mean(np.abs(err) / y) * 100)}


def cross_validate(features, estimator, data):
    """Repeated 4-fold CV on cells; scores are the mean over 25 repeats."""
    x, y = data[features], data[TARGET].to_numpy()
    folds = list(CV.split(x))
    k = len(folds) // CV.n_repeats
    scores = []
    for repeat in range(CV.n_repeats):
        pred = np.empty_like(y, dtype=float)
        for fit_idx, val_idx in folds[repeat * k:(repeat + 1) * k]:
            pred[val_idx] = clone(estimator).fit(x.iloc[fit_idx], y[fit_idx]).predict(x.iloc[val_idx])
        scores.append(metrics(y, pred))
    scores = pd.DataFrame(scores)
    return {"cv_rmse": scores.rmse.mean(), "cv_rmse_se": scores.rmse.std(ddof=1) / np.sqrt(len(scores)),
            "cv_mae": scores.mae.mean(), "cv_mape": scores.mape.mean()}


def forward_groups(train, target, log):
    """Report rule: start from the ln Var ΔQ baseline, adopt a feature group only if it lowers CV RMSE."""
    chosen = []
    current = cross_validate(BASE, regressor(make_model("Ridge (extended)"), target), train)
    log.append({"target": target, "step": 0, "tried": "baseline", "features": ", ".join(BASE),
                "cv_rmse": current["cv_rmse"], "adopted": True})
    remaining = list(GROUPS)
    step = 0
    while remaining:
        step += 1
        trials = {}
        for group in remaining:
            features = BASE + [f for g in chosen + [group] for f in GROUPS[g]]
            trials[group] = cross_validate(features, regressor(make_model("Ridge (extended)"), target), train)
        best = min(trials, key=lambda g: trials[g]["cv_rmse"])
        for group, score in trials.items():
            log.append({"target": target, "step": step, "tried": group,
                        "features": ", ".join(BASE + [f for g in chosen + [group] for f in GROUPS[g]]),
                        "cv_rmse": score["cv_rmse"],
                        "adopted": group == best and score["cv_rmse"] < current["cv_rmse"]})
        if trials[best]["cv_rmse"] >= current["cv_rmse"]:
            break
        chosen.append(best)
        remaining.remove(best)
        current = trials[best]
    return BASE + [f for g in chosen for f in GROUPS[g]], current


def bootstrap_ci(y, pred, n=2000):
    rng = np.random.default_rng(SEED)
    idx = rng.integers(0, len(y), size=(n, len(y)))
    rmse = np.sqrt(np.mean((pred[idx] - y[idx]) ** 2, axis=1))
    mape = np.mean(np.abs(pred[idx] - y[idx]) / y[idx], axis=1) * 100
    return np.percentile(rmse, [2.5, 97.5]), np.percentile(mape, [2.5, 97.5])


def weights(name, features, model):
    pipe = model.regressor_ if isinstance(model, TransformedTargetRegressor) else model
    last = pipe[-1]
    if hasattr(last, "coef_"):
        values = np.ravel(last.coef_)
    elif hasattr(last, "feature_importances_"):
        values = last.feature_importances_
    else:
        return []
    return [{"model": name, "feature": f, "weight": float(w)} for f, w in zip(features, values)]


def main():
    RESULTS.mkdir(exist_ok=True)
    cells = load_cells()
    data = split(cells)  # train = Batch 1, test = Batch 2, secondary = Batch 3 (validation)
    train = data["train"]
    print({k: len(v) for k, v in data.items()})

    # 1) Model, feature and target selection on Batch 1 only.
    cv_rows, selection_log, feature_sets = [], [], {}
    for name in COMPLEXITY:
        for target in (("raw",) if name.startswith("Baseline") else TARGETS):
            if name == "Ridge (extended)":
                features, score = forward_groups(train, target, selection_log)
            else:
                features = ALL_FEATURES if name == "Random Forest" else BASE
                score = cross_validate(features, regressor(make_model(name), target), train)
            feature_sets[(name, target)] = features
            cv_rows.append({"model": name, "target": target, "complexity": COMPLEXITY[name],
                            "n_features": len(features), "features": ", ".join(features), **score})
            print(f"CV {name:24s} {target:3s} RMSE={score['cv_rmse']:.1f}  ({len(features)} features)")
    cv = pd.DataFrame(cv_rows)
    cv["chosen_target"] = cv.groupby("model")["cv_rmse"].transform("min") == cv["cv_rmse"]
    cv.round(3).to_csv(RESULTS / "cv_batch1.csv", index=False)
    pd.DataFrame(selection_log).round(3).to_csv(RESULTS / "ridge_feature_selection.csv", index=False)
    best = cv[cv.chosen_target].set_index("model")

    def fit(name, frame):
        target = best.loc[name, "target"]
        features = feature_sets[(name, target)]
        return features, clone(regressor(make_model(name), target)).fit(frame[features], frame[TARGET])

    # Report: "if batch-wise validation error does not drop, choose the simpler model".
    # Walk up in complexity from the ln Var ΔQ baseline; a more complex model replaces the
    # current choice only if it lowers RMSE on the held-out validation batch (Batch 3).
    validation = data["secondary"]
    val_rows = []
    for name in COMPLEXITY:
        features, model = fit(name, train)
        val_rows.append({"model": name, "complexity": COMPLEXITY[name],
                         **metrics(validation[TARGET].to_numpy(), model.predict(validation[features]))})
    val = pd.DataFrame(val_rows).set_index("model")
    final = "Linear (ln Var ΔQ)"
    for name in val.index[val.complexity > COMPLEXITY[final]]:
        if val.loc[name, "rmse"] < val.loc[final, "rmse"]:
            final = name
    cv_only = best.drop(index="Baseline (train median)").cv_rmse.idxmin()
    val.assign(cv_rmse=best.cv_rmse, final=val.index == final,
               lowest_batch1_cv=val.index == cv_only).round(2).to_csv(RESULTS / "model_selection.csv")
    print("Lowest Batch 1 CV:", cv_only, "| Final (Batch 3 validation):", final)

    # 2) Fit on all of Batch 1, evaluate once on Batch 1/2/3.
    perf, preds, coefs, models = [], [], [], {}
    for name in COMPLEXITY:
        features, model = fit(name, train)
        models[name] = (features, model)
        coefs += weights(name, features, model)
        for part, frame in data.items():
            y, pred = frame[TARGET].to_numpy(), model.predict(frame[features])
            row = {"model": name, "final": name == final, "target": best.loc[name, "target"],
                   "split": part, "batch": frame["batch"].iloc[0], **metrics(y, pred)}
            if part != "train":
                (r_lo, r_hi), (m_lo, m_hi) = bootstrap_ci(y, pred)
                row.update(rmse_ci_low=r_lo, rmse_ci_high=r_hi, mape_ci_low=m_lo, mape_ci_high=m_hi)
            perf.append(row)
            preds.append(pd.DataFrame({
                "model": name, "split": part, "cell_id": frame["cell_id"], "batch": frame["batch"],
                "policy": frame["policy"], "newstructure": frame["newstructure"],
                "cycle_life": y, "predicted": pred, "ln_dq_variance": frame["ln_dq_variance"]}))
    perf = pd.DataFrame(perf).merge(
        best.reset_index()[["model", "cv_rmse", "cv_rmse_se", "cv_mae", "cv_mape"]], on="model")
    perf.round(2).to_csv(RESULTS / "model_performance.csv", index=False)
    predictions = pd.concat(preds, ignore_index=True)
    predictions["residual"] = predictions["predicted"] - predictions["cycle_life"]
    predictions["abs_pct_error"] = predictions["residual"].abs() / predictions["cycle_life"] * 100
    predictions.round(3).to_csv(RESULTS / "predictions.csv", index=False)
    pd.DataFrame(coefs).round(5).to_csv(RESULTS / "model_coefficients.csv", index=False)

    # 3) GAP against the paper (assignment): Target = paper Table 1, Test = our Batch 2.
    #    Same-form comparison uses the paper 'variance' model; the paper's best
    #    primary-test model ('discharge') is reported as the upper target.
    ours = perf.pivot_table(index="model", columns="split", values=["rmse", "mape"])
    ours.columns = [f"ours_{m}_{s}" for m, s in ours.columns]
    gap = ours.reset_index().merge(PAPER, how="cross")
    for metric in ("rmse", "mape"):
        gap[f"{metric}_gap_test_vs_primary"] = gap[f"ours_{metric}_test"] - gap[f"paper_{metric}_primary"]
        gap[f"{metric}_gap_b3_vs_secondary"] = gap[f"ours_{metric}_secondary"] - gap[f"paper_{metric}_secondary"]
    gap.round(1).to_csv(RESULTS / "paper_gap.csv", index=False)

    # 4) Report: batch-wise errors and short/long-life residuals on the test batch.
    test_pred = predictions[predictions.split == "test"]
    train_min, train_max = train[TARGET].min(), train[TARGET].max()
    groups = {
        "all": lambda f: f.cycle_life > 0,
        "short life (< 500)": lambda f: f.cycle_life < 500,
        "life >= 500": lambda f: f.cycle_life >= 500,
        f"below train range (< {train_min:.0f})": lambda f: f.cycle_life < train_min,
        f"inside train range ({train_min:.0f}-{train_max:.0f})": lambda f: f.cycle_life.between(train_min, train_max),
        "newstructure": lambda f: f.newstructure,
        "original structure": lambda f: ~f.newstructure,
    }
    breakdown = []
    for name, frame in test_pred.groupby("model", sort=False):
        for label, rule in groups.items():
            part = frame[rule(frame)]
            if len(part):
                breakdown.append({"model": name, "subgroup": label,
                                  "mean_residual": part.residual.mean(),
                                  "over_predicted": int((part.residual > 0).sum()),
                                  **metrics(part.cycle_life.to_numpy(), part.predicted.to_numpy())})
    pd.DataFrame(breakdown).round(2).to_csv(RESULTS / "batch2_error_breakdown.csv", index=False)

    # 5) Report: leave-one-batch-out — train on two batches, evaluate the held-out batch.
    #    Model form, features and target stay as chosen on Batch 1.
    lobo = []
    for name in dict.fromkeys(["Linear (ln Var ΔQ)", final]):
        for held_out in ("Batch 1", "Batch 2", "Batch 3"):
            fit_on, test = cells[cells.batch != held_out], cells[cells.batch == held_out]
            features, model = fit(name, fit_on)
            lobo.append({"model": name, "held_out": held_out, "n_train": len(fit_on),
                         **metrics(test[TARGET].to_numpy(), model.predict(test[features]))})
    pd.DataFrame(lobo).round(2).to_csv(RESULTS / "leave_one_batch_out.csv", index=False)

    # 6) Improvement check: k Batch 2 cells run to EOL recalibrate the log-scale offset;
    #    error is measured on the remaining Batch 2 cells only.
    rng = np.random.default_rng(SEED)
    recal = []
    for name in dict.fromkeys(["Linear (ln Var ΔQ)", final]):
        frame = test_pred[test_pred.model == name]
        y, pred = frame.cycle_life.to_numpy(), frame.predicted.to_numpy()
        for k in (0, 3, 5, 10):
            runs = []
            for _ in range(500 if k else 1):
                cal = rng.choice(len(y), size=k, replace=False)
                rest = np.setdiff1d(np.arange(len(y)), cal)
                offset = np.mean(np.log(y[cal]) - np.log(pred[cal])) if k else 0.0
                runs.append(metrics(y[rest], pred[rest] * np.exp(offset)))
            runs = pd.DataFrame(runs)
            recal.append({"model": name, "calibration_cells": k, "eval_cells": len(y) - k,
                          "rmse_mean": runs.rmse.mean(), "rmse_p90": runs.rmse.quantile(0.9),
                          "mape_mean": runs.mape.mean()})
    pd.DataFrame(recal).round(2).to_csv(RESULTS / "batch2_recalibration.csv", index=False)

    show = perf.pivot_table(index="model", columns="split", values="rmse").round(1)
    print(show.loc[list(COMPLEXITY), ["train", "test", "secondary"]].to_string())
    print("Saved:", RESULTS)


if __name__ == "__main__":
    main()
