"""Batch 1 policy-grouped CV and hold-out select the model; Batch 2 / Batch 3 are tested once.

Protocol (assignment DAY 2, candidates per DAY 1 report 모델 설계 전략 §3):
  1. Batch 1 is split by charging policy into Dev (~75%) and Hold-out (~25%, seed 0).
  2. Train = grouped 5-fold CV (policy groups, 10 repeats) on Dev; target (raw/log) and Ridge
     feature groups are chosen by Dev CV MAPE.
  3. Valid = candidates fit on Dev, scored on the Hold-out. Final model = simplest candidate
     within 1 %p of the best Valid MAPE (baseline excluded).
  4. Final model is refit on all of Batch 1, then predicts Batch 2 (mandatory) and Batch 3
     (optional; all 44 cells and 40 cells without paper-excluded outliers) once.
Batch 2 / Batch 3 never influence selection. Run from the project root after
``python src/preprocess.py``: ``python src/train.py``. Writes every table under ``results/``.
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
from sklearn.model_selection import GroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from features import (ALL_FEATURES, BASE, BATCH3_OUTLIERS, GROUPS, TARGET, holdout_split,  # noqa: E402
                      load_cells, split)

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SEED = 0
CV_REPEATS = 10
TOLERANCE_PP = 1.0   # Valid MAPE within this many %p of the best counts as tied -> simpler wins
TARGET_MAPE = 9.1    # assignment-given paper performance (Regression, MAPE %)
TARGETS = ("raw", "log")

# Severson et al. (2019) Table 1, kept only as a reference next to the assignment Target.
PAPER = pd.DataFrame([
    ("variance", 103, 138, 196, 14.1, 14.7, 11.4),
    ("discharge", 76, 91, 173, 9.8, 13.0, 8.6),
    ("full", 51, 118, 214, 5.6, 14.1, 10.7),
], columns=["ref_paper_model", "ref_rmse_train", "ref_rmse_primary", "ref_rmse_secondary",
            "ref_mape_train", "ref_mape_primary", "ref_mape_secondary"])


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
    """Policy-grouped 5-fold CV, repeated with shuffled groups; scores are the mean over repeats."""
    x, y = data[features], data[TARGET].to_numpy()
    scores = []
    for repeat in range(CV_REPEATS):
        pred = np.empty_like(y, dtype=float)
        for fit_idx, val_idx in GroupKFold(5, shuffle=True, random_state=repeat).split(x, groups=data["policy"]):
            pred[val_idx] = clone(estimator).fit(x.iloc[fit_idx], y[fit_idx]).predict(x.iloc[val_idx])
        scores.append(metrics(y, pred))
    scores = pd.DataFrame(scores)
    return {"cv_mape": scores.mape.mean(), "cv_mape_sd": scores.mape.std(ddof=1),
            "cv_rmse": scores.rmse.mean(), "cv_mae": scores.mae.mean()}


def forward_groups(train, target, log):
    """Start from the ln Var ΔQ baseline; adopt a feature group only if it lowers Dev CV MAPE."""
    chosen = []
    current = cross_validate(BASE, regressor(make_model("Ridge (extended)"), target), train)
    log.append({"target": target, "step": 0, "tried": "baseline", "features": ", ".join(BASE),
                "cv_mape": current["cv_mape"], "adopted": True})
    remaining = list(GROUPS)
    step = 0
    while remaining:
        step += 1
        trials = {}
        for group in remaining:
            features = BASE + [f for g in chosen + [group] for f in GROUPS[g]]
            trials[group] = cross_validate(features, regressor(make_model("Ridge (extended)"), target), train)
        best = min(trials, key=lambda g: trials[g]["cv_mape"])
        for group, score in trials.items():
            log.append({"target": target, "step": step, "tried": group,
                        "features": ", ".join(BASE + [f for g in chosen + [group] for f in GROUPS[g]]),
                        "cv_mape": score["cv_mape"],
                        "adopted": group == best and score["cv_mape"] < current["cv_mape"]})
        if trials[best]["cv_mape"] >= current["cv_mape"]:
            break
        chosen.append(best)
        remaining.remove(best)
        current = trials[best]
    return BASE + [f for g in chosen for f in GROUPS[g]], current


def select(dev, holdout, log=None):
    """Dev CV picks target/features per model; Hold-out Valid MAPE picks the final model.

    Returns (table, fits) where fits[name] = (features, target, model fit on Dev).
    """
    log = [] if log is None else log
    rows, features_by = [], {}
    for name in COMPLEXITY:
        for target in (("raw",) if name.startswith("Baseline") else TARGETS):
            if name == "Ridge (extended)":
                features, score = forward_groups(dev, target, log)
            else:
                features = ALL_FEATURES if name == "Random Forest" else BASE
                score = cross_validate(features, regressor(make_model(name), target), dev)
            features_by[(name, target)] = features
            rows.append({"model": name, "target": target, "complexity": COMPLEXITY[name],
                         "n_features": len(features), "features": ", ".join(features), **score})
    cv = pd.DataFrame(rows)
    cv["chosen_target"] = cv.groupby("model")["cv_mape"].transform("min") == cv["cv_mape"]
    table, fits = [], {}
    for _, row in cv[cv.chosen_target].iterrows():
        features = features_by[(row.model, row.target)]
        model = clone(regressor(make_model(row.model), row.target)).fit(dev[features], dev[TARGET])
        fits[row.model] = (features, row.target, model)
        valid = metrics(holdout[TARGET].to_numpy(), model.predict(holdout[features]))
        table.append({"model": row.model, "complexity": row.complexity, "target": row.target,
                      "n_features": row.n_features, "features": row.features,
                      "train_cv_mape": row.cv_mape, "train_cv_mape_sd": row.cv_mape_sd,
                      "train_cv_rmse": row.cv_rmse, "train_cv_mae": row.cv_mae,
                      "valid_mape": valid["mape"], "valid_rmse": valid["rmse"], "valid_mae": valid["mae"]})
    table = pd.DataFrame(table)
    cand = table[table.complexity > 0]
    table["within_1pp"] = (table.complexity > 0) & (table.valid_mape <= cand.valid_mape.min() + TOLERANCE_PP)
    tied = table[table.within_1pp]
    final = tied.loc[tied.complexity.idxmin(), "model"]
    table["final"] = table.model == final
    return cv, table, fits, final


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
    data = split(cells)  # train = Batch 1, test = Batch 2, secondary = Batch 3
    train = data["train"]
    dev, holdout, holdout_policies = holdout_split(train, seed=SEED)
    assert not set(dev.policy) & set(holdout.policy)
    print({k: len(v) for k, v in data.items()}, "| dev", len(dev), "holdout", len(holdout))
    pd.concat([dev.assign(role="dev"), holdout.assign(role="holdout")])[
        ["cell_id", "policy", "cycle_life", "role", "life_bin"]].sort_values(
        ["role", "cycle_life"]).to_csv(RESULTS / "holdout_split.csv", index=False)

    # 1) Selection on Batch 1 only: Dev grouped CV (target, Ridge features), Hold-out (final model).
    selection_log = []
    cv, table, fits, final = select(dev, holdout, selection_log)
    cv.round(3).to_csv(RESULTS / "cv_batch1.csv", index=False)
    pd.DataFrame(selection_log).round(3).to_csv(RESULTS / "ridge_feature_selection.csv", index=False)
    table.round(3).to_csv(RESULTS / "model_selection.csv", index=False)
    print(table[["model", "target", "n_features", "train_cv_mape", "valid_mape", "within_1pp", "final"]]
          .round(2).to_string(index=False))
    print("Hold-out policies:", holdout_policies, "| Final (Batch 1 only):", final)

    # 2) Final choice is fixed. Refit every candidate on all of Batch 1 (same features/target),
    #    then predict Batch 2 and Batch 3. Only `final` feeds the headline table.
    best = table.set_index("model")
    clean3 = data["secondary"][~data["secondary"].cell_id.isin(BATCH3_OUTLIERS)].reset_index(drop=True)
    parts = {"train": train, "test": data["test"], "secondary": data["secondary"], "secondary_clean": clean3}
    perf, preds, coefs, models = [], [], [], {}
    for name in COMPLEXITY:
        features, target, dev_model = fits[name]
        model = clone(regressor(make_model(name), target)).fit(train[features], train[TARGET])
        models[name] = (features, target, model)
        coefs += weights(name, features, model)
        base = {"model": name, "final": name == final, "target": target}
        t = best.loc[name]
        perf.append({**base, "stage": "train_cv", "n": len(dev), "mape": t.train_cv_mape,
                     "rmse": t.train_cv_rmse, "mae": t.train_cv_mae})
        hold = metrics(holdout[TARGET].to_numpy(), dev_model.predict(holdout[features]))
        perf.append({**base, "stage": "valid", **hold})
        preds.append(pd.DataFrame({
            "model": name, "split": "holdout", "cell_id": holdout["cell_id"], "batch": holdout["batch"],
            "policy": holdout["policy"], "newstructure": holdout["newstructure"],
            "cycle_life": holdout[TARGET], "predicted": dev_model.predict(holdout[features])}))
        for part, frame in parts.items():
            y, pred = frame[TARGET].to_numpy(), model.predict(frame[features])
            stage = {"train": "b1_full_fit", "test": "test_b2", "secondary": "test_b3_all",
                     "secondary_clean": "test_b3_clean"}[part]
            row = {**base, "stage": stage, **metrics(y, pred)}
            if part != "train":
                (r_lo, r_hi), (m_lo, m_hi) = bootstrap_ci(y, pred)
                row.update(rmse_ci_low=r_lo, rmse_ci_high=r_hi, mape_ci_low=m_lo, mape_ci_high=m_hi)
            perf.append(row)
            if part != "secondary_clean":
                preds.append(pd.DataFrame({
                    "model": name, "split": part, "cell_id": frame["cell_id"], "batch": frame["batch"],
                    "policy": frame["policy"], "newstructure": frame["newstructure"],
                    "cycle_life": y, "predicted": pred}))
    perf = pd.DataFrame(perf)
    perf.round(3).to_csv(RESULTS / "model_performance_all.csv", index=False)
    predictions = pd.concat(preds, ignore_index=True)
    predictions["ln_dq_variance"] = predictions["cell_id"].map(cells.set_index("cell_id")["ln_dq_variance"])
    predictions["batch3_outlier"] = predictions["cell_id"].isin(BATCH3_OUTLIERS)
    predictions["residual"] = predictions["predicted"] - predictions["cycle_life"]
    predictions["abs_pct_error"] = predictions["residual"].abs() / predictions["cycle_life"] * 100
    predictions.round(3).to_csv(RESULTS / "predictions.csv", index=False)
    pd.DataFrame(coefs).round(5).to_csv(RESULTS / "model_coefficients.csv", index=False)

    # 3) Headline tables in the assignment format. Gaps are signed so (+) = worse (MAPE is lower-is-better).
    mape = perf[perf.final].set_index("stage")["mape"]
    tr, va, b2 = mape.train_cv, mape.valid, mape.test_b2
    b3, b3_all = mape.test_b3_clean, mape.test_b3_all
    n3 = perf[perf.final & (perf.stage == "test_b3_clean")].n.iloc[0]
    report = pd.DataFrame([
        ("Regression", "Train (Batch 1 CV)", tr, f"Dev {len(dev)}셀, 정책 그룹 5-fold CV x{CV_REPEATS} 평균"),
        ("Regression", "Valid (Batch 1 Hold-out)", va, f"Hold-out {len(holdout)}셀 (정책 {len(holdout_policies)}개)"),
        ("Regression", "Test (Batch 2)", b2, f"{int(perf[perf.final & (perf.stage == 'test_b2')].n.iloc[0])}셀"),
        ("Regression", "Gap (Train-Valid)", va - tr, "(+) : 과적합 의심; Valid - Train"),
        ("Regression", "Gap (Valid-Test)", b2 - va, "(+) : 배치간 일반화 저하 의심; Test(B2) - Valid"),
        ("Regression", "Gap (Target-Test)", b2 - TARGET_MAPE, f"Target : 원논문 {TARGET_MAPE}% (과제 지정); Test(B2) - Target"),
        ("Batch 3", "Test (Batch 3)", b3, f"{int(n3)}셀(이상치 제외); 전체 44셀 = {b3_all:.1f}%"),
        ("Batch 3", "Gap (Batch2-Batch3)", b3 - b2,
         f"Test 성능 간 비교; Test(B3) - Test(B2); 전체 44셀 기준 {b3_all - b2:+.1f}"),
        ("Batch 3", "Gap (Target-Test)", b3 - TARGET_MAPE,
         f"Batch 3 기준, 원논문 {TARGET_MAPE}% 대비; Test(B3) - Target; 전체 44셀 기준 {b3_all - TARGET_MAPE:+.1f}"),
    ], columns=["section", "item", "mape", "note"])
    report.assign(mape=report.mape.round(1)).to_csv(RESULTS / "model_performance.csv", index=False)
    print("Final model:", final, best.loc[final, "target"], best.loc[final, "features"])
    print(report.assign(mape=report.mape.round(1)).to_string(index=False))

    # Reference comparison against the paper (final row = headline).
    wide = perf.pivot_table(index="model", columns="stage", values="mape")
    gap = wide[["train_cv", "valid", "test_b2", "test_b3_all", "test_b3_clean"]].add_prefix("ours_mape_").reset_index()
    gap.insert(1, "final", gap.model == final)
    gap["target_mape"] = TARGET_MAPE
    gap["gap_target_test_b2"] = gap.ours_mape_test_b2 - TARGET_MAPE
    gap["gap_target_test_b3_clean"] = gap.ours_mape_test_b3_clean - TARGET_MAPE
    gap["gap_target_test_b3_all"] = gap.ours_mape_test_b3_all - TARGET_MAPE
    gap = gap.merge(PAPER, how="cross")
    gap.round(1).to_csv(RESULTS / "paper_gap.csv", index=False)

    # 4) Batch 2 breakdown by subgroup (diagnostic only).
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

    # 5) Diagnostic (not used for selection): leave-one-batch-out with the Batch 1 features/target.
    def fit(name, frame):
        features, target, _ = fits[name]
        return features, clone(regressor(make_model(name), target)).fit(frame[features], frame[TARGET])

    lobo = []
    for name in dict.fromkeys(["Linear (ln Var ΔQ)", final]):
        for held_out in ("Batch 1", "Batch 2", "Batch 3"):
            fit_on, test = cells[cells.batch != held_out], cells[cells.batch == held_out]
            features, model = fit(name, fit_on)
            lobo.append({"model": name, "held_out": held_out, "n_train": len(fit_on),
                         **metrics(test[TARGET].to_numpy(), model.predict(test[features]))})
    pd.DataFrame(lobo).round(2).to_csv(RESULTS / "leave_one_batch_out.csv", index=False)

    # 6) Diagnostic: k Batch 2 cells run to EOL recalibrate the log-scale offset; error on the rest.
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

    # 7) Hold-out split sensitivity: rerun the whole selection for seeds 0-19 (headline stays seed 0).
    sens = []
    for seed in range(20):
        d, h, picked = holdout_split(train, seed=seed)
        _, tab, _, fin = select(d, h)
        row = tab[tab.final].iloc[0]
        sens.append({"seed": seed, "final_model": fin, "final_target": row.target,
                     "valid_mape": row.valid_mape, "train_cv_mape": row.train_cv_mape,
                     "holdout_cells": len(h), "holdout_policies": "; ".join(picked)})
    pd.DataFrame(sens).round(3).to_csv(RESULTS / "split_sensitivity.csv", index=False)
    print(pd.DataFrame(sens).final_model.value_counts().to_string())
    print("Saved:", RESULTS)


if __name__ == "__main__":
    main()
