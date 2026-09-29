"""2차: 가동일 LightGBM 하이퍼파라미터 탐색 (Optuna, 2개 검증 fold 평균 MAE)."""
import json

import lightgbm as lgb
import numpy as np
import optuna

from harness import evaluate, fmt

optuna.logging.set_verbosity(optuna.logging.WARNING)
RESID = {"none": None, "lag168": ("power_lag168", "peak_lag168"), "oplag": ("op_power_lag", "op_peak_lag")}


def make(params, cw, resid):
    def f(Xtr, ytr, Xte, meta):
        cols = RESID[resid]
        col = None if cols is None else (cols[0] if meta["target"] == "power" else cols[1])
        o_tr = Xtr[col].fillna(ytr.median()).to_numpy() if col else 0
        o_te = Xte[col].fillna(ytr.median()).to_numpy() if col else 0
        w = np.where(meta["copy_tr"], cw, 1.0)
        m = lgb.LGBMRegressor(**params).fit(Xtr, ytr - o_tr, sample_weight=w)
        return m.predict(Xte) + o_te
    return f


def objective(trial):
    params = dict(
        objective=trial.suggest_categorical("objective", ["huber", "l1", "fair"]),
        n_estimators=trial.suggest_int("n_estimators", 200, 1500, step=100),
        learning_rate=trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
        num_leaves=trial.suggest_int("num_leaves", 7, 63),
        min_child_samples=trial.suggest_int("min_child_samples", 5, 60),
        subsample=trial.suggest_float("subsample", 0.5, 1.0), subsample_freq=1,
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.3, 1.0),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 30, log=True),
        random_state=42, verbose=-1, n_jobs=4,
    )
    if params["objective"] == "huber":
        params["alpha"] = trial.suggest_float("huber_alpha", 3, 40, log=True)
    if params["objective"] == "fair":
        params["fair_c"] = trial.suggest_float("fair_c", 1, 30, log=True)
    cw = trial.suggest_float("copy_weight", 0.0, 1.0)
    resid = trial.suggest_categorical("resid", list(RESID))
    s, r = evaluate(make(params, cw, resid), keep_copies=True)
    trial.set_user_attr("folds", r)
    return s


if __name__ == "__main__":
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=0))
    study.enqueue_trial(dict(objective="huber", n_estimators=600, learning_rate=0.03, num_leaves=31,
                             min_child_samples=20, subsample=0.8, colsample_bytree=0.8, reg_lambda=0.001,
                             huber_alpha=15, copy_weight=0.3, resid="oplag"))
    for i in range(80):
        study.optimize(objective, n_trials=1)
        t = study.trials[-1]
        print(f"#{i:02d} {t.value:.3f} best {study.best_value:.3f}", flush=True)
    best = study.best_trial
    print("BEST", best.value, best.params, best.user_attrs["folds"])
    top = sorted([t for t in study.trials if t.value is not None], key=lambda t: t.value)[:10]
    json.dump([dict(value=t.value, params=t.params, folds=t.user_attrs["folds"]) for t in top],
              open("tune_lgbm_top10.json", "w"), indent=1)
