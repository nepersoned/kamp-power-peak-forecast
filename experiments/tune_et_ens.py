"""ExtraTrees 튜닝 후, 튜닝된 LightGBM(시드 평균)과의 앙상블 가중치를 검증 fold로 결정."""
import json

import numpy as np
import optuna

from boosters import cat_fp, et_fp, lgbm_fp, wrap, CAT0
from harness import df, evaluate, fmt

optuna.logging.set_verbosity(optuna.logging.WARNING)
best_lgb = json.load(open("tune_lgbm_top10.json"))[0]["params"]
LGB_T = dict(objective=best_lgb["objective"], n_estimators=best_lgb["n_estimators"],
             learning_rate=best_lgb["learning_rate"], num_leaves=best_lgb["num_leaves"],
             min_child_samples=best_lgb["min_child_samples"], subsample=best_lgb["subsample"], subsample_freq=1,
             colsample_bytree=best_lgb["colsample_bytree"], reg_lambda=best_lgb["reg_lambda"], verbose=-1, n_jobs=4)
if best_lgb["objective"] == "huber":
    LGB_T["alpha"] = best_lgb["huber_alpha"]
if best_lgb["objective"] == "fair":
    LGB_T["fair_c"] = best_lgb["fair_c"]
RESID_T = best_lgb["resid"]
CW = best_lgb["copy_weight"]


def lgb_seeds(seeds=(42, 7, 2021)):
    def fp(Xtr, y, Xte, w):
        return np.mean([lgbm_fp({**LGB_T, "random_state": s})(Xtr, y, Xte, w) for s in seeds], 0)
    return fp


def et_obj(trial):
    p = dict(n_estimators=400, min_samples_leaf=trial.suggest_int("min_samples_leaf", 1, 20),
             max_features=trial.suggest_float("max_features", 0.2, 1.0), max_depth=trial.suggest_categorical("max_depth", [None, 12, 20]),
             n_jobs=8, random_state=42)
    cw = trial.suggest_float("copy_weight", 0.0, 1.0)
    return evaluate(wrap(et_fp(p), cw=cw), keep_copies=True)[0]


if __name__ == "__main__":
    st = optuna.create_study(sampler=optuna.samplers.TPESampler(seed=0))
    st.enqueue_trial(dict(min_samples_leaf=3, max_features=0.5, max_depth=None, copy_weight=0.3))
    st.optimize(et_obj, n_trials=25)
    bp = st.best_params
    print("ET best", round(st.best_value, 3), bp, flush=True)
    ET_T = dict(n_estimators=800, min_samples_leaf=bp["min_samples_leaf"], max_features=bp["max_features"],
                max_depth=bp["max_depth"], n_jobs=8, random_state=42)

    comps = {"lgbm_t": wrap(lgb_seeds(), cw=CW, resid=RESID_T), "et_t": wrap(et_fp(ET_T), cw=bp["copy_weight"]),
             "cat_mae": wrap(cat_fp({**CAT0, "loss_function": "MAE"}), cw=0.3)}
    P = {}
    for n, f in comps.items():
        s, r, P[n] = evaluate(f, keep_copies=True, return_preds=True)
        print(n, fmt(s, r), flush=True)
    keys = list(P["lgbm_t"])
    best = None
    for wl in np.arange(0, 1.01, 0.1):
        for we in np.arange(0, 1.01 - wl, 0.1):
            wc = 1 - wl - we
            ms = [np.abs(df[k[0]][P["lgbm_t"][k].index].to_numpy()
                         - (wl * P["lgbm_t"][k] + we * P["et_t"][k] + wc * P["cat_mae"][k]).to_numpy()).mean() for k in keys]
            s = float(np.mean(ms))
            if best is None or s < best[0]:
                best = (s, round(wl, 1), round(we, 1), round(wc, 1), [round(m, 2) for m in ms])
    print("BLEND best", best)
    json.dump(dict(lgbm=LGB_T, lgbm_copy_weight=CW, et=ET_T, et_copy_weight=bp["copy_weight"],
                   weights=dict(lgbm=best[1], et=best[2], cat_mae=best[3]), valid_score=best[0], folds=best[4]),
              open("final_config.json", "w"), indent=1, default=str)
