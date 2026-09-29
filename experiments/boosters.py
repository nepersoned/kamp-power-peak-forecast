"""가동일 모델 후보: LightGBM / XGBoost / CatBoost / ExtraTrees. 공통: 복제일 가중치, 직전가동일 잔차 타깃."""
import lightgbm as lgb
import numpy as np
import xgboost as xgb
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor

RESID = {"none": None, "lag168": ("power_lag168", "peak_lag168"), "oplag": ("op_power_lag", "op_peak_lag")}


def _offset(Xtr, ytr, Xte, meta, resid):
    cols = RESID[resid]
    if cols is None:
        return 0, 0
    col = cols[0] if meta["target"] == "power" else cols[1]
    fill = ytr.median()
    return Xtr[col].fillna(fill).to_numpy(), Xte[col].fillna(fill).to_numpy()


def wrap(fit_predict, cw=0.3, resid="oplag"):
    def f(Xtr, ytr, Xte, meta):
        o_tr, o_te = _offset(Xtr, ytr, Xte, meta, resid)
        w = np.where(meta["copy_tr"], cw, 1.0)
        return fit_predict(Xtr, ytr - o_tr, Xte, w) + o_te
    return f


def lgbm_fp(params):
    def fp(Xtr, y, Xte, w):
        return lgb.LGBMRegressor(**params).fit(Xtr, y, sample_weight=w).predict(Xte)
    return fp


def xgb_fp(params):
    def fp(Xtr, y, Xte, w):
        return xgb.XGBRegressor(**params).fit(Xtr, y, sample_weight=w).predict(Xte)
    return fp


def cat_fp(params):
    def fp(Xtr, y, Xte, w):
        return CatBoostRegressor(**params).fit(Xtr, y, sample_weight=w).predict(Xte)
    return fp


def et_fp(params):
    def fp(Xtr, y, Xte, w):
        fill = Xtr.median()
        return ExtraTreesRegressor(**params).fit(Xtr.fillna(fill), y, sample_weight=w).predict(Xte.fillna(fill))
    return fp


LGB0 = dict(objective="huber", alpha=15, n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20,
            subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1, n_jobs=4)
XGB0 = dict(objective="reg:pseudohubererror", huber_slope=15, n_estimators=600, learning_rate=0.03, max_depth=6,
            subsample=0.8, colsample_bytree=0.8, min_child_weight=5, random_state=42, n_jobs=4)
CAT0 = dict(loss_function="Huber:delta=15", iterations=1000, learning_rate=0.05, depth=6, random_seed=42,
            verbose=0, thread_count=4)
ET0 = dict(n_estimators=500, min_samples_leaf=3, max_features=0.5, n_jobs=4, random_state=42)

if __name__ == "__main__":
    from harness import evaluate, fmt
    for name, fp in [("lgbm", lgbm_fp(LGB0)), ("xgb", xgb_fp(XGB0)), ("cat", cat_fp(CAT0)), ("et", et_fp(ET0)),
                     ("cat_mae", cat_fp({**CAT0, "loss_function": "MAE"})), ("xgb_mae", xgb_fp({**XGB0, "objective": "reg:absoluteerror"}))]:
        s, r = evaluate(wrap(fp), keep_copies=True)
        print(f"{name:8s}", fmt(s, r), flush=True)
