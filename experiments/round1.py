"""1차: 학습 데이터 구성·타깃 변환·가중치 등 저비용 레버."""
import lightgbm as lgb
import numpy as np

from harness import X, df, evaluate, fmt, T

BASE = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20, subsample=0.8,
            subsample_freq=1, colsample_bytree=0.8, random_state=42, verbose=-1, objective="l1")


def lgbm(params=None, resid=None, weight=None):
    def f(Xtr, ytr, Xte, meta):
        p = {**BASE, **(params or {})}
        off_tr = Xtr[resid].fillna(ytr.median()).to_numpy() if resid else 0
        off_te = Xte[resid].fillna(ytr.median()).to_numpy() if resid else 0
        w = weight(meta) if weight else None
        m = lgb.LGBMRegressor(**p).fit(Xtr, ytr - off_tr, sample_weight=w)
        return m.predict(Xte) + off_te
    return f


def recency(half_life_days):
    def w(meta):
        age = (meta["idx_tr"].max() - meta["idx_tr"]).days
        return 0.5 ** (np.asarray(age) / half_life_days)
    return w


def copy_w(cw):
    return lambda meta: np.where(meta["copy_tr"], cw, 1.0)


exps = {
    "base_l1": dict(f=lgbm()),
    "l2": dict(f=lgbm({"objective": "regression"})),
    "huber": dict(f=lgbm({"objective": "huber", "alpha": 15})),
    "resid_lag168": dict(f=lambda *a: lgbm(resid=None)(*a)),
    "keep_copies": dict(f=lgbm(), keep=True),
    "copies_w0.3": dict(f=lgbm(weight=copy_w(0.3)), keep=True),
    "recency_60d": dict(f=lgbm(weight=recency(60))),
    "recency_30d": dict(f=lgbm(weight=recency(30))),
}
# 잔차 타깃은 타깃별 시차 열이 달라 별도 처리
def resid_target(col_power, col_peak):
    def f(Xtr, ytr, Xte, meta):
        col = col_power if meta["target"] == "power" else col_peak
        return lgbm(resid=col)(Xtr, ytr, Xte, meta)
    return f


exps["resid_lag168"] = dict(f=resid_target("power_lag168", "peak_lag168"))
exps["resid_oplag"] = dict(f=resid_target("op_power_lag", "op_peak_lag"))

if __name__ == "__main__":
    for name, e in exps.items():
        s, r = evaluate(e["f"], keep_copies=e.get("keep", False))
        print(f"{name:14s}", fmt(s, r), flush=True)
