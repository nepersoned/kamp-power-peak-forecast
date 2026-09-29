"""예측 모델.

비교 대상 (동일 학습/검증 구간, 동일 피처):
  naive_24h    : 전일 같은 시각 값 (베이스라인)
  naive_168h   : 전주 같은 시각 값 (베이스라인)
  ridge        : 선형회귀 (단일 모델)
  lgbm         : LightGBM (단일 모델)
  regime_lgbm  : 운전 레짐 전환 (가동일 LightGBM-L1 / 휴무일 시간대별 중앙값)
  regime_ens   : 최종 제안 모델. regime_lgbm 에서
                 - 가동일 타깃을 '직전 가동일 동시각 값 대비 잔차'로 학습
                 - 증강 복제일을 버리지 않고 낮은 가중치로 활용
                 - Optuna 튜닝 LightGBM(3시드) 0.7 + ExtraTrees 0.2 + CatBoost(MAE) 0.1 앙상블
                 (하이퍼파라미터·가중치는 검증 fold 2개로만 결정: experiments/ 참조)

모든 모델: fit(X, y, hour, copy) / predict(X, hour).  copy = 증강 복제일 여부.
copy_weight 가 0인 모델은 복제일을 학습에서 제외.
"""
import json
from pathlib import Path

import lightgbm as lgb
import numpy as np
from catboost import CatBoostRegressor
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

SEED = 42
LGB_PARAMS = dict(n_estimators=600, learning_rate=0.03, num_leaves=31, min_child_samples=20,
                  subsample=0.8, subsample_freq=1, colsample_bytree=0.8, random_state=SEED, verbose=-1)
FINAL = json.loads((Path(__file__).parent / "final_config.json").read_text(encoding="utf-8"))


def operating(X):
    """생산계획상 가동일 여부. 계획 누락일(NaN)은 가동으로 간주(보수적)."""
    return X["plan_on_day"].fillna(1).to_numpy() == 1


def _drop_copies(X, y, hour, copy):
    if copy is None:
        return X, y, hour
    keep = ~np.asarray(copy)
    return X[keep], y[keep], hour[keep]


class Naive:
    def __init__(self, lag):
        self.lag = lag

    def fit(self, X, y, hour=None, copy=None):
        self.col = f"{'power' if y.name == 'power' else 'peak'}_lag{self.lag}"
        return self

    def predict(self, X, hour=None):
        return X[self.col].to_numpy()


class RidgeModel:
    def fit(self, X, y, hour=None, copy=None):
        X, y, _ = _drop_copies(X, y, y, copy)
        self.fill = X.median()
        self.m = make_pipeline(StandardScaler(), Ridge(alpha=3.0)).fit(X.fillna(self.fill), y)
        return self

    def predict(self, X, hour=None):
        return self.m.predict(X.fillna(self.fill))


class LGBM:
    def __init__(self, **kw):
        self.kw = {**LGB_PARAMS, **kw}

    def fit(self, X, y, hour=None, copy=None):
        X, y, _ = _drop_copies(X, y, y, copy)
        self.m = lgb.LGBMRegressor(**self.kw).fit(X, y)
        return self

    def predict(self, X, hour=None):
        return self.m.predict(X)


class RegimeModel:
    """운전 레짐 전환 모델.

    휴무일은 원본 학습 데이터에 13일뿐(전부 주말, 그중 4일은 기록되지 않은 특근)이라
    학습형 모델이 과적합됨 -> 이상치에 강건한 시간대별 중앙값 기저부하로 예측.
    """

    def __init__(self, on_model=None, off_q=0.5):
        self.on_model = on_model or LGBM(objective="l1")
        self.off_q = off_q

    def fit(self, X, y, hour, copy=None):
        on = operating(X)
        c = None if copy is None else np.asarray(copy)
        self.on_model.fit(X[on], y[on], hour[on], None if c is None else c[on])
        Xo, yo, ho = _drop_copies(X[~on], y[~on], hour[~on], None if c is None else c[~on])
        self.off_profile = yo.groupby(ho).quantile(self.off_q)
        return self

    def predict(self, X, hour):
        on = operating(X)
        out = np.empty(len(X))
        out[on] = self.on_model.predict(X[on], hour[on])
        out[~on] = hour[~on].map(self.off_profile).to_numpy()
        return out


class EnsembleOn:
    """가동일 앙상블: 직전가동일 잔차 타깃 + 복제일 가중 학습."""

    def __init__(self, cfg=FINAL, seeds=(42, 7, 2021), lgb_override=None):
        self.cfg, self.seeds = cfg, seeds
        self.lgb_params = {**cfg["lgbm"], **(lgb_override or {})}

    def _offset(self, X):
        return X[self.col].fillna(self.fill).to_numpy()

    def fit(self, X, y, hour=None, copy=None):
        self.col = "op_power_lag" if y.name == "power" else "op_peak_lag"
        self.fill = float(y.median())
        r = y.to_numpy() - self._offset(X)
        c = np.zeros(len(X), bool) if copy is None else np.asarray(copy)
        w_l = np.where(c, self.cfg["lgbm_copy_weight"], 1.0)
        w_e = np.where(c, self.cfg["et_copy_weight"], 1.0)
        w_c = np.where(c, 0.3, 1.0)
        self.lgbs = [lgb.LGBMRegressor(**{**self.lgb_params, "random_state": s}).fit(X, r, sample_weight=w_l)
                     for s in self.seeds]
        self.xfill = X.median()
        self.et = ExtraTreesRegressor(**self.cfg["et"]).fit(X.fillna(self.xfill), r, sample_weight=w_e)
        self.cat = CatBoostRegressor(loss_function="MAE", iterations=1000, learning_rate=0.05, depth=6,
                                     random_seed=SEED, verbose=0, thread_count=4).fit(X, r, sample_weight=w_c)
        return self

    def predict_components(self, X):
        return dict(lgbm=np.mean([m.predict(X) for m in self.lgbs], 0),
                    et=self.et.predict(X.fillna(self.xfill)), cat_mae=self.cat.predict(X))

    def predict(self, X, hour=None):
        comp = self.predict_components(X)
        w = self.cfg["weights"]
        return sum(w[k] * comp[k] for k in comp) + self._offset(X)


class QuantileOn:
    """P90 상한용: 직전가동일 잔차 타깃 LightGBM 분위수 회귀."""

    def __init__(self, alpha=0.9):
        self.alpha = alpha

    def fit(self, X, y, hour=None, copy=None):
        self.col = "op_power_lag" if y.name == "power" else "op_peak_lag"
        self.fill = float(y.median())
        c = np.zeros(len(X), bool) if copy is None else np.asarray(copy)
        p = {**FINAL["lgbm"], "objective": "quantile", "alpha": self.alpha, "random_state": SEED}
        self.m = lgb.LGBMRegressor(**p).fit(X, y.to_numpy() - X[self.col].fillna(self.fill).to_numpy(),
                                            sample_weight=np.where(c, FINAL["lgbm_copy_weight"], 1.0))
        return self

    def predict(self, X, hour=None):
        return self.m.predict(X) + X[self.col].fillna(self.fill).to_numpy()


def final_model():
    return RegimeModel(on_model=EnsembleOn())


def candidates():
    return {
        "naive_24h": lambda: Naive(24),
        "naive_168h": lambda: Naive(168),
        "ridge": RidgeModel,
        "lgbm": LGBM,
        "regime_lgbm": RegimeModel,
        "regime_ens": final_model,
    }


def quantile_upper(alpha=0.9):
    """피크 위험 상한(P90) — 가동일 불확실성 밴드."""
    return RegimeModel(on_model=QuantileOn(alpha), off_q=alpha)
