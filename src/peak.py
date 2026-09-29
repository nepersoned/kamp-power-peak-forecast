"""피크 위험(15분 최대수요 >= TAU) 확률 예측.

TAU는 학습 파라미터가 아니라 관리 목표수요(정책값). 기본 190 = 원본 기간 15분 수요 상위 5%.
분류기는 학습 구간으로 학습 -> 검증 구간에서 등위보정(isotonic) + 판정 임계값 결정 -> 테스트 1회 평가.
"""
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import average_precision_score, brier_score_loss, f1_score, fbeta_score, precision_score, recall_score

from .models import LGB_PARAMS, operating

TAU = 190


class PeakClassifier:
    """가동일 시간대만 학습(휴무일은 목표수요 초과가 구조적으로 불가능 -> 확률 0)."""

    def fit(self, X, y_event):
        on = operating(X)
        self.m = lgb.LGBMClassifier(**{**LGB_PARAMS, "n_estimators": 400}, is_unbalance=True).fit(X[on], y_event[on])
        return self

    def raw(self, X):
        p = np.zeros(len(X))
        on = operating(X)
        p[on] = self.m.predict_proba(X[on])[:, 1]
        return p

    def calibrate(self, X, y_event):
        self.iso = IsotonicRegression(out_of_bounds="clip").fit(self.raw(X), y_event)
        return self

    def predict_proba(self, X):
        return self.iso.predict(self.raw(X))


def pick_threshold(y, score, beta=2.0):
    """미탐지(FN) 비용이 오경보보다 크므로 F2 최대 임계값 선택(검증 구간에서만)."""
    grid = np.unique(np.quantile(score, np.linspace(0.5, 0.995, 200)))
    s = [fbeta_score(y, score >= t, beta=beta, zero_division=0) for t in grid]
    return float(grid[int(np.argmax(s))])


def event_metrics(y, score, thr, is_prob=False):
    """score: 확률(is_prob=True) 또는 예측 수요값(순위 점수). PR-AUC는 둘 다, Brier는 확률만."""
    y = np.asarray(y); score = np.asarray(score); yhat = score >= thr
    tp = int((yhat & (y == 1)).sum()); fp = int((yhat & (y == 0)).sum()); fn = int((~yhat & (y == 1)).sum())
    out = dict(threshold=round(float(thr), 3), events=int(y.sum()), TP=tp, FP=fp, FN=fn,
               precision=precision_score(y, yhat, zero_division=0), recall=recall_score(y, yhat, zero_division=0),
               F1=f1_score(y, yhat, zero_division=0), F2=fbeta_score(y, yhat, beta=2, zero_division=0),
               PR_AUC=average_precision_score(y, score))
    if is_prob:
        out["Brier"] = brier_score_loss(y, score)
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in out.items()}


def shift_plan(df, pred_peak, days, frac=0.3, k=2):
    """피크 저감 시나리오: 각 가동일에서 예측 수요 상위 k개 시간의 생산량 frac만큼을
    같은 날 가동 시간 중 예측 수요 하위 k개 시간으로 균등 이전 (일 생산 총량 보존)."""
    new = df["prod"].copy()
    pp = pd.Series(pred_peak, index=df.index)
    moves = []
    for d in days:
        sl = slice(d, d + pd.Timedelta(hours=23))
        prod_d, pk_d = new.loc[sl], pp.loc[sl]
        active = prod_d[prod_d > 0].index
        if len(active) < 2 * k:
            continue
        donors = pk_d.loc[active].nlargest(k).index
        receivers = pk_d.loc[active.difference(donors)].nsmallest(k).index
        amt = prod_d.loc[donors] * frac
        new.loc[donors] -= amt
        new.loc[receivers] += amt.sum() / k
        moves.append(dict(date=d.date(), donors=[t.hour for t in donors], receivers=[t.hour for t in receivers],
                          moved=float(amt.sum()), day_prod=float(prod_d.sum())))
    return new, pd.DataFrame(moves)


def simulate_shaving(df, X, build_fn, model, test_mask, frac=0.3, k=2):
    """학습된 피크 모델로 '생산계획 조정 후' 수요를 재예측(모델 기반 what-if).
    시차 피처(과거 실측 전력)는 그대로 두고 생산계획 파생 피처만 재계산."""
    idx = df.index[test_mask]
    base = pd.Series(model.predict(X.loc[idx], df.loc[idx, "hour"]), idx)
    on_days = pd.Index(sorted({t.normalize() for t in idx[operating(X.loc[idx])]}))
    new_prod, moves = shift_plan(df.loc[idx], base, on_days, frac, k)
    df2 = df.copy()
    df2.loc[idx, "prod"] = new_prod
    X2 = build_fn(df2)
    plan_cols = [c for c in X2.columns if c.startswith(("prod", "log_prod", "night_shift", "first_active", "last_active",
                                                         "before_start", "after_end"))]
    X_new = X.loc[idx].copy()
    X_new[plan_cols] = X2.loc[idx, plan_cols]
    after = pd.Series(model.predict(X_new, df.loc[idx, "hour"]), idx)
    return base, after, moves
