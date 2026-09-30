"""조건별 예측 오차와 모델 간 차이의 신뢰구간 (2·3장).

- 조건: 운전 레짐, 운전 상태 전환(휴무 후 재가동일·휴무 전날), 요금 시간대(경·중간·최대부하),
        피크 시간대(08~11, 13~16), 생산량 구간, 월요일
- 모델 간 차이: 날짜 단위 짝지은 부트스트랩으로 MAE 차이의 95% 신뢰구간
출력: outputs/conditions/
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import tariff as T

OUT = Path(__file__).resolve().parents[1] / "outputs" / "conditions"
MODELS = ["naive_168h", "lgbm", "regime_lgbm", "regime_ens"]


def condition_frame(df, X, mask):
    idx = df.index[mask]
    day = idx.normalize()
    on = X.loc[idx, "plan_on_day"].fillna(1).to_numpy() == 1
    prod_day = df["prod"].fillna(0).groupby(df.index.normalize()).sum()
    op = prod_day > 0
    prev_op = op.shift(1, fill_value=True)
    prev2_op = op.shift(2, fill_value=True)
    next_op = op.shift(-1, fill_value=True)
    restart = op & (~prev_op) & (~prev2_op)            # 이틀 이상 쉰 뒤 첫 가동일(주말 이후 월요일 포함)
    long_restart = op & pd.Series([(~op.shift(k, fill_value=True)).loc[d] if d in op.index else False
                                   for k in [3] for d in op.index], index=op.index) & (~prev_op)
    bands = T.hourly_bands(idx)["energy_band"].map({0: "경부하", 1: "중간부하", 2: "최대부하"}).to_numpy()
    h = df.loc[idx, "hour"].to_numpy()
    f = pd.DataFrame(index=idx)
    f["regime"] = np.where(on, "가동일", "휴무일")
    f["transition"] = np.select([day.map(long_restart).to_numpy(), day.map(restart).to_numpy(),
                                 (day.map(op) & ~day.map(next_op)).to_numpy()],
                                ["3일+ 휴무 후 재가동", "휴무 후 재가동", "휴무 전날"], "평상")
    f["tariff_band"] = bands
    f["peak_window"] = np.where(((h >= 8) & (h <= 11)) | ((h >= 13) & (h <= 16)), "주간 피크대", "기타")
    f["prod_bin"] = pd.cut(df.loc[idx, "prod"].fillna(0), [-1, 0, 500, 1500, 1e9],
                           labels=["0", "1-500", "501-1500", ">1500"]).astype(str).to_numpy()
    f["monday"] = np.where(df.loc[idx, "dow"].to_numpy() == 0, "월요일", "기타")
    return f


def by_condition(f, y, preds):
    rows = []
    for cond in ["regime", "transition", "tariff_band", "peak_window", "prod_bin", "monday"]:
        for level, g in f.groupby(cond):
            r = dict(condition=cond, level=level, hours=len(g))
            for m, p in preds.items():
                r[m] = float(np.abs(y.loc[g.index] - p.loc[g.index]).mean())
            rows.append(r)
    return pd.DataFrame(rows)


def paired_bootstrap(y, preds, a, b, n=5000, seed=0):
    """MAE(a) − MAE(b) 의 날짜 단위 부트스트랩. 음수면 a가 더 좋음."""
    e = pd.DataFrame({"a": (y - preds[a]).abs(), "b": (y - preds[b]).abs()})
    daily = e.groupby(e.index.normalize()).mean()
    d = (daily["a"] - daily["b"]).to_numpy()
    rng = np.random.default_rng(seed)
    boot = np.array([rng.choice(d, len(d)).mean() for _ in range(n)])
    return dict(a=a, b=b, days=len(d), mae_diff=float(d.mean()), ci_low=float(np.percentile(boot, 2.5)),
                ci_high=float(np.percentile(boot, 97.5)), p_a_better=float((boot < 0).mean()))


def main():
    import warnings
    warnings.filterwarnings("ignore")
    import run_all
    from .data import load
    from .features import build
    from .models import candidates

    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    X = build(df)
    run_all.COPY = df["is_copy"].to_numpy()
    tr, val, te = run_all.masks(df, X)
    all_cond, all_boot = [], []
    for period, fit_mask, ev_mask in [("valid", tr, val), ("test", tr | val, te)]:
        f = condition_frame(df, X, ev_mask)
        for tgt in ("power", "peak15"):
            y = df.loc[ev_mask, tgt]
            preds = {}
            for m in MODELS:
                p, _ = run_all.fit_predict(candidates()[m], X, df[tgt], df["hour"], fit_mask, ev_mask)
                preds[m] = pd.Series(p, y.index)
            c = by_condition(f, y, preds).assign(period=period, target=tgt)
            all_cond.append(c)
            for a, b in [("regime_ens", "lgbm"), ("regime_lgbm", "lgbm"), ("regime_ens", "naive_168h"),
                         ("lgbm", "naive_168h"), ("regime_ens", "regime_lgbm")]:
                all_boot.append(dict(period=period, target=tgt, **paired_bootstrap(y, preds, a, b)))
    cond = pd.concat(all_cond)
    boot = pd.DataFrame(all_boot)
    cond.round(2).to_csv(OUT / "error_by_condition.csv", index=False, encoding="utf-8-sig")
    boot.round(3).to_csv(OUT / "paired_bootstrap.csv", index=False, encoding="utf-8-sig")
    pd.set_option("display.width", 220)
    print(cond[cond["target"] == "power"].round(1).to_string(index=False))
    print(boot.round(2).to_string(index=False))


if __name__ == "__main__":
    main()
