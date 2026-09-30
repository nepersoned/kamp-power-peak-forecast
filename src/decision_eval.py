"""결정 기반 평가: 예측기를 MAE가 아니라 "그 예측으로 짠 계획이 실제로 얼마나 아꼈나"로 비교.

절차 (테스트 가동일마다)
  1. 예측기 F의 하루 전 예측(시간별 전력·15분최대)을 기준선으로 래칫 인지 MILP 계획을 푼다.
  2. 실제 효과는 관측값에 고정한 대리모형 변화분으로 계산한다(특정 예측기 편향 없음):
        전력_t(새 계획) = 실측_t + S_t(새 계획) − S_t(원래 계획)
  3. 절감 = 실제 비용(원래 계획) − 실제 비용(새 계획).
     후회(regret) = 실측 기준선으로 푼 계획(완전 예측 상한)의 절감 − F의 절감.
외부 예측기(TabPFN 등)는 outputs/forecast_dist_{name}.csv (timestamp, target, q50 ...)를 두면 자동으로 포함한다.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import tariff as T
from .milp import _S_numeric, solve_day
from .rl_env import START_PENALTY_WON, n_starts

OUT = Path(__file__).resolve().parents[1] / "outputs"


def true_cost(dd, actual_pw, actual_pk, plan, coefs, months_ahead=12, labor_won=0.0):
    dpw = _S_numeric(plan, coefs["power"]) - _S_numeric(dd.prod, coefs["power"])
    dpk = _S_numeric(plan, coefs["peak15"]) - _S_numeric(dd.prod, coefs["peak15"])
    pw = np.clip(actual_pw + dpw, 0, None)
    pk = np.clip(actual_pk + dpk, 0, None)
    energy = float((pw * dd.rate).sum())
    demand = float(np.max(np.where(dd.demand_band > T.OFF, pk, 0.0)))
    ratchet = T.BASE_RATE["II"] * max(0.0, demand - dd.floor) * months_ahead
    switch = START_PENALTY_WON * max(0, n_starts(plan) - n_starts(dd.prod))
    labor = float((np.asarray(plan) * (dd.labor - 1.0)).sum() * labor_won)
    return dict(total=energy + ratchet + switch + labor, energy=energy, ratchet=ratchet, demand=demand, labor=labor)


def load_external(index):
    """outputs/forecast_dist_*.csv → {이름: (power Series, peak15 Series)} (q50을 점예측으로)."""
    out = {}
    for f in sorted(OUT.glob("forecast_dist_*.csv")):
        d = pd.read_csv(f, parse_dates=["timestamp"])
        name = f.stem.replace("forecast_dist_", "")
        piv = d.pivot_table(index="timestamp", columns="target", values="q50")
        if {"power", "peak15"} <= set(piv.columns):
            out[name] = (piv["power"].reindex(index), piv["peak15"].reindex(index))
    return out


def labor_tradeoff(forecast, df, days, sim, coefs, labor_grid=(0, 0.5, 1, 2, 4, 8)):
    """인건비 할증 단가(원/생산단위, 1.5배 시간대의 추가 0.5배분 기준)별 최적 계획의 절충.
    생산 1단위당 인건비 단가는 데이터에 없어 시나리오로 둔다."""
    rows = []
    fp, fk = forecast
    for w in labor_grid:
        for dd in days:
            act_pw = df.loc[dd.idx, "power"].to_numpy(float)
            act_pk = df.loc[dd.idx, "peak15"].to_numpy(float)
            base = true_cost(dd, act_pw, act_pk, dd.prod, coefs, labor_won=w)
            plan, _ = solve_day(dd, sim, coefs, base=(fp.reindex(dd.idx).to_numpy(float), fk.reindex(dd.idx).to_numpy(float)),
                                labor_won=w)
            c = true_cost(dd, act_pw, act_pk, plan, coefs, labor_won=w)
            night = (dd.labor > 1.0)
            rows.append(dict(labor_won=w, day=str(dd.day.date()), energy_saving_won=base["energy"] - c["energy"],
                             labor_extra_won=c["labor"] - base["labor"], net_saving_won=base["total"] - c["total"],
                             night_prod_change=float(plan[night].sum() - dd.prod[night].sum()),
                             moved_share=float(np.abs(plan - dd.prod).sum() / 2 / max(dd.prod.sum(), 1))))
    r = pd.DataFrame(rows)
    return r.groupby("labor_won")[["energy_saving_won", "labor_extra_won", "net_saving_won", "night_prod_change",
                                   "moved_share"]].mean().round(2)


def evaluate(forecasts, df, days, sim, coefs):
    """forecasts: {이름: (power Series, peak15 Series)} — 테스트 시간 인덱스."""
    rows = []
    for dd in days:
        act_pw = df.loc[dd.idx, "power"].to_numpy(float)
        act_pk = df.loc[dd.idx, "peak15"].to_numpy(float)
        base_cost = true_cost(dd, act_pw, act_pk, dd.prod, coefs)
        oracle_plan, _ = solve_day(dd, sim, coefs, base=(act_pw, act_pk))
        oracle_saving = base_cost["total"] - true_cost(dd, act_pw, act_pk, oracle_plan, coefs)["total"]
        for name, (fp, fk) in forecasts.items():
            bp, bk = fp.reindex(dd.idx).to_numpy(float), fk.reindex(dd.idx).to_numpy(float)
            if np.isnan(bp).any() or np.isnan(bk).any():
                continue
            plan, status = solve_day(dd, sim, coefs, base=(bp, bk))
            c = true_cost(dd, act_pw, act_pk, plan, coefs)
            rows.append(dict(day=str(dd.day.date()), forecaster=name, status=status,
                             mae_power=float(np.abs(bp - act_pw).mean()), mae_peak=float(np.abs(bk - act_pk).mean()),
                             saving_won=base_cost["total"] - c["total"],
                             energy_saving_won=base_cost["energy"] - c["energy"],
                             ratchet_saving_won=base_cost["ratchet"] - c["ratchet"],
                             oracle_saving_won=oracle_saving, regret_won=oracle_saving - (base_cost["total"] - c["total"]),
                             moved_share=float(np.abs(plan - dd.prod).sum() / 2 / max(dd.prod.sum(), 1))))
    return pd.DataFrame(rows)


def summarize(res, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for name, g in res.groupby("forecaster"):
        s = g["saving_won"].to_numpy()
        boot = [rng.choice(s, len(s)).mean() for _ in range(n_boot)]
        rows.append(dict(forecaster=name, days=len(g), mae_power=g["mae_power"].mean(), mae_peak=g["mae_peak"].mean(),
                         saving_won_per_day=s.mean(), saving_ci95=f"[{np.percentile(boot, 2.5):,.0f}, {np.percentile(boot, 97.5):,.0f}]",
                         energy_saving_won=g["energy_saving_won"].mean(), ratchet_saving_won=g["ratchet_saving_won"].mean(),
                         regret_won_per_day=g["regret_won"].mean(), oracle_saving_won=g["oracle_saving_won"].mean(),
                         moved_share=g["moved_share"].mean()))
    t = pd.DataFrame(rows).sort_values("saving_won_per_day", ascending=False)
    t["rank_by_mae"] = t["mae_peak"].rank().astype(int)
    t["rank_by_saving"] = t["saving_won_per_day"].rank(ascending=False).astype(int)
    return t.round(2)


def main(period="test"):
    """period='test': TRAIN+VALID 학습 → 테스트(8/16~9/14, 래칫 바닥 222에 묶인 기간)
       period='valid': TRAIN 학습 → 검증(7/16~8/15, 7/19 이전은 래칫 바닥 202)"""
    import warnings
    warnings.filterwarnings("ignore")
    import run_all
    from .models import candidates
    from .run_rl import TEST0, VAL0, DayData, build_all, day_floor
    from .data import load
    from .features import build

    df = load()
    X = build(df)
    run_all.COPY = df["is_copy"].to_numpy()
    tr, val, te = run_all.masks(df, X)
    fit_mask, eval_mask = (tr | val, te) if period == "test" else (tr, val)
    idx = df.index[eval_mask]
    forecasts = {}
    for name, mk in candidates().items():
        if name in ("naive_24h", "ridge"):
            continue
        pw, _ = run_all.fit_predict(mk, X, df["power"], df["hour"], fit_mask, eval_mask)
        pk, _ = run_all.fit_predict(mk, X, df["peak15"], df["hour"], fit_mask, eval_mask)
        forecasts[name] = (pd.Series(pw, idx), pd.Series(pk, idx))
    # 위험 인지 변형: 에너지는 평균 예측, 최대수요 제약은 분위수 상한(P75/P90) 예측
    from .models import quantile_upper
    for qa in (0.75, 0.9):
        up, _ = run_all.fit_predict(lambda: quantile_upper(qa), X, df["peak15"], df["hour"], fit_mask, eval_mask)
        forecasts[f"regime_ens+peakP{int(qa * 100)}"] = (forecasts["regime_ens"][0], pd.Series(up, idx))
    forecasts.update(load_external(idx))
    sim, _, test_days, coefs = build_all()
    if period == "test":
        days = test_days
    else:
        q = df[T.QUARTERS]
        prod_day = df["prod"].fillna(0).groupby(df.index.normalize()).sum()
        days = [DayData(df, X, d, day_floor(q, d)) for d in pd.date_range(VAL0, TEST0, freq="D", inclusive="left")
                if prod_day.get(d, 0) > 0 and eval_mask[df.index.get_indexer(pd.date_range(d, periods=24, freq="h"))].all()]
    res = evaluate(forecasts, df, days, sim, coefs)
    (OUT / "decision").mkdir(exist_ok=True)
    res.to_csv(OUT / "decision" / f"decision_eval_days_{period}.csv", index=False, encoding="utf-8-sig")
    tab = summarize(res)
    tab.to_csv(OUT / "decision" / f"decision_eval_{period}.csv", index=False, encoding="utf-8-sig")
    lt = labor_tradeoff(forecasts["regime_ens"], df, days, sim, coefs)
    lt.to_csv(OUT / "decision" / f"labor_tradeoff_{period}.csv", encoding="utf-8-sig")
    print(lt.to_string())
    print(f"[{period}] days {len(days)}, floors {sorted({round(d.floor) for d in days})}")
    print(tab.to_string(index=False))


if __name__ == "__main__":
    import sys
    main(sys.argv[1] if len(sys.argv) > 1 else "test")
