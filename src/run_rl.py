"""피크·요금 제어: 기준 정책(무조치·규칙·완전정보 탐색)과 PPO 비교.

    python -m src.run_rl [--timesteps 60000] [--labor 0]

학습 에피소드: 테스트 이전 원본 가동일. 평가: 테스트 가동일(8/16~9/14) × 잔차 표본 3개.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from . import tariff as T
from .data import load
from .features import build
from .models import RegimeModel, operating
from .milp import evaluate_milp, fit_surrogate
from .rl_env import DayData, PeakControlEnv, Simulator, policy_noop, policy_peak_rule, run_policy

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parents[1] / "outputs" / "rl"
START, VAL0, TEST0, END = "2021-01-08", "2021-07-16", "2021-08-16", "2021-09-15"


def day_floor(q, day):
    """해당 일 이전까지의 래칫 바닥: 이전 래칫 대상 월 최대 + 같은 달 전날까지의 최대."""
    base = T.ratchet_floor(q, day)
    dp = T.demand_peak(q)
    same = dp[(dp.index >= day.to_period("M").to_timestamp()) & (dp.index < day)]
    return max(base, float(same.max()) if len(same) else 0.0)


def build_all():
    df = load()
    X = build(df)
    ok = (~df["outage"]).to_numpy() & X["power_lag168"].notna().to_numpy()
    t = df.index
    copy = df["is_copy"].to_numpy()
    tr = ok & (t >= START) & (t < VAL0)
    val = ok & (t >= VAL0) & (t < TEST0)
    trval = tr | val
    h = df["hour"]
    models, resid = {}, {}
    # 잔차 경로: 검증 구간 가동일(24시간 모두 유효한 날) 하루 단위
    on_val = val & operating(X)
    vdays = pd.Series(on_val, index=df.index).groupby(df.index.normalize()).sum()
    vdays = vdays[vdays == 24].index
    vmask = df.index.normalize().isin(vdays)
    for tgt in ("power", "peak15"):
        y = df[tgt]
        m_tr = RegimeModel().fit(X[tr], y[tr], h[tr], copy[tr])          # 잔차 추정용(학습 구간만)
        r = y[vmask].to_numpy() - m_tr.predict(X[vmask], h[vmask])
        resid[tgt] = r.reshape(-1, 24)
        models[tgt] = RegimeModel().fit(X[trval], y[trval], h[trval], copy[trval])  # 시뮬레이터
    sim = Simulator(models["power"], models["peak15"], resid["power"], resid["peak15"])

    q = df[T.QUARTERS]
    prod_day = df["prod"].fillna(0).groupby(df.index.normalize()).sum()
    daymask = df.groupby(df.index.normalize()).agg(copy=("is_copy", "first"), outage=("outage", "max"))

    def days(a, b):
        out = []
        for d in pd.date_range(a, b, freq="D", inclusive="left"):
            if d not in prod_day.index or prod_day[d] <= 0 or daymask.loc[d, "copy"] or daymask.loc[d, "outage"]:
                continue
            if X.loc[pd.date_range(d, periods=24, freq="h"), "power_lag168"].isna().any():
                continue
            out.append(DayData(df, X, d, day_floor(q, d)))
        return out

    sur_mask = (operating(X) & ~copy & (~df["outage"]).to_numpy() & (t >= START) & (t < TEST0))
    coefs = fit_surrogate(df, sur_mask)
    return sim, days(START, TEST0), days(TEST0, END), coefs


def summarize(res, name):
    g = res.groupby("seed")
    return dict(policy=name, days=int(res["day"].nunique()),
                saving_kwon_per_day=round(res["reward_kwon"].mean(), 2),
                saving_ci95=f"[{np.percentile(res['reward_kwon'], 2.5):.2f}, {np.percentile(res['reward_kwon'], 97.5):.2f}]",
                energy_saving_won=round((res["base_energy"] - res["new_energy"]).mean(), 0),
                ratchet_saving_won=round((res["base_ratchet"] - res["new_ratchet"]).mean(), 0),
                demand_change_kw=round((res["new_demand"] - res["base_demand"]).mean(), 2),
                labor_premium_won=round(res["new_labor"].mean(), 0),
                moved_share=round((res["moved"] / res["day_prod"]).mean(), 3),
                extra_starts=round(res["extra_starts"].mean(), 2),
                seed_std=round(g["reward_kwon"].mean().std(), 3))


def main(timesteps=60000, labor_won=0.0, load=False):
    OUT.mkdir(parents=True, exist_ok=True)
    sim, train_days, test_days, coefs = build_all()
    print(f"train days {len(train_days)}, test days {len(test_days)}", flush=True)
    env = PeakControlEnv(test_days, sim, labor_won=labor_won, seed=123)
    rows, per_day = [], []
    for name, pol, oracle in [("무조치", policy_noop, False), ("최대부하 절반 규칙", policy_peak_rule, False),
                              ("완전정보 탐색(상한)", None, True)]:
        r = run_policy(env, pol, len(test_days), oracle=oracle, labor_won=labor_won)
        rows.append(summarize(r, name)); per_day.append(r.assign(policy=name))
        print(rows[-1], flush=True)
    r, _ = evaluate_milp(sim, test_days, coefs, labor_won=labor_won)
    rows.append(summarize(r, "래칫 인지 MILP")); per_day.append(r.assign(policy="래칫 인지 MILP"))
    print(rows[-1], flush=True)

    if timesteps > 0 or load:
        from stable_baselines3 import PPO
        if load:
            model = PPO.load(OUT / f"ppo_labor{labor_won:g}")
        else:
            tenv = PeakControlEnv(train_days, sim, labor_won=labor_won, seed=7, shaped=True)
            model = PPO("MlpPolicy", tenv, n_steps=24 * 64, batch_size=256, gamma=1.0, learning_rate=3e-4,
                        ent_coef=0.01, seed=0, verbose=0)
            model.learn(total_timesteps=timesteps)
            model.save(OUT / f"ppo_labor{labor_won:g}")
        r = run_policy(env, lambda o, e: int(model.predict(o, deterministic=True)[0]), len(test_days))
        rows.append(summarize(r, "PPO")); per_day.append(r.assign(policy="PPO"))
        print(rows[-1], flush=True)

    tab = pd.DataFrame(rows)
    tab.to_csv(OUT / f"policy_compare_labor{labor_won:g}.csv", index=False, encoding="utf-8-sig")
    pd.concat(per_day).to_csv(OUT / f"policy_days_labor{labor_won:g}.csv", index=False, encoding="utf-8-sig")
    print(tab.to_string(index=False))
    return tab


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=60000)
    ap.add_argument("--labor", type=float, default=0.0)
    ap.add_argument("--load", action="store_true", help="저장된 PPO 모델로 평가만")
    a = ap.parse_args()
    main(a.timesteps, a.labor, a.load)
