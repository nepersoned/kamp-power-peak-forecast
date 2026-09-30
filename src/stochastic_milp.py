"""시나리오 기반 확률 MILP (CVaR): 예측 하나 대신 과거 예측 오차 경로 K개로 최대수요 시나리오를 만들고,
계획 하나로 모든 시나리오의 래칫 초과를 함께 관리한다.

목적 = 전력량요금(평균 예측) + 기본요금 × 12개월 × [(1−λ)·평균_k(초과_k) + λ·CVaR_α(초과)]
CVaR 선형화: η + 1/((1−α)K) Σ_k u_k,  u_k ≥ 초과_k − η,  u_k ≥ 0
오차 경로는 평가 구간보다 앞선 날에서만 뽑는다(7월 평가: 학습 구간 교차적합 오차 / 테스트: 학습+검증 오차).
평가는 결정 기반 평가와 같은 실측 고정 비용.
"""
from pathlib import Path

import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp

from . import tariff as T
from .decision_eval import true_cost
from .milp import _S_numeric, solve_day
from .rl_env import CAP_MULT, MAX_DELAY, n_starts

OUT = Path(__file__).resolve().parents[1] / "outputs" / "stochastic"


def solve_stochastic(d, coefs, peak_scen, alpha=0.9, lam=0.5, months_ahead=12, time_limit_s=30):
    plan = d.prod
    cap = max(plan.max() * CAP_MULT, 1.0)
    K = len(peak_scen)
    S0p = _S_numeric(plan, coefs["power"])
    S0k = _S_numeric(plan, coefs["peak15"])
    solver = pywraplp.Solver.CreateSolver("SCIP")
    H = range(24)
    x = {(h, k): solver.NumVar(0, float(plan[h]), f"x{h}_{k}") for h in H for k in range(MAX_DELAY + 1)
         if plan[h] > 0 and h + k <= 23}
    for h in H:
        if plan[h] > 0:
            solver.Add(sum(x[h, k] for k in range(MAX_DELAY + 1) if (h, k) in x) == float(plan[h]))
    p = [sum(x[h, t - h] for h in range(max(0, t - MAX_DELAY), t + 1) if (h, t - h) in x) for t in H]
    on = [solver.BoolVar(f"on{t}") for t in H]
    start = [solver.NumVar(0, 1, f"st{t}") for t in H]
    for t in H:
        prev = on[t - 1] if t > 0 else 0
        solver.Add(p[t] <= cap * on[t]); solver.Add(p[t] >= 1.0 * on[t])
        solver.Add(start[t] >= on[t] - prev); solver.Add(start[t] <= on[t]); solver.Add(start[t] <= 1 - prev)
    solver.Add(sum(start) <= n_starts(plan))

    def S(cf, t):
        prev = on[t - 1] if t > 0 else 0
        nxt = on[t + 1] if t < 23 else 0
        return cf["gain"][t] * on[t] + cf["prev"] * prev + cf["next"] * nxt + cf["start"] * start[t] + cf["beta"] * p[t]

    dS_pk = [S(coefs["peak15"], t) - S0k[t] for t in H]
    energy = sum(float(d.rate[t]) * (d.fc_pw[t] - S0p[t] + S(coefs["power"], t)) for t in H)
    ex = []
    for k in range(K):
        Dk = solver.NumVar(0, solver.infinity(), f"D{k}")
        for t in H:
            if d.demand_band[t] > T.OFF:
                solver.Add(Dk >= float(peak_scen[k][t]) + dS_pk[t])
        rk = solver.NumVar(0, solver.infinity(), f"r{k}")
        solver.Add(rk >= Dk - d.floor)
        ex.append(rk)
    eta = solver.NumVar(0, solver.infinity(), "eta")
    u = [solver.NumVar(0, solver.infinity(), f"u{k}") for k in range(K)]
    for k in range(K):
        solver.Add(u[k] >= ex[k] - eta)
    cvar = eta + sum(u) / ((1 - alpha) * K)
    risk = (1 - lam) * sum(ex) / K + lam * cvar
    solver.Minimize(energy + T.BASE_RATE["II"] * months_ahead * risk)
    solver.SetTimeLimit(int(time_limit_s * 1000))
    st = solver.Solve()
    if st not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return plan.copy()
    sol = np.array([sum(x[h, t - h].solution_value() for h in range(max(0, t - MAX_DELAY), t + 1) if (h, t - h) in x)
                    for t in H])
    sol[sol < 1e-3] = 0.0
    return sol


def residual_paths(days):
    return np.array([d.act_pk - d.fc_pk for d in days if np.isfinite(d.fc_pk).all()])


def evaluate_period(days, coefs, resid, K=30, seed=0, settings=((0.9, 0.0), (0.9, 0.5), (0.9, 1.0))):
    rng = np.random.default_rng(seed)
    rows = []
    for d in days:
        base = true_cost(d, d.act_pw, d.act_pk, d.prod, coefs)
        pick = resid[rng.choice(len(resid), size=min(K, len(resid)), replace=len(resid) < K)]
        scen = d.fc_pk[None, :] + pick
        plans = {"MILP 평균 예측": solve_day(d, None, coefs, base=(d.fc_pw, d.fc_pk))[0],
                 "MILP P75 상한": solve_day(d, None, coefs, base=(d.fc_pw, d.fc_pk75))[0]}
        for a, lam in settings:
            plans[f"확률 MILP (λ={lam:g}, CVaR{int(a * 100)})"] = solve_stochastic(d, coefs, scen, alpha=a, lam=lam)
        for name, plan in plans.items():
            c = true_cost(d, d.act_pw, d.act_pk, plan, coefs)
            rows.append(dict(day=str(d.day.date()), policy=name, saving_won=base["total"] - c["total"],
                             energy_saving_won=base["energy"] - c["energy"], ratchet_saving_won=base["ratchet"] - c["ratchet"],
                             moved_share=float(np.abs(plan - d.prod).sum() / 2 / max(d.prod.sum(), 1))))
    return pd.DataFrame(rows)


def main():
    import warnings
    warnings.filterwarnings("ignore")
    from .ratchet_rl import build_days, summarize
    OUT.mkdir(parents=True, exist_ok=True)
    coefs, train_days, val_days, test_days = build_days()
    for period, days, pool in [("valid", val_days, residual_paths(train_days)),
                               ("test", test_days, residual_paths(train_days + val_days))]:
        res = evaluate_period(days, coefs, pool)
        res.to_csv(OUT / f"days_{period}.csv", index=False, encoding="utf-8-sig")
        tab = summarize(res)
        tab.to_csv(OUT / f"compare_{period}.csv", index=False, encoding="utf-8-sig")
        print(f"\n[{period}] residual paths {len(pool)}\n{tab.to_string(index=False)}", flush=True)
        if period == "valid":
            print(res[res["day"] == "2021-07-19"][["policy", "saving_won", "ratchet_saving_won"]].to_string(index=False))


if __name__ == "__main__":
    main()
