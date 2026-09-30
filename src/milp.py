"""래칫 인지 MILP 부하이동 기준선 (predict-then-optimize).

결정변수 x[h,k]: 원래 h시 계획 물량 중 h+k시에 처리하는 양 (k = 0..MAX_DELAY) — 강화학습 환경과 같은 제약.
전력 반응은 선형 대리모형으로 근사하고, 하루 기준선에 고정한다:
    전력_t(새 계획) ≈ LightGBM_t(원래 계획) + S_t(새 계획) − S_t(원래 계획)
    S_t = Σ_h γ_h·[h=t]·on_t + a·on_{t−1} + b·on_{t+1} + c·start_t + β·p_t
목적: 전력량요금 Σ 단가_t·전력_t + 기본요금 × 12개월 × max(0, 최대수요 − 래칫 바닥)
풀이 결과 계획은 강화학습과 똑같은 시뮬레이터·잔차 경로로 평가한다.
"""
import numpy as np
import pandas as pd
from ortools.linear_solver import pywraplp
from sklearn.linear_model import Ridge

from . import tariff as T
from .rl_env import CAP_MULT, MAX_DELAY, n_starts


def surrogate_design(prod, hour, day):
    d = pd.DataFrame({"prod": np.asarray(prod, float), "hour": np.asarray(hour)}, index=day.index if hasattr(day, "index") else None)
    on = (d["prod"] > 0).astype(float)
    g = pd.Series(np.asarray(day)).to_numpy()
    s = pd.DataFrame({"on": on.to_numpy(), "hour": d["hour"].to_numpy(), "g": g, "prod": d["prod"].to_numpy()})
    s["on_prev"] = s.groupby("g")["on"].shift(1).fillna(0)
    s["on_next"] = s.groupby("g")["on"].shift(-1).fillna(0)
    s["start"] = ((s["on"] == 1) & (s["on_prev"] == 0)).astype(float)
    Z = pd.get_dummies(s["hour"], prefix="h").astype(float)          # 시간대 기저(대리모형 적합용)
    for h in range(24):
        Z[f"on_h{h}"] = s["on"] * (s["hour"] == h)
    Z[["on_prev", "on_next", "start", "prod"]] = s[["on_prev", "on_next", "start", "prod"]]
    return Z


def fit_surrogate(df, mask):
    """가동일 원본 시간으로 전력·15분최대 선형 대리모형 적합. 반환: {target: 계수 dict}."""
    sub = df[mask]
    Z = surrogate_design(sub["prod"].fillna(0), sub["hour"], sub.index.normalize())
    coefs = {}
    for tgt in ("power", "peak15"):
        m = Ridge(1.0).fit(Z, sub[tgt])
        c = pd.Series(m.coef_, Z.columns)
        coefs[tgt] = dict(gain=np.array([c[f"on_h{h}"] for h in range(24)]), prev=c["on_prev"], next=c["on_next"],
                          start=c["start"], beta=c["prod"])
    return coefs


def _S_numeric(p, cf):
    on = (np.asarray(p) > 0).astype(float)
    prev = np.r_[0, on[:-1]]; nxt = np.r_[on[1:], 0]
    start = on * (1 - prev)
    return cf["gain"] * on + cf["prev"] * prev + cf["next"] * nxt + cf["start"] * start + cf["beta"] * np.asarray(p)


def solve_day(dd, sim, coefs, max_delay=MAX_DELAY, cap_mult=CAP_MULT, months_ahead=12, time_limit_s=20, base=None,
              labor_won=0.0):
    """base=(시간별 전력, 15분최대) 예측을 직접 주면 그 예측기로 계획(결정 기반 평가용). 없으면 시뮬레이터 기대값."""
    plan = dd.prod
    cap = max(plan.max() * cap_mult, 1.0)
    base_pw, base_pk = base if base is not None else sim.expected(dd, plan)
    S0 = {t: _S_numeric(plan, coefs[t]) for t in ("power", "peak15")}
    solver = pywraplp.Solver.CreateSolver("SCIP") or pywraplp.Solver.CreateSolver("CBC")
    H = range(24)
    x = {(h, k): solver.NumVar(0, float(plan[h]), f"x{h}_{k}") for h in H for k in range(max_delay + 1) if plan[h] > 0 and h + k <= 23}
    for h in H:
        if plan[h] > 0:
            solver.Add(sum(x[h, k] for k in range(max_delay + 1) if (h, k) in x) == float(plan[h]))
    p = [sum(x[h, t - h] for h in range(max(0, t - max_delay), t + 1) if (h, t - h) in x) for t in H]
    on = [solver.BoolVar(f"on{t}") for t in H]
    start = [solver.NumVar(0, 1, f"st{t}") for t in H]
    minp = 1.0
    for t in H:
        solver.Add(p[t] <= cap * on[t])
        solver.Add(p[t] >= minp * on[t])
        prev = on[t - 1] if t > 0 else 0
        solver.Add(start[t] >= on[t] - prev)          # start = on[t] AND NOT on[t-1] 을 선형으로 정확히 표현
        solver.Add(start[t] <= on[t])
        solver.Add(start[t] <= 1 - prev)
    solver.Add(sum(start) <= n_starts(plan))          # 가동 시작 횟수는 원래 계획 이하(펄스 운전 금지)

    def S(tgt, t):
        cf = coefs[tgt]
        prev = on[t - 1] if t > 0 else 0
        nxt = on[t + 1] if t < 23 else 0
        return cf["gain"][t] * on[t] + cf["prev"] * prev + cf["next"] * nxt + cf["start"] * start[t] + cf["beta"] * p[t]
    power = [base_pw[t] - S0["power"][t] + S("power", t) for t in H]
    peak = [base_pk[t] - S0["peak15"][t] + S("peak15", t) for t in H]
    D = solver.NumVar(0, solver.infinity(), "D")
    for t in H:
        if dd.demand_band[t] > T.OFF:
            solver.Add(D >= peak[t])
    r = solver.NumVar(0, solver.infinity(), "ratchet_excess")
    solver.Add(r >= D - dd.floor)
    labor = sum(float(dd.labor[t] - 1.0) * labor_won * p[t] for t in H)     # 1.5배 시간대 할증분
    solver.Minimize(sum(float(dd.rate[t]) * power[t] for t in H) + T.BASE_RATE["II"] * months_ahead * r + labor)
    solver.SetTimeLimit(int(time_limit_s * 1000))
    status = solver.Solve()
    if status not in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        return plan.copy(), "fail"
    sol = np.array([sum(x[h, t - h].solution_value() for h in range(max(0, t - max_delay), t + 1) if (h, t - h) in x)
                    for t in H])
    sol[sol < 1e-3] = 0.0            # 풀이기 찌꺼기(1e-9 수준)가 '가동'으로 세어지지 않게
    return sol, "optimal" if status == pywraplp.Solver.OPTIMAL else "feasible"


def evaluate_milp(sim, days, coefs, seeds=(0, 1, 2), labor_won=0.0):
    rows = []
    plans = {}
    for i, dd in enumerate(days):
        plans[i] = solve_day(dd, sim, coefs)
    for s in seeds:
        for i, dd in enumerate(days):
            noise = sim.noise(np.random.default_rng(1000 * s + i))
            newp, status = plans[i]
            base = sim.cost(dd, dd.prod, noise, labor_won)
            new = sim.cost(dd, newp, noise, labor_won)
            rows.append(dict(seed=s, day=str(dd.day.date()), status=status,
                             reward_kwon=(base["total"] - new["total"]) / 1000,
                             moved=float(np.abs(newp - dd.prod).sum() / 2), day_prod=float(dd.prod.sum()),
                             base_total=base["total"], new_total=new["total"], base_energy=base["energy"],
                             new_energy=new["energy"], base_demand=base["demand"], new_demand=new["demand"],
                             base_ratchet=base["ratchet"], new_ratchet=new["ratchet"], new_labor=new["labor"],
                             extra_starts=new["extra_starts"]))
    return pd.DataFrame(rows), plans
