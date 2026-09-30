"""메타휴리스틱(유전 알고리즘 GA, 타부 탐색)으로 하루 생산 이동 계획 최적화.

결정 변수: 시각별 행동 a_h ∈ {0, 1, 2} (신규 물량 0/50/100% 가동) — 강화학습과 같은 제약
  (최대 2시간 지연, 시간당 상한, 가동 중 끊김 금지 행동 차단, 23시 잔량 처리).
비교 두 가지:
  A. 선형 대리모형 세계(실측 고정, 결정 기반 평가와 동일): MILP가 정확한 최적 → GA·타부가 얼마나 근접하나
  B. LightGBM 시뮬레이터 세계(비선형 블랙박스): MILP는 선형 근사로 풀 수밖에 없지만 GA·타부는 직접 최적화
평가: A는 실측 고정 비용, B는 시뮬레이터 기대 비용과 검증 잔차 경로 3개(공통 난수).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .decision_eval import true_cost
from .milp import solve_day
from .rl_env import CAP_MULT, FRACTIONS, MAX_DELAY, schedule_step

OUT = Path(__file__).resolve().parents[1] / "outputs" / "metaheuristics"


def rollout(prod, acts):
    """행동 벡터 → 실행 계획 (행동 차단 포함)."""
    cap = max(prod.max() * CAP_MULT, 1.0)
    done, queue = np.zeros(24), []
    for h in range(24):
        a = int(acts[h])
        if a == 0 and h > 0 and done[h - 1] > 0:
            q = queue + ([(h, float(prod[h]))] if prod[h] > 0 else [])
            forced = sum(x for t, x in q if h - t >= MAX_DELAY)
            free = sum(x for t, x in q if h - t < MAX_DELAY)
            if forced <= 0 and free > 0:
                a = 1
        done[h], queue = schedule_step(h, prod[h], queue, a, cap, MAX_DELAY)
    return done


def ga(cost_fn, prod, pop=40, gens=40, pmut=0.08, elite=4, seed=0):
    rng = np.random.default_rng(seed)
    active = prod > 0
    P = np.full((pop, 24), 2)
    P[1:] = np.where(active & (rng.random((pop - 1, 24)) < 0.3), rng.integers(0, 3, (pop - 1, 24)), 2)
    cache = {}

    def f(a):
        k = a.tobytes()
        if k not in cache:
            cache[k] = cost_fn(rollout(prod, a))
        return cache[k]

    fit = np.array([f(a) for a in P])
    for _ in range(gens):
        order = np.argsort(fit)
        new = [P[i].copy() for i in order[:elite]]
        while len(new) < pop:
            i, j = rng.integers(pop, size=2), rng.integers(pop, size=2)
            p1 = P[i[np.argmin(fit[i])]]; p2 = P[j[np.argmin(fit[j])]]
            child = np.where(rng.random(24) < 0.5, p1, p2)
            m = (rng.random(24) < pmut) & active
            child[m] = rng.integers(0, 3, m.sum())
            new.append(child)
        P = np.array(new)
        fit = np.array([f(a) for a in P])
    best = P[np.argmin(fit)]
    return rollout(prod, best), float(fit.min()), len(cache)


def tabu(cost_fn, prod, iters=60, tenure=7, seed=0):
    rng = np.random.default_rng(seed)
    active = np.flatnonzero(prod > 0)
    cur = np.full(24, 2)
    cur_c = cost_fn(rollout(prod, cur))
    best, best_c = cur.copy(), cur_c
    tabu_until = {}
    evals = 1
    for it in range(iters):
        cands = []
        for h in active:
            for a in (0, 1, 2):
                if a == cur[h]:
                    continue
                nb = cur.copy(); nb[h] = a
                c = cost_fn(rollout(prod, nb)); evals += 1
                is_tabu = tabu_until.get((h, a), -1) > it
                if not is_tabu or c < best_c:                 # 열망 기준
                    cands.append((c, h, a, nb))
        if not cands:
            break
        c, h, a, nb = min(cands, key=lambda z: z[0])
        tabu_until[(h, cur[h])] = it + tenure                  # 되돌리는 이동 금지
        cur, cur_c = nb, c
        if c < best_c:
            best, best_c = nb.copy(), c
    return rollout(prod, best), best_c, evals


def world_A(days, coefs):
    """선형 대리모형 세계: 실측 고정 비용을 목적함수로(완전 정보) — MILP와 최적성 비교."""
    rows = []
    for d in days:
        base = true_cost(d, d.act_pw, d.act_pk, d.prod, coefs)["total"]
        cf = lambda plan: true_cost(d, d.act_pw, d.act_pk, plan, coefs)["total"]
        plans = {"MILP(정확 최적)": solve_day(d, None, coefs, base=(d.act_pw, d.act_pk))[0]}
        plans["GA"], _, _ = ga(cf, d.prod)
        plans["타부 탐색"], _, _ = tabu(cf, d.prod)
        for n, p in plans.items():
            rows.append(dict(day=str(d.day.date()), policy=n, saving_won=base - cf(p)))
    return pd.DataFrame(rows)


def world_B(sim, days_env, coefs, seeds=(0, 1, 2)):
    """LightGBM 시뮬레이터 세계: 기대 비용을 목적함수로 GA·타부, MILP는 선형 근사로 계획."""
    rows = []
    for i, dd in enumerate(days_env):
        cf = lambda plan: sim.cost(dd, plan, None)["total"]
        plans = {"MILP(선형 근사)": solve_day(dd, sim, coefs)[0]}
        plans["GA(블랙박스 직접)"], _, ne_ga = ga(cf, dd.prod)
        plans["타부(블랙박스 직접)"], _, ne_tb = tabu(cf, dd.prod)
        for s in seeds:
            noise = sim.noise(np.random.default_rng(1000 * s + i))
            base = sim.cost(dd, dd.prod, noise)
            for n, p in plans.items():
                c = sim.cost(dd, p, noise)
                rows.append(dict(day=str(dd.day.date()), seed=s, policy=n, saving_won=base["total"] - c["total"],
                                 energy_saving_won=base["energy"] - c["energy"],
                                 expected_saving_won=cf(dd.prod) - cf(p),
                                 moved_share=float(np.abs(p - dd.prod).sum() / 2 / max(dd.prod.sum(), 1))))
    return pd.DataFrame(rows)


def main():
    import warnings
    warnings.filterwarnings("ignore")
    from .ratchet_rl import build_days
    from .run_rl import build_all
    OUT.mkdir(parents=True, exist_ok=True)
    coefs, _, val_days, test_days = build_days()
    a = pd.concat([world_A(val_days, coefs).assign(period="valid"), world_A(test_days, coefs).assign(period="test")])
    a.to_csv(OUT / "world_A_days.csv", index=False, encoding="utf-8-sig")
    ta = a.groupby(["period", "policy"], sort=False)["saving_won"].agg(["mean", "min", "size"]).round(0)
    print("[A] 선형 대리모형 세계(완전 정보) — MILP가 정확한 최적\n", ta.to_string(), flush=True)

    sim, _, test_env_days, coefs_b = build_all()
    b = world_B(sim, test_env_days, coefs_b)
    b.to_csv(OUT / "world_B_days.csv", index=False, encoding="utf-8-sig")
    tb = b.groupby("policy", sort=False)[["expected_saving_won", "saving_won", "energy_saving_won", "moved_share"]].mean().round(1)
    print("\n[B] LightGBM 시뮬레이터 세계(테스트 25일 × 잔차 경로 3)\n", tb.to_string(), flush=True)


# ---------- 연속 분할 유전자 GA (MILP와 같은 결정 공간) ----------
def shares_to_plan(prod, G):
    """G[h] = (그대로, +1시간, +2시간) 비율(합 1). 23시 이후로 넘어가는 몫은 가능한 마지막 시각으로."""
    p = np.zeros(24)
    for h in range(24):
        if prod[h] <= 0:
            continue
        for k in range(MAX_DELAY + 1):
            p[min(h + k, 23)] += prod[h] * G[h, k]
    return p


def ga_split(cost_fn, prod, pop=60, gens=80, sigma=0.15, elite=4, seed=0, init=None):
    """cost_fn(plan)을 최소화. 상한 초과는 벌점(초과량 × 1,000원)으로 처리."""
    rng = np.random.default_rng(seed)
    cap = max(prod.max() * CAP_MULT, 1.0)
    active = prod > 0

    def norm(G):
        G = np.clip(G, 0, None) + 1e-9
        return G / G.sum(axis=1, keepdims=True)

    def fitness(G):
        plan = shares_to_plan(prod, G)
        plan[plan < 1e-3] = 0.0
        return cost_fn(plan) + 1000.0 * np.clip(plan - cap, 0, None).sum(), plan

    base = np.zeros((24, 3)); base[:, 0] = 1.0
    P = [base.copy()] + ([init] if init is not None else [])
    while len(P) < pop:
        G = base + rng.exponential(0.3, (24, 3)) * (rng.random((24, 1)) < 0.4)
        P.append(norm(G))
    P = np.array(P)
    fit = np.array([fitness(G)[0] for G in P])
    for _ in range(gens):
        order = np.argsort(fit)
        new = [P[i] for i in order[:elite]]
        while len(new) < pop:
            i, j = rng.integers(pop, size=2), rng.integers(pop, size=2)
            a, b = P[i[np.argmin(fit[i])]], P[j[np.argmin(fit[j])]]
            mask = rng.random((24, 1)) < 0.5
            child = np.where(mask, a, b)
            m = (rng.random((24, 1)) < 0.2) & active[:, None]
            child = np.where(m, child + rng.normal(0, sigma, (24, 3)), child)
            new.append(norm(child))
        P = np.array(new)
        fit = np.array([fitness(G)[0] for G in P])
    best = P[np.argmin(fit)]
    return fitness(best)[1], float(fit.min())


def main_split():
    import warnings
    warnings.filterwarnings("ignore")
    from .ratchet_rl import build_days
    from .run_rl import build_all
    OUT.mkdir(parents=True, exist_ok=True)
    coefs, _, val_days, test_days = build_days()
    rows = []
    for period, days in [("valid", val_days), ("test", test_days)]:
        for d in days:
            base = true_cost(d, d.act_pw, d.act_pk, d.prod, coefs)["total"]
            cf = lambda plan: true_cost(d, d.act_pw, d.act_pk, plan, coefs)["total"]
            milp = solve_day(d, None, coefs, base=(d.act_pw, d.act_pk))[0]
            g, _ = ga_split(cf, d.prod)
            rows.append(dict(period=period, day=str(d.day.date()), milp=base - cf(milp), ga_split=base - cf(g)))
    a = pd.DataFrame(rows)
    a.to_csv(OUT / "world_A_split_days.csv", index=False, encoding="utf-8-sig")
    print("[A] 선형 대리모형 세계: MILP vs 연속 분할 GA\n", a.groupby("period")[["milp", "ga_split"]].mean().round(0).to_string(), flush=True)

    sim, _, env_days, coefs_b = build_all()
    rows = []
    for i, dd in enumerate(env_days):
        cf = lambda plan: sim.cost(dd, plan, None)["total"]
        milp = solve_day(dd, sim, coefs_b)[0]
        g, _ = ga_split(cf, dd.prod, pop=40, gens=50)
        for s in (0, 1, 2):
            noise = sim.noise(np.random.default_rng(1000 * s + i))
            b = sim.cost(dd, dd.prod, noise)
            for n, p in [("MILP(선형 근사)", milp), ("GA 연속 분할(블랙박스 직접)", g)]:
                c = sim.cost(dd, p, noise)
                rows.append(dict(day=str(dd.day.date()), seed=s, policy=n, expected_saving_won=cf(dd.prod) - cf(p),
                                 saving_won=b["total"] - c["total"], energy_saving_won=b["energy"] - c["energy"],
                                 moved_share=float(np.abs(p - dd.prod).sum() / 2 / max(dd.prod.sum(), 1))))
    b = pd.DataFrame(rows)
    b.to_csv(OUT / "world_B_split_days.csv", index=False, encoding="utf-8-sig")
    print("\n[B] LightGBM 세계: MILP vs 연속 분할 GA\n",
          b.groupby("policy")[["expected_saving_won", "saving_won", "energy_saving_won", "moved_share"]].mean().round(1).to_string())


if __name__ == "__main__":
    import sys
    main_split() if "--split" in sys.argv else main()
