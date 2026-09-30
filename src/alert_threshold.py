"""비용 기준 래칫 위험 경보 임계값 (3장 미탐지·오경보를 돈으로).

경보 규칙: 래칫 초과 확률 p(과거 오차 시나리오 중 기본요금 시간 최대수요가 바닥을 넘는 비율) ≥ θ 이면
           보호 계획(시나리오 평균 래칫 최소화 확률 MILP), 아니면 전력량요금 최적 계획(MILP 평균 예측).
실제 래칫 갱신 사건은 7/19 하나뿐이라 θ를 실제 데이터로 고를 수 없다 →
  학습 구간 원본 가동일에 래칫 바닥을 무작위(실제 최대 + U(−20, +15)kW)로 둔 합성일에서 총비용 최소 θ 선택,
  그 θ를 실제 7월(검증)·테스트에 그대로 적용.
사건 = 원래 계획대로 갔을 때 실제 최대수요가 바닥을 넘는 날. TP/FP/FN/TN과 날짜별 비용을 기록.
"""
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from .dashboard import exceed_prob
from .decision_eval import true_cost
from .milp import solve_day
from .ratchet_rl import realized_peak
from .stochastic_milp import residual_paths, solve_stochastic

OUT = Path(__file__).resolve().parents[1] / "outputs" / "alert"
THETAS = (0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.01)


def day_record(d, coefs, pool):
    scen = d.fc_pk[None, :] + pool
    p = exceed_prob(d, d.prod, scen, coefs)
    event = realized_peak(d, d.prod, coefs, 24) > d.floor
    base = true_cost(d, d.act_pw, d.act_pk, d.prod, coefs)["total"]
    energy_plan = solve_day(d, None, coefs, base=(d.fc_pw, d.fc_pk))[0]
    prot_plan = solve_stochastic(d, coefs, scen, lam=0.0, time_limit_s=60)
    return dict(day=str(d.day.date()), floor=d.floor, p=p, event=bool(event),
                cost_energy_plan=true_cost(d, d.act_pw, d.act_pk, energy_plan, coefs)["total"] - base,
                cost_protect_plan=true_cost(d, d.act_pw, d.act_pk, prot_plan, coefs)["total"] - base)


def policy_table(rec):
    rows = []
    for th in THETAS:
        alert = rec["p"] >= th
        cost = np.where(alert, rec["cost_protect_plan"], rec["cost_energy_plan"])
        ev = rec["event"]
        rows.append(dict(theta=th, days=len(rec), events=int(ev.sum()), alerts=int(alert.sum()),
                         TP=int((alert & ev).sum()), FP=int((alert & ~ev).sum()), FN=int((~alert & ev).sum()),
                         saving_won_per_day=float(-cost.mean()),
                         fn_cost_won=float(np.where(~alert & ev, rec["cost_energy_plan"], 0).sum()),
                         fp_cost_won=float(np.where(alert & ~ev, rec["cost_protect_plan"] - rec["cost_energy_plan"], 0).sum())))
    return pd.DataFrame(rows)


def main(n_synth=300, seed=0):
    import warnings
    warnings.filterwarnings("ignore")
    from .ratchet_rl import build_days, realized_peak as rp
    OUT.mkdir(parents=True, exist_ok=True)
    coefs, train_days, val_days, test_days = build_days()
    rng = np.random.default_rng(seed)
    pool_tr = residual_paths(train_days)
    syn = []
    for k in range(n_synth):
        i = int(rng.integers(len(train_days)))
        d0 = train_days[i]
        d = replace(d0, floor=rp(d0, d0.prod, coefs, 24) + float(rng.uniform(-20, 15)))
        pool = np.delete(pool_tr, i, axis=0) if len(pool_tr) == len(train_days) else pool_tr   # 자기 오차 제외
        syn.append(day_record(d, coefs, pool))
    syn = pd.DataFrame(syn)
    syn.to_csv(OUT / "synthetic_days.csv", index=False, encoding="utf-8-sig")
    ts = policy_table(syn)
    ts.to_csv(OUT / "theta_synthetic.csv", index=False, encoding="utf-8-sig")
    best = float(ts.loc[ts["saving_won_per_day"].idxmax(), "theta"])
    print("[합성 래칫 위험일에서 θ 선택]\n", ts.round(1).to_string(index=False), f"\n선택 θ = {best}", flush=True)

    pools = {"valid": residual_paths(train_days), "test": residual_paths(train_days + val_days)}
    for period, days in (("valid", val_days), ("test", test_days)):
        rec = pd.DataFrame([day_record(d, coefs, pools[period]) for d in days])
        rec.to_csv(OUT / f"days_{period}.csv", index=False, encoding="utf-8-sig")
        t = policy_table(rec)
        t.to_csv(OUT / f"theta_{period}.csv", index=False, encoding="utf-8-sig")
        print(f"\n[{period}] 선택 θ={best} 적용\n", t[t["theta"] == best].round(1).to_string(index=False))
        print(t.round(1).to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
