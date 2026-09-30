"""운영 유연성의 가치: 생산 이동 허용 시간(최대 지연 1·2·3시간)별 절감 (4장).
실측 고정 비용(결정 기반 평가와 동일), 계획은 레짐 예측 기반 MILP와 완전 예측 MILP.
"""
from pathlib import Path

import numpy as np
import pandas as pd

from .decision_eval import true_cost
from .milp import solve_day

OUT = Path(__file__).resolve().parents[1] / "outputs" / "flexibility"


def main(delays=(1, 2, 3)):
    import warnings
    warnings.filterwarnings("ignore")
    from .ratchet_rl import build_days
    OUT.mkdir(parents=True, exist_ok=True)
    coefs, _, val_days, test_days = build_days()
    rows = []
    for period, days in (("valid", val_days), ("test", test_days)):
        for d in days:
            base = true_cost(d, d.act_pw, d.act_pk, d.prod, coefs)
            for k in delays:
                for name, b in (("예측 기반 MILP", (d.fc_pw, d.fc_pk)), ("완전 예측 MILP", (d.act_pw, d.act_pk))):
                    plan = solve_day(d, None, coefs, base=b, max_delay=k)[0]
                    c = true_cost(d, d.act_pw, d.act_pk, plan, coefs)
                    rows.append(dict(period=period, day=str(d.day.date()), max_delay=k, plan=name,
                                     saving_won=base["total"] - c["total"], energy_saving_won=base["energy"] - c["energy"],
                                     moved_share=float(np.abs(plan - d.prod).sum() / 2 / max(d.prod.sum(), 1))))
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "days.csv", index=False, encoding="utf-8-sig")
    t = r.groupby(["period", "plan", "max_delay"])[["saving_won", "energy_saving_won", "moved_share"]].mean().round(1)
    t.to_csv(OUT / "summary.csv", encoding="utf-8-sig")
    print(t.to_string())


if __name__ == "__main__":
    main()
