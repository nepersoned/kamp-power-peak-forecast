"""요금 선택(Ⅰ/Ⅱ/Ⅲ) 비교와 ESS 규모 산정 (4장: 운영을 바꾸지 않는 절감 / 설비 투자 판단).

1. 요금 선택: 같은 실측 부하로 선택 Ⅰ/Ⅱ/Ⅲ의 월 요금(전력량 + 래칫 기본요금)을 계산.
   데이터가 1~9월(약 8.5개월)이라 연간 환산은 해당 기간 합계 × 12/기간 개월 수로 단순 환산(가정).
   부하율(평균/최대수요)과 월 사용시간(kWh/kW)도 함께 제시 — 선택Ⅲ은 사용시간이 긴 고객에 유리.
2. ESS 규모: 기본요금 산정 시간(경부하 제외)의 15분 수요를 목표값 C 이하로 깎는 데 필요한
   일별 방전량(kWh)·출력(kW)의 최대치 → 최대수요를 (원래 최대 − C)만큼 낮출 때의 배터리 요구량과 연간 기본요금 절감.
   충전은 경부하 시간에 가능하다고 가정, 효율·열화·설치비는 미반영(보고서에 한계로 명시).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from . import tariff as T

OUT = Path(__file__).resolve().parents[1] / "outputs" / "tariff"


def option_bills(q):
    rows = []
    for opt in ("I", "II", "III"):
        b = T.monthly_bill(q, option=opt)
        rows.append(dict(option=opt, energy_won=b["energy_won"].sum(), base_won=b["base_won"].sum(),
                         total_won=b["total_won"].sum(), months=len(b)))
    t = pd.DataFrame(rows)
    t["annualized_won"] = t["total_won"] * 12 / t["months"]
    t["vs_II_won"] = t["annualized_won"] - t.loc[t.option == "II", "annualized_won"].iloc[0]
    return t


def ess_sizing(q, targets):
    """목표 최대수요 C별: 필요한 배터리 출력(kW)·용량(kWh), 연간 기본요금 절감(래칫 고려 전후 최대수요 차이 기준)."""
    bands = T.hourly_bands(q.index)["demand_band"].to_numpy() > T.OFF
    v = q[T.QUARTERS].to_numpy()                                  # 시간 × 4 (15분 kW)
    over = np.clip(v - 0, 0, None)
    day = q.index.normalize()
    peak = float(v[bands].max())
    rows = []
    for C in targets:
        ex = np.where(bands[:, None], np.clip(v - C, 0, None), 0.0)   # kW 초과
        kwh_day = pd.Series(ex.sum(axis=1) * 0.25, index=q.index).groupby(day).sum()
        power_need = float(ex.max())
        rows.append(dict(target_kw=C, shave_kw=peak - C, battery_kw=power_need, battery_kwh=float(kwh_day.max()),
                         days_needing_discharge=int((kwh_day > 0).sum()),
                         base_saving_won_per_year=(peak - C) * T.BASE_RATE["II"] * 12))
    return pd.DataFrame(rows)


def main():
    from .data import load
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    q = df.loc[~df["outage"], T.QUARTERS]
    t = option_bills(df[T.QUARTERS])
    kwh = (df[T.QUARTERS].sum(axis=1) * 0.25).sum()
    peak = T.demand_peak(df[T.QUARTERS]).max()
    hours = len(df) / 24 * 24
    lf = (df[T.QUARTERS].mean(axis=1).mean()) / peak
    use_h = kwh / (len(df) / 24 / 30.4) / peak
    t.round(0).to_csv(OUT / "option_compare.csv", index=False, encoding="utf-8-sig")
    print(t.round(0).to_string(index=False))
    print(f"부하율 {lf:.2f}, 월 사용시간 약 {use_h:.0f}시간 (kWh/최대수요kW)")
    e = ess_sizing(q, [215, 210, 205, 200, 195, 190])
    e.round(1).to_csv(OUT / "ess_sizing.csv", index=False, encoding="utf-8-sig")
    print(e.round(1).to_string(index=False))


if __name__ == "__main__":
    main()
