"""운영 대시보드 예시: 날짜 하나 → 전날 저녁 운영자가 보는 한 장짜리 보고서(PNG).

  (1) 내일 시간별 15분 최대수요 예측과 오차 시나리오 범위(10~90%), 요금 시간대, 래칫 바닥선
  (2) 래칫 초과 확률 = 과거 예측 오차 경로를 더한 시나리오 중 기본요금 시간 최대수요가 바닥을 넘는 비율
  (3) 권고 생산계획(시나리오 평균 래칫을 최소화하는 확률 MILP) vs 원래 계획
  (4) 요약: 위험 등급, 권고 조치, 예상 전력량요금 절감, 권고 후 초과 확률
출력: outputs/dashboard/dashboard_YYYY-MM-DD.png
사용: python -m src.dashboard 2021-07-19 2021-08-24
"""
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from . import tariff as T
from .decision_eval import true_cost
from .milp import _S_numeric
from .plots import C_ALT, C_GRAY, C_MAIN, C_RED, _save
from .stochastic_milp import residual_paths, solve_stochastic

OUT = Path(__file__).resolve().parents[1] / "outputs" / "dashboard"
BAND_COLOR = {T.OFF: "#eef3f6", T.MID: "#fdf3e1", T.PEAK: "#fbe3dc"}
BAND_NAME = {T.OFF: "경부하", T.MID: "중간부하", T.PEAK: "최대부하"}


def exceed_prob(d, plan, scen, coefs):
    dpk = _S_numeric(plan, coefs["peak15"]) - _S_numeric(d.prod, coefs["peak15"])
    s = scen + dpk[None, :]
    billable = d.demand_band > T.OFF
    peaks = np.where(billable[None, :], s, 0).max(axis=1)
    return float((peaks > d.floor).mean())


def expected_excess(d, plan, scen, coefs):
    """시나리오 평균 래칫 초과량(kW) = 확률 MILP가 실제로 줄이는 값(초과 확률이 아니라 초과 '크기'의 기대값)."""
    dpk = _S_numeric(plan, coefs["peak15"]) - _S_numeric(d.prod, coefs["peak15"])
    peaks = np.where((d.demand_band > T.OFF)[None, :], scen + dpk[None, :], 0).max(axis=1)
    return float(np.clip(peaks - d.floor, 0, None).mean())


def render(d, coefs, pool):
    scen = d.fc_pk[None, :] + pool
    plan = solve_stochastic(d, coefs, scen, lam=0.0, time_limit_s=60)
    p0, p1 = exceed_prob(d, d.prod, scen, coefs), exceed_prob(d, plan, scen, coefs)
    dpk = _S_numeric(plan, coefs["peak15"]) - _S_numeric(d.prod, coefs["peak15"])
    dpw = _S_numeric(plan, coefs["power"]) - _S_numeric(d.prod, coefs["power"])
    exp_energy = float((d.rate * (-dpw)).sum())            # 예측 기준 전력량요금 절감(원)
    realized = true_cost(d, d.act_pw, d.act_pk, d.prod, coefs)["total"] - true_cost(d, d.act_pw, d.act_pk, plan, coefs)["total"]
    level = "높음" if p0 >= 0.2 else ("주의" if p0 >= 0.05 else "낮음")
    h = np.arange(24)

    fig = plt.figure(figsize=(12, 8.2))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.25, 1.0, 0.55], hspace=0.55, wspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])
    for t in h:
        ax1.axvspan(t - 0.5, t + 0.5, color=BAND_COLOR[d.energy_band[t]], lw=0)
    lo, hi = np.percentile(scen, 10, axis=0), np.percentile(scen, 90, axis=0)
    ax1.fill_between(h, lo, hi, color=C_MAIN, alpha=0.15, label="오차 시나리오 10~90%")
    ax1.plot(h, d.fc_pk, color=C_MAIN, marker="o", ms=3, label="원래 계획 예측")
    ax1.plot(h, d.fc_pk + dpk, color=C_ALT, marker="o", ms=3, label="권고 계획 예측")
    ax1.axhline(d.floor, color=C_RED, ls="--", lw=1.2, label=f"래칫 바닥 {d.floor:.0f}kW")
    ax1.set(xlim=(-0.5, 23.5), xticks=range(0, 24, 2), xlabel="시각", ylabel="15분 최대수요(kW)",
            title=f"{d.day:%Y-%m-%d (%a)}  내일 최대수요 예측과 요금 시간대 (배경: 경부하·중간·최대부하)")
    ax1.legend(fontsize=8, ncol=4, frameon=False, loc="upper left")

    ax2 = fig.add_subplot(gs[1, :2])
    ax2.bar(h - 0.2, d.prod, width=0.4, color=C_GRAY, label="원래 생산계획")
    ax2.bar(h + 0.2, plan, width=0.4, color=C_ALT, label="권고 생산계획")
    ax2.set(xticks=range(0, 24, 2), xlabel="시각", ylabel="생산량", title="생산계획 조정 (최대 2시간 이동, 일 총량 동일)")
    ax2.legend(fontsize=8, frameon=False)

    ax3 = fig.add_subplot(gs[1, 2])
    ax3.bar(["원래 계획", "권고 계획"], [p0 * 100, p1 * 100], color=[C_GRAY, C_ALT])
    for i, v in enumerate([p0 * 100, p1 * 100]):
        ax3.text(i, v + 1, f"{v:.0f}%", ha="center", fontsize=10)
    ax3.set(ylim=(0, max(100, p0 * 100 + 10)), ylabel="%", title="래칫 초과 확률")

    ax4 = fig.add_subplot(gs[2, :])
    ax4.axis("off")
    moved = np.abs(plan - d.prod).sum() / 2
    lines = [
        f"위험 등급: {level}   (내일 최대수요가 래칫 바닥 {d.floor:.0f}kW를 넘을 확률 {p0 * 100:.0f}% → 권고 시 {p1 * 100:.0f}%)",
        f"권고 조치: 생산 {moved:,.0f}단위({moved / max(d.prod.sum(), 1) * 100:.0f}%)를 최대 2시간 늦춤"
        + (" — 기본요금 시간대 피크 억제 우선" if p0 >= 0.05 else " — 최대부하 → 중간·경부하 전력량요금 절감"),
        f"예상 효과: 전력량요금 {exp_energy:,.0f}원/일 절감(예측 기준). 래칫 갱신 시 비용 = 초과 kW × {T.BASE_RATE['II']:,}원 × 12개월",
        f"사후 확인(실측 기준): 이날 권고를 따랐다면 {realized:,.0f}원 절감",
    ]
    for i, s in enumerate(lines):
        ax4.text(0.0, 0.9 - i * 0.27, s, fontsize=10.5, transform=ax4.transAxes,
                 fontweight="bold" if i == 0 else "normal", color=C_RED if (i == 0 and level == "높음") else "black")
    _save(fig, OUT, f"dashboard_{d.day:%Y-%m-%d}.png")
    import pandas as pd
    dpk_true = dpk  # 실측 반사실 = 실측 + 대리모형 변화분(decision_eval.true_cost와 같은 방식)
    pd.DataFrame({"hour": h, "tariff_band": [BAND_NAME[b] for b in d.energy_band], "prod_original": d.prod,
                  "prod_recommended": plan, "forecast_peak15_original": d.fc_pk, "forecast_peak15_recommended": d.fc_pk + dpk,
                  "actual_peak15": d.act_pk, "actual_peak15_if_recommended": np.clip(d.act_pk + dpk_true, 0, None)}
                 ).round(1).to_csv(OUT / f"hourly_{d.day:%Y-%m-%d}.csv", index=False, encoding="utf-8-sig")
    return dict(day=str(d.day.date()), level=level, p_exceed_before=p0, p_exceed_after=p1,
                expected_excess_kw_before=expected_excess(d, d.prod, scen, coefs), expected_excess_kw_after=expected_excess(d, plan, scen, coefs),
                expected_energy_saving=exp_energy, realized_saving=realized)


def main(dates):
    import warnings
    warnings.filterwarnings("ignore")
    from .ratchet_rl import build_days
    OUT.mkdir(parents=True, exist_ok=True)
    coefs, train_days, val_days, test_days = build_days()
    pool_v, pool_t = residual_paths(train_days), residual_paths(train_days + val_days)
    by_date = {str(d.day.date()): (d, pool_v) for d in val_days}
    by_date.update({str(d.day.date()): (d, pool_t) for d in test_days})
    for s in dates:
        if s not in by_date:
            print(f"{s}: 평가 가동일 아님"); continue
        d, pool = by_date[s]
        print(render(d, coefs, pool))


if __name__ == "__main__":
    main(sys.argv[1:] or ["2021-07-19", "2021-08-24"])
