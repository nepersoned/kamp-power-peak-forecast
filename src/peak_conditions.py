"""피크 발생 조건 분석 (3장 · 강화학습 상태·보상 설계 근거).

원본일(복제 제외)·정전 제외, 목표수요 TAU(190) 이상 15분 수요를 피크 이벤트로 본다.
출력: outputs/peak_conditions/ 에 표(csv)와 그림 4종.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .data import load
from .features import build
from .peak import TAU
from .plots import C_ALT, C_GRAY, C_MAIN, C_RED, _save

OUT = Path(__file__).resolve().parents[1] / "outputs" / "peak_conditions"
DOW = ["월", "화", "수", "목", "금", "토", "일"]


def frame():
    df = load()
    X = build(df)
    o = df[~df["is_copy"] & ~df["outage"]].copy()
    o["on_day"] = (X.loc[o.index, "plan_on_day"].fillna(1) == 1).to_numpy()
    o["event"] = (o["peak15"] >= TAU).astype(int)
    day = o.index.normalize()
    o["day_prod_hours"] = (o["prod"] > 0).groupby(day).transform("sum")
    o["prev_prod"] = o.groupby(day)["prod"].shift(1).fillna(0)
    return o


def daily(o):
    on = o[o["on_day"]]
    d = on.groupby(on.index.normalize()).agg(
        peak=("peak15", "max"), prod=("prod", "sum"), tmax=("temp", "max"), dow=("dow", "first"),
        month=("month", "first"), prod_hours=("day_prod_hours", "first"),
        peak_hour=("peak15", lambda s: int(s.idxmax().hour)))
    return d[d["prod"] > 0]


def tables(o, d):
    on = o[o["on_day"]]
    t = {}
    t["by_hour"] = on.groupby("hour")["event"].agg(rate="mean", n="size", events="sum")
    t["by_dow"] = on.groupby("dow")["event"].agg(rate="mean", n="size", events="sum").rename(index=dict(enumerate(DOW)))
    t["by_month"] = on.groupby("month")["event"].agg(rate="mean", n="size", events="sum")
    pb = pd.cut(on["prod"], [-1, 0, 500, 1500, 1e9], labels=["0", "1-500", "501-1500", ">1500"])
    t["by_prod"] = on.groupby(pb, observed=True)["event"].agg(rate="mean", n="size", events="sum")
    start = (on["prev_prod"] == 0) & (on["prod"] > 0)
    cont = (on["prev_prod"] > 0) & (on["prod"] > 0)
    t["startup_vs_continuing"] = pd.DataFrame({
        "rate": [on.loc[start, "event"].mean(), on.loc[cont, "event"].mean()],
        "n": [int(start.sum()), int(cont.sum())]}, index=["가동 첫 시간", "생산 지속 시간"])
    ev = on[on["event"] == 1]
    t["max_quarter_in_event_hour"] = ev[["q15", "q30", "q45", "q60"]].idxmax(axis=1).value_counts().rename("count").to_frame()
    t["daily_by_prod_hours"] = d.groupby(pd.cut(d["prod_hours"], [0, 6, 12, 19, 24]), observed=True)["peak"].agg(
        days="size", mean_peak="mean", max_peak="max", share_over_tau=lambda s: (s >= TAU).mean())
    t["daily_corr"] = pd.DataFrame({"corr_with_daily_peak": [d["peak"].corr(d["prod"]), d["peak"].corr(d["prod_hours"]),
                                                             d["peak"].corr(d["tmax"])]},
                                   index=["일 생산량", "일 생산시간", "일 최고기온"])
    return t


def figures(o, d, t):
    on = o[o["on_day"]]
    # 1. 시간대별 피크 확률 + 평균 15분 최대수요
    fig, ax = plt.subplots(figsize=(9, 3.4))
    r = t["by_hour"]["rate"]
    ax.bar(r.index, r.values, color=C_RED, alpha=0.8, label=f"피크(≥{TAU}) 확률")
    ax.set(xlabel="시각", ylabel="피크 확률", title="가동일 시간대별 피크 발생 확률과 평균 수요")
    ax2 = ax.twinx()
    m = on.groupby("hour")["peak15"].mean()
    ax2.plot(m.index, m.values, color=C_MAIN, marker="o", ms=3, label="평균 15분 최대수요")
    ax2.set_ylabel("15분 최대수요")
    h1, l1 = ax.get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, fontsize=8, frameon=False, loc="upper left")
    _save(fig, OUT, "pc1_hour.png")

    # 2. 요일 × 월 피크 확률
    pv = on.pivot_table(index="dow", columns="month", values="event", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(6.5, 3.6))
    im = ax.imshow(pv.values, cmap="Reds", vmin=0, aspect="auto")
    ax.set_xticks(range(pv.shape[1]), [f"{m}월" for m in pv.columns])
    ax.set_yticks(range(pv.shape[0]), [DOW[i] for i in pv.index])
    for i in range(pv.shape[0]):
        for j in range(pv.shape[1]):
            v = pv.values[i, j]
            if np.isfinite(v):
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(im, label="피크 확률")
    ax.set_title("요일 × 월 피크 발생 확률 (가동일, 원본일)")
    _save(fig, OUT, "pc2_dow_month.png")

    # 3. 일 생산시간 vs 일 최대수요
    fig, ax = plt.subplots(figsize=(6.5, 4))
    col = np.where(d["peak"] >= TAU, C_RED, C_GRAY)
    ax.scatter(d["prod_hours"], d["peak"], c=col, s=22)
    ax.axhline(TAU, color=C_RED, ls=":", lw=1)
    ax.set(xlabel="일 생산시간 (생산량>0인 시간 수)", ylabel="일 최대 15분 수요",
           title=f"일 생산시간과 일 최대수요 (상관 {d['peak'].corr(d['prod_hours']):.2f})")
    _save(fig, OUT, "pc3_prod_hours.png")

    # 4. 08시 15분 단위 상승 + 이벤트 시간의 최대 15분 위치
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    for h, c in [(7, C_GRAY), (8, C_RED), (13, C_ALT)]:
        v = on[on["hour"] == h][["q15", "q30", "q45", "q60"]].mean()
        axes[0].plot(["15분", "30분", "45분", "60분"], v.values, marker="o", color=c, label=f"{h:02d}시")
    axes[0].set(ylabel="평균 15분 수요", title="가동일 시간 내 15분 수요 변화")
    axes[0].legend(fontsize=8, frameon=False)
    q = t["max_quarter_in_event_hour"]["count"].reindex(["q15", "q30", "q45", "q60"]).fillna(0)
    axes[1].bar(["15분", "30분", "45분", "60분"], q.values, color=C_RED)
    axes[1].set(ylabel="건수", title="피크 시간에서 최대가 찍힌 15분 구간")
    _save(fig, OUT, "pc4_quarter.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    o = frame()
    d = daily(o)
    t = tables(o, d)
    with pd.ExcelWriter(OUT / "peak_conditions.xlsx") as w:
        for k, v in t.items():
            v.round(3).to_excel(w, sheet_name=k[:31])
    d.round(1).to_csv(OUT / "daily_peaks.csv", encoding="utf-8-sig")
    figures(o, d, t)
    for k in ["by_hour", "daily_by_prod_hours", "daily_corr", "startup_vs_continuing"]:
        print(f"\n[{k}]\n{t[k].round(3).to_string()}")
    return t


if __name__ == "__main__":
    main()
