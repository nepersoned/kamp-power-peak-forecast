"""제출 보고서용 그림 (저장된 결과 CSV만 읽는다. 모델을 학습하거나 고르지 않는다).

    python -m src.report_figures      -> outputs/report/r*.png

입력: outputs/forecast_models/, outputs/final_system/, outputs/final_analysis/, outputs/rolling/
"""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyBboxPatch

ROOT = Path(__file__).resolve().parents[1]
O = ROOT / "outputs"
OUT = O / "report"
plt.rcParams.update({"font.family": "Malgun Gothic", "axes.unicode_minus": False, "font.size": 10,
                     "axes.spines.top": False, "axes.spines.right": False})
ENS, TAB = "#9aa3ad", "#d9731a"          # 비교 모델 회색, 최종 모델 주황
NAME = {"regime_ens": "레짐 앙상블", "regime_tabpfn_all": "Regime-TabPFN(최종)"}


def save(fig, name):
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=200, bbox_inches="tight")
    plt.close(fig)


def pipeline():
    steps = [("전날 확정 정보", "생산계획·달력\n기상·과거 실측"), ("운전 레짐 판별", "생산계획 있으면 가동일\n없으면 휴무일"),
             ("Regime-TabPFN", "가동일 TabPFN\n휴무일 시간대 기저부하"), ("오차 시나리오 30개", "과거 24시간 오차\n경로를 통째로"),
             ("래칫 인지 확률 MILP", "최대 2시간 이동\n일 총량 동일"), ("현장 조치", "위험 등급 경보\n권고 생산계획")]
    fig, ax = plt.subplots(figsize=(11, 2.3))
    ax.set_xlim(0, len(steps)); ax.set_ylim(0, 1); ax.axis("off")
    for i, (t, s) in enumerate(steps):
        c = TAB if i in (2, 4) else "#4a6fa5"
        ax.add_patch(FancyBboxPatch((i + .06, .12), .82, .76, boxstyle="round,pad=0.02", fc=c, ec="none", alpha=.92))
        ax.text(i + .47, .66, t, ha="center", va="center", color="white", weight="bold", fontsize=10)
        ax.text(i + .47, .36, s, ha="center", va="center", color="white", fontsize=8.5)
        if i < len(steps) - 1:
            ax.annotate("", (i + 1.06, .5), (i + .9, .5), arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.4))
    save(fig, "r1_pipeline.png")


def model_compare():
    valid = {"전일 같은 시각": 37.38, "Ridge": 34.92, "SARIMAX": 32.88, "전주 같은 시각": 27.55, "LightGBM 단일": 23.30,
             "TabPFN 단일": 18.32, "레짐 LightGBM": 14.47, "레짐 앙상블": 10.24, "Regime-TabPFN": 8.38}
    test = {"전주 같은 시각": 8.26, "LightGBM 단일": 6.84, "레짐 LightGBM": 6.65, "레짐 앙상블": 6.23, "Regime-TabPFN": 5.77}
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw=dict(width_ratios=[1.25, 1]))
    for ax, d, title in ((axs[0], valid, "검증 점수(2 fold × 2 타깃 MAE 평균, kW) — 모델 선정 기준"),
                         (axs[1], test, "테스트 전력 MAE(kW) — 동결 후 1회")):
        k = list(d)
        ax.barh(k, list(d.values()), color=[TAB if n == "Regime-TabPFN" else ENS for n in k])
        for i, v in enumerate(d.values()):
            ax.text(v, i, f" {v:.2f}", va="center", fontsize=9)
        ax.set_title(title, fontsize=10); ax.set_xlim(0, max(d.values()) * 1.18)
    save(fig, "r2_model_compare.png")


def regime_effect():
    fig, ax = plt.subplots(figsize=(5.2, 3.2))
    x = np.arange(2)
    ax.bar(x - .18, [23.30, 18.32], .36, color="#c9ced4", label="레짐 없음(단일 모델)")
    ax.bar(x + .18, [14.47, 8.38], .36, color=[ENS, TAB], label="레짐 전환")
    for xi, (a, b) in enumerate(((23.30, 14.47), (18.32, 8.38))):
        ax.text(xi - .18, a, f"{a:.1f}", ha="center", va="bottom"); ax.text(xi + .18, b, f"{b:.1f}", ha="center", va="bottom")
        ax.text(xi, max(a, b) + 3, f"-{100 * (a - b) / a:.0f}%", ha="center", weight="bold")
    ax.set_xticks(x, ["LightGBM", "TabPFN"]); ax.set_ylabel("검증 점수(kW)"); ax.set_ylim(0, 30)
    ax.legend(frameon=False, fontsize=8.5); ax.set_title("레짐 구조 × 가동일 모델", fontsize=10)
    save(fig, "r3_regime_effect.png")


def conditions():
    c = pd.read_csv(O / "final_analysis/conditions.csv")
    pick = [("transition", "3일+ 휴무 후 재가동", "3일+ 휴무 후\n재가동일"), ("transition", "휴무 전날", "휴무 전날"),
            ("monday", "월요일", "월요일"), ("transition", "평상", "평상일"),
            ("peak_window", "주간 피크대", "주간 피크대\n(08~11·13~16시)"), ("tariff_band", "최대부하", "최대부하\n요금 시간")]
    fig, axs = plt.subplots(1, 2, figsize=(11, 3.4), sharey=True)
    for ax, split, title in ((axs[0], "valid_fold2", "검증(7/16~8/15, 하계휴가 포함)"), (axs[1], "test", "테스트(8/16~9/14, 평상 운전)")):
        g = c[c.split == split]
        rows = [g[(g.condition == a) & (g.level == b)] for a, b, _ in pick]
        lab = [l for (_, _, l), r in zip(pick, rows) if len(r)]
        rows = [r.iloc[0] for r in rows if len(r)]
        x = np.arange(len(rows))
        ax.bar(x - .2, [r.regime_ens for r in rows], .4, color=ENS, label=NAME["regime_ens"])
        ax.bar(x + .2, [r.regime_tabpfn_all for r in rows], .4, color=TAB, label=NAME["regime_tabpfn_all"])
        ax.set_xticks(x, lab, fontsize=8); ax.set_title(title, fontsize=10)
    axs[0].set_ylabel("전력 MAE(kW)"); axs[0].legend(frameon=False, fontsize=8.5)
    save(fig, "r4_conditions.png")


def rolling():
    f = O / "rolling/weekly_mae_regime_ens_vs_regime_tabpfn_all.csv"
    if not f.exists():
        return
    r = pd.read_csv(f)
    w = r.pivot_table(index="week", columns="model", values="MAE").reset_index()  # 두 타깃 평균
    fig, ax = plt.subplots(figsize=(11, 3.0))
    x = np.arange(len(w))
    ax.bar(x - .2, w["regime_ens"], .4, color=ENS, label=NAME["regime_ens"])
    ax.bar(x + .2, w["regime_tabpfn_all"], .4, color=TAB, label=NAME["regime_tabpfn_all"])
    ax.axvspan(-.5, 4.5, color="#eef2f7", zorder=0); ax.text(2, ax.get_ylim()[1] * .92, "검증 구간 포함 주", ha="center", fontsize=8.5)
    ax.set_xticks(x, [s[5:] + "~" for s in w["week"]], fontsize=8.5); ax.set_ylabel("주별 MAE(kW, 두 타깃 평균)")
    ax.set_title("주 단위 롤링 평가: 매주 그 이전 데이터로 재학습 → 그 주 예측(원본 날짜)", fontsize=10)
    ax.legend(frameon=False, fontsize=8.5)
    save(fig, "r5_rolling.png")


def importance():
    f = O / "final_analysis/group_importance_peak15.csv"
    if not f.exists():
        return
    r = pd.read_csv(f)
    order = r[r.model == "regime_tabpfn_all"].sort_values("mae_increase").group.tolist()
    fig, ax = plt.subplots(figsize=(7.5, 3.4))
    y = np.arange(len(order))
    for m, off, col in (("regime_ens", -.2, ENS), ("regime_tabpfn_all", .2, TAB)):
        g = r[r.model == m].set_index("group").loc[order]
        ax.barh(y + off, g.mae_increase, .4, color=col, label=NAME[m])
    ax.set_yticks(y, order); ax.set_xlabel("섞었을 때 15분 최대수요 MAE 증가(kW)")
    ax.set_title("피처 그룹 순열 중요도(테스트 가동일)", fontsize=10); ax.legend(frameon=False, fontsize=8.5)
    save(fig, "r6_importance.png")


def value():
    """연간 가치를 성격별로 나눠 표시(합산하지 않음)."""
    items = [("평상시 부하 이동\n(운영 절감, 연 250일)", 9_242 * 250, "#d9731a"), ("래칫 갱신 1건 회피\n(사건 발생 시)", 1_634_756, "#e8a56a"),
             ("요금제 Ⅱ 유지·변경\n(현재 Ⅰ·Ⅲ일 때만)", 1_100_000, "#4a6fa5"), ("소형 ESS 22kW\n(설치비 차감 전)", 2_200_000, "#8aa2c8")]
    fig, ax = plt.subplots(figsize=(8, 3.1))
    ax.bar([k for k, _, _ in items], [v / 1e4 for _, v, _ in items], color=[c for *_, c in items])
    for i, (_, v, _) in enumerate(items):
        ax.text(i, v / 1e4, f"{v / 1e4:,.0f}만 원", ha="center", va="bottom")
    ax.axvline(1.5, color="#888", ls=":", lw=1)
    ax.text(0.5, 262, "모델 기반 생산 조정", ha="center", fontsize=9, color="#d9731a")
    ax.text(2.5, 262, "계약·설비 판단(모델과 별개)", ha="center", fontsize=9, color="#4a6fa5")
    ax.set_ylabel("연간 금액(만 원)"); ax.set_ylim(0, 285); ax.tick_params(axis="x", labelsize=8.5)
    ax.set_title("성격별 연간 가치 추정(서로 더하지 않음)", fontsize=10)
    save(fig, "r7_value.png")


def surrogate():
    """권고 계획 평가에 쓰는 대리모형: 시간대별로 '가동하면' 15분 최대수요가 얼마나 오르는가."""
    import json
    c = json.loads((O / "final_system/surrogate_coefficients.json").read_text())["peak15"]
    g = np.array(c["gain"])
    band = ["경"] * 9 + ["중"] + ["최"] * 2 + ["중"] + ["최"] * 4 + ["중"] * 6 + ["경"]   # 여름 요금 시간대(09~23시 기준)
    col = {"경": "#c9ced4", "중": "#f2b134", "최": "#d9731a"}
    fig, ax = plt.subplots(figsize=(9, 3.0))
    ax.bar(range(24), g, color=[col[b] for b in band])
    ax.axhline(0, color="#333", lw=0.8)
    ax.set_xticks(range(0, 24, 2)); ax.set_xlabel("시각"); ax.set_ylabel("가동 시 15분 최대수요 증분(kW)")
    ax.set_title(f"대리모형: 그 시각에 가동하면 +kW (생산 1,000단위당 +{c['beta'] * 1000:.1f}kW, 가동 시작 시 +{c['start']:.0f}kW)", fontsize=10)
    from matplotlib.patches import Patch
    ax.legend(handles=[Patch(color=col[k], label=v) for k, v in (("최", "최대부하"), ("중", "중간부하"), ("경", "경부하"))], frameon=False, fontsize=8.5, ncol=3)
    save(fig, "r8_surrogate.png")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for f in (pipeline, model_compare, regime_effect, conditions, rolling, importance, value, surrogate):
        f()
    print(sorted(p.name for p in OUT.glob("*.png")))


if __name__ == "__main__":
    main()
