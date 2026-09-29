"""보고서용 그림 생성 (outputs/fig*.png)."""
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from .peak import TAU  # noqa: E402

for f in ("Malgun Gothic", "AppleGothic", "NanumGothic"):
    if any(f == x.name for x in matplotlib.font_manager.fontManager.ttflist):
        plt.rcParams["font.family"] = f
        break
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["figure.dpi"] = 130

C_MAIN, C_ALT, C_GRAY, C_RED = "#2a6fdb", "#e08a1e", "#9aa0a6", "#d64545"


def _save(fig, out, name):
    fig.tight_layout()
    fig.savefig(out / name, bbox_inches="tight")
    plt.close(fig)


def make_all(df, X, pred, fr, days, pr, out):
    # 1. 데이터 개요: 일평균 전력, 복제일 표시, 구간 경계
    d = df.resample("D").agg(power=("power", "mean"), copy=("is_copy", "first"))
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(d.index, d["power"], color=C_GRAY, lw=0.8, zorder=1)
    ax.scatter(d.index[~d["copy"]], d["power"][~d["copy"]], s=9, color=C_MAIN, label="원본일", zorder=2)
    ax.scatter(d.index[d["copy"]], d["power"][d["copy"]], s=9, color=C_ALT, label="증강 복제일(동일 프로파일)", zorder=2)
    for t, lab in [("2021-07-16", "VALID"), ("2021-08-16", "TEST")]:
        ax.axvline(pd.Timestamp(t), color="k", ls="--", lw=0.8)
        ax.text(pd.Timestamp(t), ax.get_ylim()[1] * 0.97, f" {lab}", va="top", fontsize=9)
    ax.set_ylabel("일평균 전력")
    ax.set_title("일평균 전력사용량과 증강 복제일 분포")
    ax.legend(loc="lower left", fontsize=8, frameon=False)
    _save(fig, out, "fig1_data_overview.png")

    # 2. 운전 레짐별 시간 프로파일 + 피크 발생 시각
    orig = df[~df["is_copy"] & ~df["outage"]]
    on = X.loc[orig.index, "plan_on_day"].fillna(1) == 1
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.4))
    for mask, lab, c in [(on, "가동일(생산계획 있음)", C_MAIN), (~on, "휴무일(생산계획 없음)", C_ALT)]:
        g = orig.loc[mask].groupby("hour")["peak15"]
        axes[0].plot(g.mean().index, g.mean(), color=c, label=lab)
        axes[0].fill_between(g.mean().index, g.quantile(0.1), g.quantile(0.9), color=c, alpha=0.15)
    axes[0].axhline(TAU, color=C_RED, ls=":", lw=1)
    axes[0].text(0, TAU + 2, f"목표수요 TAU={TAU}", color=C_RED, fontsize=8)
    axes[0].set(xlabel="시각", ylabel="15분 최대수요", title="운전 레짐별 시간대 수요 (평균, 10~90%)")
    axes[0].legend(fontsize=8, frameon=False)
    ev = orig[orig["peak15"] >= TAU]["hour"].value_counts().reindex(range(24), fill_value=0)
    axes[1].bar(ev.index, ev.values, color=C_RED)
    axes[1].set(xlabel="시각", ylabel="건수", title=f"목표수요 초과(>= {TAU}) 발생 시각")
    _save(fig, out, "fig2_regime_profile.png")

    # 3. 테스트 구간 예측 vs 실측 (15분 최대수요, CQR 상한)
    fig, ax = plt.subplots(figsize=(11, 3.6))
    ax.plot(pred.index, pred["peak15_actual"], color="k", lw=0.8, label="실측")
    ax.plot(pred.index, pred["peak15_pred"], color=C_MAIN, lw=0.9, label="예측(제안 모델)")
    ax.fill_between(pred.index, pred["peak15_pred"], pred["peak15_upper_cqr"], color=C_MAIN, alpha=0.15,
                    label="P90 상한(컨포멀 보정)")
    ax.axhline(TAU, color=C_RED, ls=":", lw=1)
    ax.set(ylabel="15분 최대수요", title="테스트 구간(8/16~9/14) 시간별 15분 최대수요 예측")
    ax.legend(fontsize=8, ncol=3, frameon=False, loc="lower left")
    _save(fig, out, "fig3_test_forecast.png")

    # 4. 시간대별 오차(MAE, 편향)
    g = fr[fr["regime"] == "operating"].groupby("hour")
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.bar(g["abs_err"].mean().index, g["abs_err"].mean(), color=C_GRAY, label="MAE")
    ax.plot(g["err"].mean().index, g["err"].mean(), color=C_RED, marker="o", ms=3, label="편향(예측-실측)")
    ax.axhline(0, color="k", lw=0.6)
    ax.set(xlabel="시각", ylabel="오차", title="가동일 시간대별 예측오차 (테스트)")
    ax.legend(fontsize=8, frameon=False)
    _save(fig, out, "fig4_error_by_hour.png")

    # 5. SHAP 중요도
    imp = pd.read_csv(out / "shap_importance_peak15.csv", index_col=0).iloc[:, 0].head(15)[::-1]
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.barh(imp.index, imp.values, color=C_MAIN)
    ax.set(xlabel="평균 |SHAP|", title="15분 최대수요 모델 변수 기여도 (가동일, 테스트)")
    _save(fig, out, "fig5_shap_importance.png")

    # 6. 일 단위 위험 판정
    col = days["type"].map({"TP": C_MAIN, "FN": C_RED, "FP": C_ALT, "TN": C_GRAY})
    fig, ax = plt.subplots(figsize=(6.5, 4.5))
    ax.scatter(days["pred_max"], days["actual_max"], c=col, s=30)
    ax.axhline(TAU, color=C_RED, ls=":", lw=1)
    ax.axvline(pr["day_thr"], color="k", ls="--", lw=0.8)
    for t, c in [("TP", C_MAIN), ("FN", C_RED), ("FP", C_ALT), ("TN", C_GRAY)]:
        ax.scatter([], [], c=c, label=f"{t} ({int((days['type'] == t).sum())})")
    ax.set(xlabel="예측 일최대 수요", ylabel="실측 일최대 수요",
           title=f"일 단위 피크 위험 판정 (경보 임계 {pr['day_thr']:.0f})")
    ax.legend(fontsize=8, frameon=False)
    _save(fig, out, "fig6_daily_risk.png")

    # 7. 피크 저감 시뮬레이션 예시 (저감 폭이 가장 큰 날)
    s = pd.read_csv(out / "shaving_hourly_frac30.csv", index_col=0, parse_dates=True)
    gain = s["before"].resample("D").max() - s["after"].resample("D").max()
    day = gain.idxmax()
    sd = s.loc[day.strftime("%Y-%m-%d")]
    fig, ax = plt.subplots(figsize=(8, 3.2))
    ax.plot(sd.index.hour, sd["before"], color=C_GRAY, marker="o", ms=3, label="현행 계획")
    ax.plot(sd.index.hour, sd["after"], color=C_MAIN, marker="o", ms=3, label="생산 30% 이전 후")
    ax.axhline(TAU, color=C_RED, ls=":", lw=1)
    ax.set(xlabel="시각", ylabel="예측 15분 최대수요",
           title=f"피크 저감 시뮬레이션 예시 ({day:%m/%d}, 일최대 {gain.max():.1f} 감소)")
    ax.legend(fontsize=8, frameon=False)
    _save(fig, out, "fig7_shaving_example.png")
