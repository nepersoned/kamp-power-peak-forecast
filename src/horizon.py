"""예측 구간(H시간 앞)별 성능 곡선 — 문제의 "향후 지정된 시간구간"에 대한 근거.

H = 1, 3, 6, 12, 24시간 앞. H마다 따로 학습하는 직접(direct) 모델.
  피처 = 하루 전 피처(시차 24시간 이상, 생산계획, 달력, 기상) + H시간 전까지의 최근 실측
         (H, H+1, H+2시간 전 전력·15분최대, H시간 전 기준 최근 3시간 평균)
  모델 = 레짐 전환 LightGBM(L1) — 가동일/휴무일 분리
  기준선 = 지속 예측(H시간 전 값), 전주 같은 시각
평가 = 검증(학습 구간 학습), 테스트(학습+검증 학습). 출력: outputs/horizon/
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs" / "horizon"
HORIZONS = (1, 3, 6, 12, 24)


def add_recent(X, df, H):
    Z = X.copy()
    for tgt, name in (("power", "p"), ("peak15", "k")):
        for j in (0, 1, 2):
            Z[f"recent_{name}_lag{H + j}"] = df[tgt].shift(H + j)
        Z[f"recent_{name}_mean3_lag{H}"] = df[tgt].shift(H).rolling(3, min_periods=1).mean()
    return Z


def main():
    import warnings
    warnings.filterwarnings("ignore")
    import matplotlib.pyplot as plt
    import run_all
    from .data import load
    from .features import build
    from .models import RegimeModel
    from .plots import C_ALT, C_GRAY, C_MAIN, _save

    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    X0 = build(df)
    run_all.COPY = df["is_copy"].to_numpy()
    tr, val, te = run_all.masks(df, X0)
    rows = []
    for H in HORIZONS:
        X = add_recent(X0, df, H)
        for period, fit_mask, ev_mask in (("valid", tr, val), ("test", tr | val, te)):
            ok = ev_mask & X[f"recent_p_lag{H + 2}"].notna().to_numpy()
            for tgt in ("power", "peak15"):
                y = df.loc[ok, tgt]
                model, _ = run_all.fit_predict(RegimeModel, X, df[tgt], df["hour"], fit_mask, ok)
                persist = df[tgt].shift(H)[ok]
                weekly = df[tgt].shift(168)[ok]
                for name, p in (("레짐 LightGBM", model), ("지속 예측(H시간 전 값)", persist.to_numpy()),
                                ("전주 같은 시각", weekly.to_numpy())):
                    rows.append(dict(H=H, period=period, target=tgt, model=name,
                                     MAE=float(np.nanmean(np.abs(y.to_numpy() - p)))))
        print(f"H={H} done", flush=True)
    t = pd.DataFrame(rows)
    t.round(2).to_csv(OUT / "horizon_mae.csv", index=False, encoding="utf-8-sig")
    piv = t.pivot_table(index=["period", "target", "model"], columns="H", values="MAE").round(2)
    piv.to_csv(OUT / "horizon_mae_pivot.csv", encoding="utf-8-sig")
    print(piv.to_string())

    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), sharey=False)
    for ax, period in zip(axes, ("valid", "test")):
        for name, c in (("레짐 LightGBM", C_MAIN), ("지속 예측(H시간 전 값)", C_ALT), ("전주 같은 시각", C_GRAY)):
            s = t[(t.period == period) & (t.target == "power") & (t.model == name)].sort_values("H")
            ax.plot(s["H"], s["MAE"], marker="o", color=c, label=name)
        ax.set(xlabel="예측 구간 H (시간 앞)", ylabel="전력 MAE", xticks=list(HORIZONS),
               title=f"예측 구간별 오차 ({'검증: 휴무 주간 포함' if period == 'valid' else '테스트: 평상 운전'})")
        ax.legend(fontsize=8, frameon=False)
    _save(fig, OUT, "horizon_curve.png")


if __name__ == "__main__":
    main()
