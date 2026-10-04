"""롤링 원점 평가: 원본 구간을 한 주씩 밀며 "그 주 이전 데이터로 학습 → 그 주 예측"을 반복.
모델 순위가 특정 검증·테스트 구간에서만 나온 우연인지 확인한다 (2장).
출력: outputs/rolling/ (주별 MAE, 평균, 주별 1위 횟수)
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs" / "rolling"
MODELS = ["naive_168h", "lgbm", "regime_lgbm", "regime_ens"]


def main(first="2021-07-12", last="2021-09-13", models=None, tag=""):
    """models: 비교할 모델 이름(기본 MODELS). tag: 출력 파일 접미사.
    Regime-TabPFN은 검증에서 선정을 끝낸 뒤의 강건성 확인용으로만 돌린다(선정에 쓰지 않음)."""
    import warnings
    warnings.filterwarnings("ignore")
    import run_all
    from .data import load
    from .features import build
    from .forecast_adapters import make_forecaster

    models = list(models or MODELS)

    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    X = build(df)
    run_all.COPY = df["is_copy"].to_numpy()
    t = df.index
    ok = (~df["outage"]).to_numpy() & X["power_lag168"].notna().to_numpy()
    rows, daily = [], []
    for ws in pd.date_range(first, last, freq="7D"):
        we = ws + pd.Timedelta(days=7)
        fit = ok & (t >= run_all.START) & (t < ws)
        ev = ok & (t >= ws) & (t < we) & ~df["is_copy"].to_numpy()
        if ev.sum() == 0:
            continue
        for tgt in ("power", "peak15"):
            y = df.loc[ev, tgt].to_numpy()
            for m in models:
                p, _ = run_all.fit_predict(lambda m=m: make_forecaster(m), X, df[tgt], df["hour"], fit, ev)
                rows.append(dict(week=str(ws.date()), target=tgt, model=m, hours=int(ev.sum()),
                                 MAE=float(np.abs(y - p).mean())))
                daily.append(pd.DataFrame(dict(date=t[ev].normalize(), target=tgt, model=m, err=np.abs(y - p))))
        print(f"week {ws.date()} done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / f"weekly_mae{tag}.csv", index=False, encoding="utf-8-sig")
    d = pd.concat(daily).groupby(["date", "target", "model"])["err"].mean().reset_index()
    d.to_csv(OUT / f"daily_mae{tag}.csv", index=False, encoding="utf-8-sig")
    piv = r.pivot_table(index=["target", "week"], columns="model", values="MAE").round(2)
    print(piv.to_string())
    summ = []
    for tgt, g in r.groupby("target"):
        w = g.pivot_table(index="week", columns="model", values="MAE")
        wins = w.idxmin(axis=1).value_counts()
        for m in models:
            summ.append(dict(target=tgt, model=m, mean_MAE=w[m].mean(), median_MAE=w[m].median(),
                             weeks_best=int(wins.get(m, 0)), weeks=len(w),
                             beats_lgbm_weeks=int((w[m] < w["lgbm"]).sum()) if "lgbm" in w and m != "lgbm" else np.nan))
    s = pd.DataFrame(summ).round(2)
    s.to_csv(OUT / f"summary{tag}.csv", index=False, encoding="utf-8-sig")
    print(s.to_string(index=False))


if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1:      # 예: python -m src.rolling_origin regime_ens regime_tabpfn_all
        main(models=sys.argv[1:], tag="_" + "_vs_".join(sys.argv[1:]))
    else:
        main()
