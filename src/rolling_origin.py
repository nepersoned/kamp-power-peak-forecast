"""롤링 원점 평가: 원본 구간을 한 주씩 밀며 "그 주 이전 데이터로 학습 → 그 주 예측"을 반복.
모델 순위가 특정 검증·테스트 구간에서만 나온 우연인지 확인한다 (2장).
출력: outputs/rolling/ (주별 MAE, 평균, 주별 1위 횟수)
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "outputs" / "rolling"
MODELS = ["naive_168h", "lgbm", "regime_lgbm", "regime_ens"]


def main(first="2021-07-12", last="2021-09-13"):
    import warnings
    warnings.filterwarnings("ignore")
    import run_all
    from .data import load
    from .features import build
    from .models import candidates

    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    X = build(df)
    run_all.COPY = df["is_copy"].to_numpy()
    t = df.index
    ok = (~df["outage"]).to_numpy() & X["power_lag168"].notna().to_numpy()
    rows = []
    for ws in pd.date_range(first, last, freq="7D"):
        we = ws + pd.Timedelta(days=7)
        fit = ok & (t >= run_all.START) & (t < ws)
        ev = ok & (t >= ws) & (t < we) & ~df["is_copy"].to_numpy()
        if ev.sum() == 0:
            continue
        for tgt in ("power", "peak15"):
            y = df.loc[ev, tgt].to_numpy()
            for m in MODELS:
                p, _ = run_all.fit_predict(candidates()[m], X, df[tgt], df["hour"], fit, ev)
                rows.append(dict(week=str(ws.date()), target=tgt, model=m, hours=int(ev.sum()),
                                 MAE=float(np.abs(y - p).mean())))
        print(f"week {ws.date()} done", flush=True)
    r = pd.DataFrame(rows)
    r.to_csv(OUT / "weekly_mae.csv", index=False, encoding="utf-8-sig")
    piv = r.pivot_table(index=["target", "week"], columns="model", values="MAE").round(2)
    print(piv.to_string())
    summ = []
    for tgt, g in r.groupby("target"):
        w = g.pivot_table(index="week", columns="model", values="MAE")
        wins = w.idxmin(axis=1).value_counts()
        for m in MODELS:
            summ.append(dict(target=tgt, model=m, mean_MAE=w[m].mean(), median_MAE=w[m].median(),
                             weeks_best=int(wins.get(m, 0)), weeks=len(w),
                             beats_lgbm_weeks=int((w[m] < w["lgbm"]).sum()) if m != "lgbm" else np.nan))
    s = pd.DataFrame(summ).round(2)
    s.to_csv(OUT / "summary.csv", index=False, encoding="utf-8-sig")
    print(s.to_string(index=False))


if __name__ == "__main__":
    main()
