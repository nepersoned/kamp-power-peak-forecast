"""최종 모델(Regime-TabPFN) 기준 보고서 보조 분석 (2.4·3.2·3.3절, 한계 절).

    python -m src.final_analysis --stage tables       # F1·조건별 오차 (저장된 예측 파일만 사용, 수 초)
    python -m src.final_analysis --stage importance   # 피처 그룹 순열 중요도 (TabPFN·앙상블, 수십 분)
    python -m src.final_analysis --stage plan         # 생산계획 가정 민감도 (TabPFN, 수십 분)

입력:
  outputs/forecast_models/valid_*.csv  (python -m experiments.forecast_model_search --stage base / tabpfn)
  outputs/final_system/test_predictions.csv  (python -m experiments.final_system --stage test)
출력: outputs/final_analysis/

모델·임계값은 검증 구간에서만 정한다. 테스트 라벨로 다시 고르지 않는다.
"""
import argparse
import warnings

import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.metrics import average_precision_score, f1_score, precision_score, recall_score

from .data import load
from .eval_conditions import by_condition, condition_frame
from .features import build
from .forecast_adapters import make_forecaster
from .models import operating
from .peak import TAU

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "final_analysis"
MODELS = ["regime_ens", "regime_tabpfn_all"]
START, TEST0, END = "2021-01-08", "2021-08-16", "2021-09-15"

GROUPS = {
    "직전 가동일 같은 시각 값": ["op_power_lag", "op_peak_lag", "op_prod_lag", "days_since_op"],
    "전력 시차(전일·전주)": ["power_lag24", "power_lag48", "power_lag168", "peak_lag24", "peak_lag48", "peak_lag168",
                        "power_prevday_mean", "power_prevday_max", "power_prevweek_mean", "intrahour_ramp_lag24"],
    "생산량 계획": ["prod", "prod_day", "prod_share", "prod_prev_h", "prod_next_h", "log_prod", "prod_roll3", "night_shift"],
    "가동 스케줄(시작·종료)": ["first_active_h", "last_active_h", "before_start", "after_end"],
    "시각·달력": ["hour", "dow", "month", "hour_sin", "hour_cos", "is_weekend", "is_holiday", "is_offday", "is_monday_am"],
    "기상": ["temp", "humid", "wind", "rain", "temp_day_max", "cdd", "hdd"],
    "요금·인건비 구간": ["tariff", "labor_rate"],
}
HOURLY_PLAN = ["prod", "prod_share", "prod_prev_h", "prod_next_h", "log_prod", "prod_roll3", "night_shift",
               "first_active_h", "last_active_h", "before_start", "after_end", "op_prod_lag"]


def _valid(m):
    return pd.read_csv(ROOT / f"outputs/forecast_models/valid_{m}.csv", parse_dates=["timestamp"])


def _test():
    return pd.read_csv(ROOT / "outputs/final_system/test_predictions.csv", parse_dates=["timestamp"])


def _daily_max(p):
    p = p[p.target == "peak15"].assign(d=lambda x: x.timestamp.dt.normalize())
    return p.groupby("d").agg(pred=("point", "max"), act=("actual", "max"))


def peak_day_f1():
    """일 단위 목표수요(TAU) 초과일 판정. 임계값은 검증 두 fold(45일)에서 F1 최대."""
    test = _test()
    rows = []
    for m in MODELS:
        v, t = _daily_max(_valid(m)), _daily_max(test[test.model == m])
        yv, yt = (v.act >= TAU).astype(int), (t.act >= TAU).astype(int)
        grid = np.arange(120, 215, 1.0)
        thr = grid[int(np.argmax([f1_score(yv, v.pred >= g, zero_division=0) for g in grid]))]
        for name, th in (("검증 임계값", thr), ("TAU 그대로", float(TAU))):
            yh = (t.pred >= th).astype(int)
            rows.append(dict(model=m, rule=name, threshold=th, test_days=len(t), events=int(yt.sum()),
                             alarms=int(yh.sum()), f1=f1_score(yt, yh, zero_division=0),
                             recall=recall_score(yt, yh, zero_division=0),
                             precision=precision_score(yt, yh, zero_division=0),
                             pr_auc=average_precision_score(yt, t.pred)))
    return pd.DataFrame(rows)


def conditions(df, X):
    """조건별 전력 MAE: 검증 fold2(7/16~8/15)와 테스트, 두 모델을 같은 시간에 대해."""
    out = []
    test = _test()
    for split, frame in (("valid_fold2", None), ("test", test)):
        preds, y = {}, None
        for m in MODELS:
            p = _valid(m) if frame is None else frame[frame.model == m]
            p = p[p.target == "power"]
            if frame is None:
                p = p[p.fold == 2]
            p = p.set_index("timestamp")
            preds[m], y = p["point"], p["actual"]
        f = condition_frame(df, X, df.index.isin(y.index))
        r = by_condition(f, y.loc[f.index], {m: s.loc[f.index] for m, s in preds.items()})
        out.append(r.assign(split=split))
    return pd.concat(out, ignore_index=True)


def _fit_test(df, X, name, target):
    """동결 설정과 같은 구간(1/8~8/15, 정전·시차 결측 제외)으로 학습해 테스트 시간을 예측."""
    t = df.index
    ok = (~df["outage"]).to_numpy() & X["power_lag168"].notna().to_numpy()
    tr = ok & (t >= START) & (t < TEST0)
    te = ok & (t >= TEST0) & (t < END)
    m = make_forecaster(name).fit(X[tr], df[target][tr], df["hour"][tr], df["is_copy"].to_numpy()[tr])
    return m, tr, te


def importance(df, X, repeats=3, seed=0):
    """가동일 15분 최대수요의 피처 그룹 순열 중요도(테스트 가동일 시간). 그룹 단위로 같은 행 순열을 적용."""
    rows = []
    for name in MODELS:
        m, tr, te = _fit_test(df, X, name, "peak15")
        Xt = X[te]
        on = operating(Xt)
        Xo, y = Xt[on], df["peak15"][te][on].to_numpy()
        base = np.abs(m.on_model.predict(Xo, None) - y).mean()
        rng = np.random.default_rng(seed)
        for g, cols in GROUPS.items():
            inc = []
            for _ in range(repeats):
                Xp = Xo.copy()
                idx = rng.permutation(len(Xp))
                Xp[cols] = Xo[cols].to_numpy()[idx]
                inc.append(np.abs(m.on_model.predict(Xp, None) - y).mean() - base)
            rows.append(dict(model=name, group=g, base_mae=base, mae_increase=float(np.mean(inc)),
                             sd=float(np.std(inc))))
            print(name, g, round(float(np.mean(inc)), 3), flush=True)
    r = pd.DataFrame(rows)
    r["rank"] = r.groupby("model")["mae_increase"].rank(ascending=False).astype(int)
    return r.sort_values(["model", "rank"])


def plan_sensitivity(df, name="regime_tabpfn_all", sigmas=(0.2, 0.5), seed=0):
    """생산계획을 전날 확정값으로 가정한 것이 얼마나 중요한가(테스트 전력 MAE).
    - 시간별 계획: 기준
    - 일 단위 계획만: 시간별 계획 피처 없이 일 총량·가동일 여부만으로 학습·예측
    - 계획 오차: 학습은 실제 계획, 테스트 시점 시간별 계획에 로그정규 잡음(가동 시간은 유지)"""
    X = build(df)
    rows = []
    m, tr, te = _fit_test(df, X, name, "power")
    y = df["power"][te].to_numpy()
    rows.append(dict(variant="시간별 계획(기준)", mae=np.abs(m.predict(X[te], df["hour"][te]) - y).mean()))
    for s in sigmas:
        d = df.copy()
        test = d.index >= TEST0
        noise = np.exp(np.random.default_rng(seed).normal(0, s, test.sum()))
        d.loc[test, "prod"] = d.loc[test, "prod"] * noise
        Xn = build(d)
        rows.append(dict(variant=f"계획 잡음 σ={s}", mae=np.abs(m.predict(Xn[te], df["hour"][te]) - y).mean()))
        print(rows[-1], flush=True)
    Xd = X.drop(columns=HOURLY_PLAN)
    md, _, _ = _fit_test(df, Xd, name, "power")
    rows.append(dict(variant="일 단위 계획만", mae=np.abs(md.predict(Xd[te], df["hour"][te]) - y).mean()))
    return pd.DataFrame(rows).assign(model=name)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tables", "importance", "plan"], required=True)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    df = load()
    X = build(df)
    if a.stage == "tables":
        f1 = peak_day_f1()
        f1.to_csv(OUT / "peak_day_f1.csv", index=False, encoding="utf-8-sig")
        c = conditions(df, X)
        c.to_csv(OUT / "conditions.csv", index=False, encoding="utf-8-sig")
        print(f1.round(3).to_string(), "\n", c.round(2).to_string())
    elif a.stage == "importance":
        r = importance(df, X)
        r.to_csv(OUT / "group_importance_peak15.csv", index=False, encoding="utf-8-sig")
        print(r.round(3).to_string())
    else:
        r = plan_sensitivity(df)
        r.to_csv(OUT / "plan_sensitivity.csv", index=False, encoding="utf-8-sig")
        print(r.round(3).to_string())


if __name__ == "__main__":
    main()
