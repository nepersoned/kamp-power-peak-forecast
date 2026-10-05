"""최종 모델(Regime-TabPFN) 기준 보고서 보조 분석 (2.4·3.2·3.3절, 한계 절).

    python -m src.final_analysis --stage tables       # F1·조건별 오차 (저장된 예측 파일만 사용, 수 초)
    python -m src.final_analysis --stage importance   # 피처 그룹 순열 중요도 (TabPFN·앙상블, 수십 분)
    python -m src.final_analysis --stage plan         # 생산계획 가정 민감도 (TabPFN, 수십 분)
    python -m src.final_analysis --stage regime_info  # 생산계획 정보 수준별 성능(계획/지난주 가동/달력만, TabPFN)
    python -m src.final_analysis --stage peak         # 권고 생산계획의 피크·전력량 변화(kW·kWh, 저장된 계획만 사용)

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


def peak_effect():
    """최종 시스템이 권고한 생산계획(test_production_plans.csv)의 피크·전력량 변화.
    실측 + 대리모형 변화분(decision_eval.true_cost와 같은 방식)으로 시간별 값을 다시 계산한다."""
    import json
    from . import tariff as T
    from .milp import _S_numeric
    fs = ROOT / "outputs/final_system"
    coefs = {k: {kk: (np.array(vv) if isinstance(vv, list) else vv) for kk, vv in v.items()}
             for k, v in json.loads((fs / "surrogate_coefficients.json").read_text()).items()}
    plans = pd.read_csv(fs / "test_production_plans.csv")
    df = load()
    rows = []
    for (day, system, seed), g in plans.groupby(["day", "system", "seed"]):
        g = g.sort_values("hour")
        idx = pd.date_range(day, periods=24, freq="h")
        pk, pw = df.loc[idx, "peak15"].to_numpy(float), df.loc[idx, "power"].to_numpy(float)
        a, b = g["original"].to_numpy(float), g["recommended"].to_numpy(float)
        pk_new = np.clip(pk + _S_numeric(b, coefs["peak15"]) - _S_numeric(a, coefs["peak15"]), 0, None)
        pw_new = np.clip(pw + _S_numeric(b, coefs["power"]) - _S_numeric(a, coefs["power"]), 0, None)
        bands = T.hourly_bands(idx)
        peak_h = (bands["energy_band"] == 2).to_numpy()          # 최대부하 요금 시간
        bill_h = (bands["demand_band"] > T.OFF).to_numpy()         # 기본요금 산정 시간(경부하 제외)
        rows.append(dict(day=day, system=system, seed=seed,
                         bill_peak_before=pk[bill_h].max(initial=0.0), bill_peak_after=pk_new[bill_h].max(initial=0.0),
                         peakband_peak_before=pk[peak_h].max(initial=0.0), peakband_peak_after=pk_new[peak_h].max(initial=0.0),
                         peakband_kwh_before=pw[peak_h].sum(), peakband_kwh_after=pw_new[peak_h].sum(),
                         hours_over_tau_before=int((pk[bill_h] >= TAU).sum()), hours_over_tau_after=int((pk_new[bill_h] >= TAU).sum()),
                         day_kwh_before=pw.sum(), day_kwh_after=pw_new.sum()))
    r = pd.DataFrame(rows)
    day_mean = r.groupby(["system", "day"]).mean(numeric_only=True).reset_index()   # 시드 평균 후 날짜 평균
    summ = day_mean.groupby("system").mean(numeric_only=True).drop(columns="seed")
    summ["days"] = day_mean.groupby("system").size()
    return r, summ


def regime_info(name="regime_tabpfn_all"):
    """'생산계획을 전날 안다'는 가정 점검. 레짐 판별과 계획 피처의 정보 수준을 낮추며 같은 모델을 재학습한다.
    - 전날 확정 계획(기준): 시간별 생산계획 피처 + 계획으로 가동일 판별
    - 지난주 같은 요일 가동 여부: 계획 피처 없음, 7일 전 같은 요일의 가동 여부로 가동일 판별
    - 달력만: 계획 피처 없음, 평일(공휴일 제외)이면 가동일
    평가: 검증 fold2(7/16~8/15, 휴가 포함)와 테스트. 모델 선정에는 쓰지 않는다."""
    df = load()
    X_plan = build(df)
    X_none = build(df, use_plan=False)
    day = df.index.normalize()
    on_true = (df["prod"].fillna(0).groupby(day).transform("sum") > 0).astype(float)
    on_lastweek = on_true.groupby(day).first().shift(7).reindex(day).to_numpy()
    calendar = (X_none["is_offday"] == 0).astype(float).to_numpy()
    variants = {"전날 확정 계획(기준)": X_plan,
                "지난주 같은 요일 가동 여부": X_none.assign(plan_on_day=np.nan_to_num(on_lastweek, nan=1.0)),
                "달력만(평일=가동)": X_none.assign(plan_on_day=calendar)}
    t = df.index
    ok = (~df["outage"]).to_numpy() & X_plan["power_lag168"].notna().to_numpy()
    rows = []
    for split, (fit_end, ev0, ev1) in {"검증 fold2": ("2021-07-16", "2021-07-16", "2021-08-16"),
                                       "테스트": (TEST0, TEST0, END)}.items():
        tr = ok & (t >= START) & (t < fit_end)
        ev = ok & (t >= ev0) & (t < ev1)
        for vname, X in variants.items():
            wrong = float((X["plan_on_day"].fillna(1).to_numpy()[ev] != on_true.to_numpy()[ev]).mean())
            r = dict(split=split, variant=vname, regime_error_rate=wrong)
            for tgt in ("power", "peak15"):
                m = make_forecaster(name).fit(X[tr], df[tgt][tr], df["hour"][tr], df["is_copy"].to_numpy()[tr])
                p = m.predict(X[ev], df["hour"][ev])
                y = df[tgt][ev].to_numpy()
                r[f"{tgt}_mae"] = float(np.abs(p - y).mean())
                if tgt == "peak15":
                    dm = pd.DataFrame({"p": p, "y": y}, index=t[ev]).groupby(t[ev].normalize()).max()
                    r["daily_max_mae"] = float((dm.p - dm.y).abs().mean())
            rows.append(r)
            print(r, flush=True)
    return pd.DataFrame(rows)


def fnfp():
    """목표수요(TAU) 초과 판정의 FN·FP가 어떤 생산조건에 몰리는가 (최종 모델, 임계값은 검증에서 결정).
    일 단위: 예측 일최대 >= 임계값이면 경보. 시간 단위: 예측 15분최대 >= 임계값이면 경보.
    비교 기준선: '가동 평일이면 경보' 규칙."""
    df = load()
    day = df.index.normalize()
    prod_day = df["prod"].fillna(0).groupby(day).sum()
    hours_day = (df["prod"].fillna(0) > 0).groupby(day).sum()
    op = prod_day > 0
    after_off = op & ~op.shift(1, fill_value=True)          # 휴무 다음 첫 가동일
    out = {}
    for level in ("day", "hour"):
        rows = []
        for m in MODELS:
            v = _valid(m); t = _test(); t = t[t.model == m]
            if level == "day":
                v, t = _daily_max(v), _daily_max(t)
            else:
                v = v[v.target == "peak15"].set_index("timestamp")[["point", "actual"]].rename(columns={"point": "pred", "actual": "act"})
                t = t[t.target == "peak15"].set_index("timestamp")[["point", "actual"]].rename(columns={"point": "pred", "actual": "act"})
            grid = np.arange(120, 215, 1.0)
            thr = grid[int(np.argmax([f1_score(v.act >= TAU, v.pred >= g, zero_division=0) for g in grid]))]
            t = t.assign(event=(t.act >= TAU), alarm=(t.pred >= thr))
            idx = t.index.normalize() if level == "hour" else t.index
            t["weekday"] = pd.Index(idx).dayofweek.map(lambda d: "월" if d == 0 else ("토·일" if d >= 5 else "화~금"))
            t["after_off"] = pd.Index(idx).map(after_off).fillna(False).map({True: "휴무 다음 가동일", False: "그 외"})
            t["long_day"] = pd.Index(idx).map(hours_day).map(lambda h: "생산 13시간 이상" if h >= 13 else "생산 12시간 이하")
            if level == "hour":
                h = t.index.hour
                t["hour_band"] = np.select([(h >= 8) & (h <= 11), (h >= 13) & (h <= 16)], ["08~11시", "13~16시"], "그 외 시간")
            t["outcome"] = np.select([t.event & t.alarm, t.event & ~t.alarm, ~t.event & t.alarm], ["TP", "FN", "FP"], "TN")
            conds = ["weekday", "after_off", "long_day"] + (["hour_band"] if level == "hour" else [])
            for c in conds:
                for lv, g in t.groupby(c):
                    vc = g.outcome.value_counts()
                    rows.append(dict(model=m, threshold=thr, condition=c, level=lv, n=len(g),
                                     TP=int(vc.get("TP", 0)), FN=int(vc.get("FN", 0)), FP=int(vc.get("FP", 0)), TN=int(vc.get("TN", 0))))
            vc = t.outcome.value_counts()
            rows.append(dict(model=m, threshold=thr, condition="전체", level="전체", n=len(t),
                             TP=int(vc.get("TP", 0)), FN=int(vc.get("FN", 0)), FP=int(vc.get("FP", 0)), TN=int(vc.get("TN", 0))))
        # 기준선: 가동 평일이면 경보
        tt = _test(); tt = tt[(tt.model == MODELS[1]) & (tt.target == "peak15")].set_index("timestamp")
        if level == "day":
            dd = tt.groupby(tt.index.normalize()).agg(act=("actual", "max"))
            rule = pd.Index(dd.index).map(lambda d: d.dayofweek < 5 and op.get(d, False))
            ev = dd.act >= TAU
        else:
            rule = pd.Index(tt.index).map(lambda ts: ts.dayofweek < 5 and op.get(ts.normalize(), False) and 8 <= ts.hour <= 16)
            ev = tt.actual >= TAU
        rule = np.asarray(rule, bool); ev = np.asarray(ev, bool)
        rows.append(dict(model="규칙(가동 평일" + ("" if level == "day" else " 08~16시") + ")", threshold=np.nan, condition="전체", level="전체",
                         n=len(ev), TP=int((rule & ev).sum()), FN=int((~rule & ev).sum()), FP=int((rule & ~ev).sum()), TN=int((~rule & ~ev).sum())))
        r = pd.DataFrame(rows)
        r["precision"] = r.TP / (r.TP + r.FP).replace(0, np.nan)
        r["recall"] = r.TP / (r.TP + r.FN).replace(0, np.nan)
        r["f1"] = 2 * r.TP / (2 * r.TP + r.FP + r.FN).replace(0, np.nan)
        out[level] = r
    return out


def interaction(df, X, pairs=None, repeats=3, seed=0):
    """최종 모델(Regime-TabPFN) 그룹 상호작용 = 두 그룹을 함께 섞은 오차 증가 − 각각 섞은 증가의 합.
    양수면 두 그룹이 함께 쓰일 때 더 많은 정보를 준다(시너지). 테스트 가동일, 15분 최대수요."""
    pairs = pairs or [("생산량 계획", "시각·달력"), ("전력 시차(전일·전주)", "시각·달력"), ("생산량 계획", "전력 시차(전일·전주)"),
                      ("가동 스케줄(시작·종료)", "시각·달력"), ("생산량 계획", "가동 스케줄(시작·종료)"), ("기상", "시각·달력")]
    m, tr, te = _fit_test(df, X, "regime_tabpfn_all", "peak15")
    Xt = X[te]; on = operating(Xt); Xo, y = Xt[on], df["peak15"][te][on].to_numpy()
    f = lambda Z: np.abs(m.on_model.predict(Z, None) - y).mean()
    base = f(Xo)
    rng = np.random.default_rng(seed)
    single, rows = {}, []
    def perm(groups):
        out = []
        for _ in range(repeats):
            Z = Xo.copy(); idx = rng.permutation(len(Z))
            cols = [c for g in groups for c in GROUPS[g]]
            Z[cols] = Xo[cols].to_numpy()[idx]
            out.append(f(Z) - base)
        return float(np.mean(out))
    for a, b in pairs:
        for g in (a, b):
            if g not in single:
                single[g] = perm([g])
        joint = perm([a, b])
        rows.append(dict(group_a=a, group_b=b, inc_a=single[a], inc_b=single[b], inc_joint=joint,
                         interaction=joint - single[a] - single[b]))
        print(rows[-1], flush=True)
    return pd.DataFrame(rows).sort_values("interaction", ascending=False)


def main():
    warnings.filterwarnings("ignore")
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["tables", "importance", "plan", "peak", "regime_info", "fnfp", "interaction"], required=True)
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
    elif a.stage == "interaction":
        r = interaction(df, X)
        r.to_csv(OUT / "interaction_peak15.csv", index=False, encoding="utf-8-sig")
        print(r.round(3).to_string())
    elif a.stage == "fnfp":
        for level, r in fnfp().items():
            r.to_csv(OUT / f"fnfp_{level}.csv", index=False, encoding="utf-8-sig")
            print(level); print(r.round(2).to_string())
    elif a.stage == "regime_info":
        r = regime_info()
        r.to_csv(OUT / "regime_info.csv", index=False, encoding="utf-8-sig")
        print(r.round(3).to_string())
    elif a.stage == "peak":
        r, summ = peak_effect()
        r.to_csv(OUT / "peak_effect_daily.csv", index=False, encoding="utf-8-sig")
        summ.to_csv(OUT / "peak_effect_summary.csv", encoding="utf-8-sig")
        print(summ.round(2).T.to_string())
    else:
        r = plan_sensitivity(df)
        r.to_csv(OUT / "plan_sensitivity.csv", index=False, encoding="utf-8-sig")
        print(r.round(3).to_string())


if __name__ == "__main__":
    main()
