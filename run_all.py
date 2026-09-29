"""전처리 -> 학습 -> 검증/테스트 -> 오류분석 -> 피크 저감 시뮬레이션 -> 결과 파일 생성 (한 번에 실행).

    python run_all.py

구간 (원본 데이터만 평가. 1~6월은 증강 복제일이 대부분이라 평가에서 제외)
  TRAIN : 2021-01-08 ~ 07-15  (lag168 확보 이후. 복제일은 모델별로 제외 또는 저가중)
  VALID : 2021-07-16 ~ 08-15  (모델·임계값·보정량 선택)
  TEST  : 2021-08-16 ~ 09-14  (최종 1회 평가, TRAIN+VALID로 재학습)
"""
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold

from src.data import Q, load
from src.evaluate import daily_peak_metrics, error_by, metrics_table
from src.features import build
from src.explain import dice_peak_actions, shap_report, waterfall
from src.models import LGBM, RegimeModel, candidates, final_model, operating, quantile_upper
from src.peak import TAU, PeakClassifier, event_metrics, pick_threshold, simulate_shaving

warnings.filterwarnings("ignore")
OUT = Path(__file__).resolve().parent / "outputs"
OUT.mkdir(exist_ok=True)
START, VAL0, TEST0, END = "2021-01-08", "2021-07-16", "2021-08-16", "2021-09-15"
TARGETS = ["power", "peak15"]


def masks(df, X):
    ok = (~df["outage"]).to_numpy() & X["power_lag168"].notna().to_numpy()
    t = df.index
    train = ok & (t >= START) & (t < VAL0)
    val = ok & (t >= VAL0) & (t < TEST0)
    test = ok & (t >= TEST0) & (t < END)
    return train, val, test


COPY = None


def fit_predict(make, X, y, hour, tr, te, copy=True):
    c = COPY[tr] if copy else None
    m = make().fit(X[tr], y[tr], hour[tr], c)
    return m.predict(X[te], hour[te]), m


def compare(df, X, tr, te, tag):
    tables = {}
    for tgt in TARGETS:
        y = df[tgt]
        preds = {n: fit_predict(mk, X, y, df["hour"], tr, te)[0] for n, mk in candidates().items()}
        t = metrics_table(y[te].to_numpy(), preds)
        t["DailyPeak_MAE"] = [round(daily_peak_metrics(y[te], p, df.index[te])["MAE"], 2) for p in preds.values()]
        on = operating(X[te])
        t["MAE_operating"] = metrics_table(y[te].to_numpy(), preds, on)["MAE"]
        t["MAE_shutdown"] = metrics_table(y[te].to_numpy(), preds, ~on)["MAE"]
        tables[tgt] = t
        t.to_csv(OUT / f"model_compare_{tag}_{tgt}.csv", encoding="utf-8-sig")
    return tables


def ablation(df, val):
    """검증 구간 기준 설계 선택 근거: 생산계획 사용, 복제일 제거, 레짐 분리."""
    rows = []
    t = df.index
    base_tr = (~df["outage"]).to_numpy() & (t >= START) & (t < VAL0)
    for use_plan in (True, False):
        X = build(df, use_plan)
        ok = X["power_lag168"].notna().to_numpy()
        for drop_copy in (True, False):
            tr = base_tr & ok
            makers = {"lgbm_single": LGBM}
            if use_plan:
                makers["regime_lgbm"] = RegimeModel
            for name, mk in makers.items():
                p, _ = fit_predict(mk, X, df["power"], df["hour"], tr, val, copy=drop_copy)
                e = np.abs(df["power"].to_numpy()[val] - p)
                rows.append(dict(model=name, use_plan=use_plan, drop_copies=drop_copy,
                                 train_rows=int(tr.sum()), val_MAE=round(e.mean(), 2)))
    X = build(df)
    ok = X["power_lag168"].notna().to_numpy()
    p, _ = fit_predict(final_model, X, df["power"], df["hour"], base_tr & ok, val)
    rows.append(dict(model="regime_ens(final)", use_plan=True, drop_copies="weighted",
                     train_rows=int((base_tr & ok).sum()), val_MAE=round(np.abs(df["power"].to_numpy()[val] - p).mean(), 2)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "ablation_valid.csv", index=False, encoding="utf-8-sig")
    return out


def leakage_demo(df, X):
    """학습 기간(복제일 포함)에서 교차검증 방식별 MAE.
    무작위 K-fold는 복제일의 원본이 학습 fold에 들어가 오차를 과소평가한다."""
    ok = (~df["outage"]).to_numpy() & X["power_lag168"].notna().to_numpy() & (df.index < VAL0)
    Xo, yo, ho = X[ok], df["power"][ok], df["hour"][ok]
    day = df.index[ok].normalize()
    sig = df[Q].groupby(df.index.normalize()).apply(lambda x: hash(tuple(x.to_numpy().ravel())))
    schemes = {"random_hourly_kfold": (KFold(5, shuffle=True, random_state=0), None),
               "day_grouped_kfold": (GroupKFold(5), day.factorize()[0]),
               "profile_grouped_kfold": (GroupKFold(5), day.map(sig).factorize()[0])}
    rows = []
    for name, (cv, groups) in schemes.items():
        p = np.zeros(len(yo))
        for a, b in cv.split(Xo, yo, groups):
            p[b] = RegimeModel().fit(Xo.iloc[a], yo.iloc[a], ho.iloc[a], None).predict(Xo.iloc[b], ho.iloc[b])
        e = np.abs(yo.to_numpy() - p)
        cp = df["is_copy"].to_numpy()[ok]
        rows.append(dict(scheme=name, MAE=round(e.mean(), 2), MAE_copy_days=round(e[cp].mean(), 2),
                         MAE_original_days=round(e[~cp].mean(), 2)))
    out = pd.DataFrame(rows)
    out.to_csv(OUT / "leakage_cv_demo.csv", index=False, encoding="utf-8-sig")
    return out


def conformal_adjust(y_val, upper_val, on_val, alpha=0.9):
    """분할 컨포멀 보정(CQR): 검증 구간 가동시간의 초과 잔차 분위수만큼 상한을 올림."""
    s = (y_val - upper_val)[on_val]
    n = len(s)
    return float(np.quantile(s, min(1.0, np.ceil((n + 1) * alpha) / n)))


def daily_frame(y_event, score, index):
    return pd.DataFrame({"y": y_event, "s": score}, index=index).resample("D").max().dropna()


def peak_risk(df, X, tr, val, te, trval):
    """피크 위험 판정.
    1차(일 단위): 내일 목표수요(TAU) 초과일 여부 — 현장 의사결정 단위
    2차(시간 단위): 위험일 내 관리 대상 시간 — 참고 지표
    점수 후보: 전일 실측(naive) / 제안모델 예측 / P90 상한 / 전용 분류기(보정 확률)
    임계값은 모두 VALID에서 결정, TEST는 1회 평가."""
    h, y = df["hour"], df["peak15"]
    ev = (y >= TAU).astype(int)
    # VALID 점수: TRAIN만으로 학습
    reg_v, _ = fit_predict(final_model, X, y, h, tr, val)
    up_v, _ = fit_predict(quantile_upper, X, y, h, tr, val)
    nc = ~COPY
    clf_v = PeakClassifier().fit(X[tr & nc], ev[tr & nc]).calibrate(X[val], ev[val])
    # TEST 점수: TRAIN+VALID로 재학습 (분류기 보정은 VALID 기준 유지)
    reg_t, reg_model = fit_predict(final_model, X, y, h, trval, te)
    up_t, _ = fit_predict(quantile_upper, X, y, h, trval, te)
    clf_t = PeakClassifier().fit(X[trval & nc], ev[trval & nc]).calibrate(X[val], ev[val])
    adj = conformal_adjust(y[val].to_numpy(), up_v, operating(X[val]))
    up_t_cqr = up_t.copy()
    up_t_cqr[operating(X[te])] += adj

    def rule(m):  # 단순 규칙: 평일 가동일의 주간(08~15시)은 모두 위험
        Xm = X[m]
        return ((Xm["is_offday"] == 0) & operating(Xm) & Xm["hour"].between(8, 15)).astype(float).to_numpy()

    scores = {
        "rule_weekday_daytime": (rule(val), rule(te), False),
        "naive_24h": (X.loc[val, "peak_lag24"].to_numpy(), X.loc[te, "peak_lag24"].to_numpy(), False),
        "regime_ens": (reg_v, reg_t, False),
        "regime_P90": (up_v, up_t, False),
        "classifier": (clf_v.predict_proba(X[val]), clf_t.predict_proba(X[te]), True),
    }
    day_rows, hour_rows, day_thr = {}, {}, {}
    for name, (sv, st, is_prob) in scores.items():
        dv = daily_frame(ev[val].to_numpy(), sv, df.index[val])
        dt = daily_frame(ev[te].to_numpy(), st, df.index[te])
        thr = pick_threshold(dv["y"].to_numpy(), dv["s"].to_numpy(), beta=1.0)
        day_thr[name] = thr
        day_rows[name] = event_metrics(dt["y"].to_numpy(), dt["s"].to_numpy(), thr, is_prob)
        hthr = pick_threshold(ev[val].to_numpy(), sv, beta=1.0)
        hour_rows[name] = event_metrics(ev[te].to_numpy(), st, hthr, is_prob)
    day_tab, hour_tab = pd.DataFrame(day_rows).T, pd.DataFrame(hour_rows).T
    day_tab.to_csv(OUT / "peak_risk_daily_test.csv", encoding="utf-8-sig")
    hour_tab.to_csv(OUT / "peak_risk_hourly_test.csv", encoding="utf-8-sig")

    # 판정기준(TAU) 민감도 — 제안 점수(regime_ens 일최대 예측)
    sens = []
    for tau in (180, 190, 200):
        e = (y >= tau).astype(int)
        dv = daily_frame(e[val].to_numpy(), reg_v, df.index[val])
        dt = daily_frame(e[te].to_numpy(), reg_t, df.index[te])
        thr = pick_threshold(dv["y"].to_numpy(), dv["s"].to_numpy(), beta=1.0)
        sens.append({"TAU": tau, **event_metrics(dt["y"].to_numpy(), dt["s"].to_numpy(), thr)})
    pd.DataFrame(sens).to_csv(OUT / "peak_risk_tau_sensitivity.csv", index=False, encoding="utf-8-sig")

    return dict(day_tab=day_tab, hour_tab=hour_tab, day_thr=day_thr["regime_ens"], reg_t=reg_t, up_t=up_t,
                up_t_cqr=up_t_cqr, cqr_adj=adj, reg_model=reg_model, sens=pd.DataFrame(sens))


def main():
    global COPY
    df = load()
    X = build(df)
    COPY = df["is_copy"].to_numpy()
    tr, val, te = masks(df, X)
    trval = tr | val
    hour = df["hour"]
    summary = {"rows": len(df), "copied_days": int(df["is_copy"].resample("D").first().sum()),
               "corrupt_hour_rows": int(df["hour_corrupt"].sum()), "outage_hours": int(df["outage"].sum()),
               "train_rows_original": int((tr & ~COPY).sum()), "train_rows_with_copies": int(tr.sum()), "valid_rows": int(val.sum()), "test_rows": int(te.sum()), "TAU": TAU}

    # 1) 검증 구간 비교(모델 선택) + 설계 근거 + 검증방식 누수 시연
    val_tab = compare(df, X, tr, val, "valid")
    abl = ablation(df, val)
    leak = leakage_demo(df, X)

    # 2) 테스트: TRAIN+VALID 재학습 후 1회 평가
    test_tab = compare(df, X, trval, te, "test")
    power_pred, _ = fit_predict(final_model, X, df["power"], hour, trval, te)
    power_up, _ = fit_predict(quantile_upper, X, df["power"], hour, trval, te)

    # 3) 피크 위험(일/시간) + 불확실성 보정
    pr = peak_risk(df, X, tr, val, te, trval)
    yt = df["peak15"].to_numpy()[te]
    summary["P90_coverage_test_raw"] = round(float(np.mean(yt <= pr["up_t"])), 3)
    summary["P90_coverage_test_cqr"] = round(float(np.mean(yt <= pr["up_t_cqr"])), 3)
    summary["cqr_adjustment"] = round(pr["cqr_adj"], 2)
    summary["risk_day_threshold"] = round(pr["day_thr"], 1)

    # 4) 테스트 예측 결과 파일
    idx = df.index[te]
    pred = pd.DataFrame({
        "power_actual": df.loc[idx, "power"], "power_pred": power_pred.round(2), "power_pred_p90": power_up.round(2),
        "peak15_actual": df.loc[idx, "peak15"], "peak15_pred": pr["reg_t"].round(2),
        "peak15_upper_cqr": pr["up_t_cqr"].round(2),
        "regime": np.where(operating(X.loc[idx]), "operating", "shutdown"),
    }, index=idx)
    day_max = pred["peak15_pred"].groupby(idx.normalize()).transform("max")
    pred["risk_day_alert"] = (day_max >= pr["day_thr"]).astype(int)
    pred["risk_hour_rank"] = pred.groupby(idx.normalize())["peak15_pred"].rank(ascending=False, method="first").astype(int)
    pred.index.name = "timestamp"
    pred.to_csv(OUT / "test_predictions.csv", encoding="utf-8-sig")

    # 5) 오류 분석 (테스트, 제안 모델, 15분 최대수요)
    fr = pd.DataFrame({"err": pred["peak15_pred"] - pred["peak15_actual"], "hour": df.loc[idx, "hour"],
                       "dow": df.loc[idx, "dow"], "regime": pred["regime"],
                       "days_since_op": X.loc[idx, "days_since_op"], "prod": df.loc[idx, "prod"],
                       "temp": df.loc[idx, "temp"]}, index=idx)
    fr["abs_err"] = fr["err"].abs()
    fr["prod_bin"] = pd.cut(fr["prod"], [-1, 0, 500, 1500, 1e9], labels=["0", "1-500", "501-1500", ">1500"])
    fr["temp_bin"] = pd.cut(fr["temp"], [-50, 20, 25, 30, 50], labels=["<20", "20-25", "25-30", ">=30"])
    fr["restart"] = np.where(fr["days_since_op"] > 3, "after_long_stop", "normal")
    ea = {k: error_by(fr, k) for k in ["hour", "dow", "regime", "prod_bin", "temp_bin", "restart"]}
    with pd.ExcelWriter(OUT / "error_analysis_test.xlsx") as w:
        for k, t in ea.items():
            t.to_excel(w, sheet_name=k)
    # 일 단위 경보 FN/FP 사례
    d = pd.DataFrame({"actual_max": pred["peak15_actual"].resample("D").max(),
                      "pred_max": pred["peak15_pred"].resample("D").max(),
                      "alert": pred["risk_day_alert"].resample("D").max(),
                      "prod_day": df.loc[idx, "prod"].resample("D").sum(),
                      "temp_max": df.loc[idx, "temp"].resample("D").max(),
                      "dow": df.loc[idx, "dow"].resample("D").first()}).dropna()
    d["event"] = (d["actual_max"] >= TAU).astype(int)
    d["type"] = np.select([(d.event == 1) & (d.alert == 1), (d.event == 1) & (d.alert == 0), (d.event == 0) & (d.alert == 1)],
                          ["TP", "FN", "FP"], "TN")
    d.round(1).to_csv(OUT / "peak_risk_daily_cases_test.csv", encoding="utf-8-sig")

    # 6) 피크 저감 시뮬레이션 (모델 기반 what-if, 테스트 가동일 대상)
    shave = []
    for frac in (0.2, 0.3, 0.5):
        base, after, moves = simulate_shaving(df, X, build, pr["reg_model"], te, frac=frac, k=2)
        db, da = base.resample("D").max(), after.resample("D").max()
        on_days = db[db > 60].index
        shave.append(dict(frac=frac, days=len(on_days),
                          mean_daily_peak_before=round(db[on_days].mean(), 1),
                          mean_daily_peak_after=round(da[on_days].mean(), 1),
                          max_daily_peak_before=round(db.max(), 1), max_daily_peak_after=round(da.max(), 1),
                          hours_over_TAU_before=int((base >= TAU).sum()), hours_over_TAU_after=int((after >= TAU).sum()),
                          prod_moved_share=round(moves["moved"].sum() / moves["day_prod"].sum(), 3)))
        if frac == 0.3:
            moves.to_csv(OUT / "shaving_moves_frac30.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame({"before": base, "after": after}).to_csv(OUT / "shaving_hourly_frac30.csv", encoding="utf-8-sig")
    shave = pd.DataFrame(shave)
    shave.to_csv(OUT / "shaving_summary.csv", index=False, encoding="utf-8-sig")

    # 7) 모델 해석: SHAP(LightGBM 구성요소, 오프셋 포함 정확 분해) + DiCE(앙상블 전체) — 가동일 15분 최대수요
    on_te = te & operating(X)
    ens = pr["reg_model"].on_model
    booster = ens.lgbs[0].booster_
    sv, base, shap_imp, inter = shap_report(booster, X[on_te], OUT, offset_col="op_peak_lag")
    shap_imp.round(3).to_csv(OUT / "shap_importance_peak15.csv", encoding="utf-8-sig")
    sub = pred.loc[X.index[on_te]]
    lg_pred = sv.sum(1) + base
    act = sub["peak15_actual"].to_numpy()
    for kind, cond in [("FN", (act >= TAU) & (lg_pred < TAU)), ("FP", (act < TAU) & (lg_pred >= TAU))]:
        if cond.any():
            k = int(np.argmax(np.where(cond, np.abs(act - lg_pred), -1)))
            ts = sub.index[k]
            waterfall(sv[k], base, X.loc[ts], f"{kind} 사례 {ts:%m/%d %H}시 (실측 {act[k]:.0f})",
                      OUT / f"fig11_shap_waterfall_{kind}.png")
    cand = X[on_te][pr["reg_t"][operating(X[te])] >= TAU - 5]
    dice = dice_peak_actions(lambda Z: ens.predict(Z), cand,
                             df["prod"].groupby(df.index.normalize()).transform("sum"), OUT)

    from src.plots import make_all
    make_all(df, X, pred, fr, d, pr, OUT)

    summary["valid"] = {t: val_tab[t][["MAE", "RMSE"]].to_dict() for t in TARGETS}
    summary["test"] = {t: test_tab[t][["MAE", "RMSE"]].to_dict() for t in TARGETS}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    pd.set_option("display.width", 220)
    pd.set_option("display.max_columns", 20)
    for tgt in TARGETS:
        print(f"\n[VALID] {tgt}\n{val_tab[tgt]}\n[TEST] {tgt}\n{test_tab[tgt]}")
    print("\n[ABLATION valid, power]\n", abl.to_string(index=False))
    print("\n[LEAKAGE CV demo]\n", leak.to_string(index=False))
    print("\n[PEAK RISK daily test]\n", pr["day_tab"].to_string())
    print("\n[PEAK RISK hourly test]\n", pr["hour_tab"].to_string())
    print("\n[TAU sensitivity]\n", pr["sens"].to_string(index=False))
    print("\n[SHAVING]\n", shave.to_string(index=False))
    print("\n[SHAP top10]\n", shap_imp.head(10).round(2).to_string())
    print("\n[SHAP interactions top5]\n", inter.head(5).to_string(index=False))
    if len(dice):
        f = dice[dice["found"]]
        print(f"\n[DiCE] 대상 {len(dice)}시간, 해 발견 {len(f)}시간, 필요 생산감축 중앙값 {f['cut_prod_pct'].median():.1f}%")
    print("\n", json.dumps({k: v for k, v in summary.items() if k not in ("valid", "test")}, ensure_ascii=False))


if __name__ == "__main__":
    main()
