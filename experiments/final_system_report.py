"""Postprocess the single frozen evaluation; never fits or selects models."""
import json
import shutil
import zipfile
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/final_system"
MODELS=("regime_ens","regime_tabpfn_all")


def table(df):
    def fmt(v):
        return f"{v:,.3f}" if isinstance(v,(float,np.floating)) else str(v)
    return "| "+" | ".join(df.columns)+" |\n|"+"|".join(["---"]*len(df.columns))+"|\n"+"\n".join(
        "| "+" | ".join(fmt(v) for v in row)+" |" for row in df.itertuples(index=False,name=None))


def run(delivery=None):
    read=lambda name:pd.read_csv(OUT/name)
    config=json.loads((OUT/"final_system_config.json").read_text())
    audit=json.loads((OUT/"pretest_audit.json").read_text())
    ledger=json.loads((OUT/"test_execution.json").read_text())
    assert ledger["status"]=="completed" and ledger["test_evaluations"]==1
    point=read("test_forecast_comparison.csv").set_index("model")
    prob=read("test_probabilistic_metrics.csv")
    decision=read("test_decision_metrics.csv").set_index("system")
    bootf=read("test_bootstrap_forecast.csv");bootd=read("test_bootstrap_decision.csv")
    res=read("residual_pool.csv");summ=read("residual_summary.csv")
    cohort=read("data_eligibility.csv");events=read("test_event_summary.csv")
    daily=read("test_daily_decisions.csv")
    valid={MODELS[0]:10.240550169390211,MODELS[1]:8.375723143779238}
    improvement=100*(point.loc[MODELS[0],"primary_score"]-point.loc[MODELS[1],"primary_score"])/point.loc[MODELS[0],"primary_score"]
    base=decision.loc[MODELS[0]+" + empirical stochastic"]
    candidate=decision.loc[MODELS[1]+" + empirical stochastic"]
    delta=candidate.saving_won-base.saving_won
    ci=bootd[bootd.metric=="saving_won"].iloc[0]
    if improvement<=0: conclusion="TEST에서 forecasting 개선 자체가 재현되지 않음"
    elif ci.ci_low<=0<=ci.ci_high: conclusion="forecasting 개선, decision 차이는 CI로 확정할 수 없음"
    elif delta>0: conclusion="forecasting과 모델링된 decision 모두 개선"
    else: conclusion="forecasting 개선에도 모델링된 decision 악화"
    comparison=point.reset_index();comparison.insert(1,"valid_primary",comparison.model.map(valid))
    ds=decision.reset_index()[["system","n_days","saving_won","oracle_saving_won","regret_won","worst_day_saving","negative_saving_days","moved_share","energy_saving_won","ratchet_saving_won"]]
    rp=res.groupby(["model","target"]).agg(n_days=("forecast_date","nunique"),start=("forecast_date","min"),end=("forecast_date","max"),bias=("residual","mean"),mae=("residual",lambda x:x.abs().mean())).reset_index()
    excluded=cohort[~cohort.eligible].groupby(["stage","excluded_reason"]).size().rename("n_dates").reset_index()
    status=daily.groupby(["system","solver_status"]).size().rename("n_solves").reset_index()
    # Sanity comparison only: align actual dates before comparing old/new pools.
    oldpath=ROOT/"outputs/joint_uncertainty/residual_provenance.csv"
    if oldpath.exists():
        old=pd.read_csv(oldpath);rows=[]
        for target in ("power","peak15"):
            a=old[old.target==target].pivot(index="forecast_date",columns="hour",values="residual")
            b=res[(res.model==MODELS[1])&(res.target==target)].pivot(index="forecast_date",columns="hour",values="residual")
            a.index=pd.to_datetime(a.index);b.index=pd.to_datetime(b.index);idx=a.index.intersection(b.index)
            aa,bb=a.loc[idx],b.loc[idx]
            rows.append(dict(target=target,matched_days=len(idx),old_bias=aa.to_numpy().mean(),new_bias=bb.to_numpy().mean(),
                  old_mae=np.abs(aa.to_numpy()).mean(),new_mae=np.abs(bb.to_numpy()).mean(),
                  covariance_difference_frobenius=np.linalg.norm(bb.cov()-aa.cov())))
        pd.DataFrame(rows).to_csv(OUT/"residual_vs_phase1_sanity.csv",index=False)
    fig,ax=plt.subplots(figsize=(9,4))
    preds=read("test_predictions.csv");preds["error"]=(preds.actual-preds.point).abs()
    errors=preds.groupby(["model","date"]).error.mean().unstack("model")
    for name in MODELS:ax.plot(pd.to_datetime(errors.index),errors[name],label=name)
    ax.set(ylabel="Daily mean target absolute error (kW)",title="Frozen final TEST: daily point errors");ax.legend();fig.autofmt_xdate();fig.tight_layout()
    fig.savefig(OUT/"fig_test_daily_errors.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,4))
    savings=daily.groupby(["system","day"]).saving_won.mean().unstack("system")
    for name in savings:ax.plot(pd.to_datetime(savings.index),savings[name],label=name)
    ax.set(ylabel="Surrogate counterfactual saving (won/day)",title="Original operating-day decision evaluation");ax.legend(fontsize=7)
    fig.autofmt_xdate();fig.tight_layout();fig.savefig(OUT/"fig_test_decision_savings.png",dpi=160);plt.close(fig)
    fig,ax=plt.subplots(figsize=(9,9));ax.axis("off")
    labels=["D-1 available lags / production plan / weather assumption","Operating regime: plan_on_day (missing -> on)",
            "Regime-TabPFN: operating v2 / shutdown original-hour median","24h power + peak15 point forecasts",
            "Fixed pre-TEST, past-only original operating residual paths","30 empirical joint scenarios (seeds 0 / 1 / 2)",
            "Existing stochastic MILP: lambda=0 / delay<=2h / no extra starts","Production schedule recommendation",
            "Actual baseline + surrogate change -> tariff/ratchet saving & regret"]
    ys=np.linspace(.95,.05,len(labels))
    for i,(label,y) in enumerate(zip(labels,ys)):
        ax.text(.5,y,label,ha="center",va="center",fontsize=10,bbox=dict(boxstyle="round,pad=.5",fc="#e9f1fa",ec="#325d88"))
        if i<len(labels)-1:ax.annotate("",xy=(.5,ys[i+1]+.035),xytext=(.5,y-.035),arrowprops=dict(arrowstyle="->"))
    fig.savefig(OUT/"fig_final_pipeline.png",dpi=160,bbox_inches="tight");plt.close(fig)
    pipeline="""```mermaid
flowchart TD
    A[D-1 available information] --> B[Operating regime detection]
    B --> C[Regime-TabPFN]
    C --> D[24h power / peak15 forecasts]
    D --> E[Past-only empirical residual paths]
    E --> F[30 joint scenarios]
    F --> G[Stochastic MILP]
    G --> H[Production schedule recommendation]
    H --> I[Tariff / ratchet counterfactual cost evaluation]
```"""
    forecast_section=f"""### 2.6 오성민: 동결한 Regime-TabPFN 최종 평가

최종 모델은 TEST를 보기 전에 VALID 두 fold·두 target MAE 평균으로 선택했다. Primary10.241→8.376(18.21%), fold별10.405→9.401/10.077→7.351, 원본 weekly rolling5주 중4주 개선/1주 동률이었다. 단순 weighted ensemble의 추가 이득은 사실상 없었다. 일반 TabPFN은 휴무를 포함한 Fold2에서 실패했고 SARIMAX는 nonconverged baseline으로만 보존했다. Chronos/SimpleRNN/stacking은 구현했다고 주장하지 않는다.

가동일은 original-history TabPFN-v2, 휴무일은 기존 시간별 median을 사용한다. 46개 피처, estimator4/median/cache/CPU4 threads/seed42, checkpoint revision 및 hash를 고정했다. TEST 이후 모델/weight/feature/uncertainty/제약을 변경하지 않았다.

{table(comparison[["model","valid_primary","power_mae","peak15_mae","daily_max_mae","peak_hour_mae","primary_score"]])}

VALID와 TEST는 구간 난이도가 다르므로 절대 score끼리 직접 일반화 정도를 판단하지 않고 동일 TEST의 두 모델 차이를 비교한다. TEST primary 상대 개선={improvement:.2f}%. VALID 선택 모델은 결과와 관계없이 Regime-TabPFN으로 유지한다.

{table(bootf)}

날짜 단위 paired CI5000회(seed42). Negative difference는 선택 모델의 개선이다. TEST의 CI는 결과 불확실성 설명이며 모델 재선택에 사용하지 않는다. 이 CI는 날짜를 동일 비중으로 평균하므로 일부 outage 시간이 제외된 날 때문에 hourly aggregate difference와 약간 다를 수 있다. 그림 `outputs/final_system/fig_test_daily_errors.png`.

"""
    decision_section=f"""### 4.5 오성민: 최종 forecasting → joint uncertainty → decision 연결

Phase1 empirical full-path/K30/lambda0는 그대로 유지했다. 두 forecaster의 residual을 같은 pre-TEST 원본 가동일·past-only origins에서 각각 새로 생성했다. 기존 champion residual을 새 모델에 재사용하지 않았다. TRAIN plan_missing은 Phase2 규칙대로 fitting에 포함하지만 residual/decision은 유효한 생산계획의 완전24h 원본 가동일만 평가한다. TEST actual은 fitting/residual pool에 들어가지 않으며 TEST 동안 pool 갱신도 없다.

{table(rp)}

{table(prob)}

CRPS/coverage와 Energy/Variogram/event metrics는 decision-eligible 원본 가동일 peak15 중심이다(power CRPS 별도). 보고값은3개 사전 고정 seed의 날짜별 평균이다. Q10–Q90은 empirical scenario quantile이며 conformal guarantee가 아니다.

{table(ds)}

{table(bootd)}

판단: **{conclusion}**. B−A saving 차이={delta:,.1f}원/일. Ratchet event는 {int(events.ratchet_events.max())}일/{int(events.n_days.max())}일이다. Ratchet floor가 이미 높게 정해진 TEST에서는 TOU 절감과 rare ratchet 회피를 구분해야 한다. 7/19은 기존 VALID 사례이며 이 TEST 표와 섞거나 다시 튜닝하지 않았다.

{pipeline}

이 절감은 실제 공장 intervention 결과가 아니라 실측 baseline+linear surrogate change에 기반한다. Forecast MAE가 낮아도 plan/regret 순위가 같다는 보장은 없다. 추천안과3seed별 결과는 production plans/daily decisions CSV, 그림은 `fig_test_decision_savings.png`, `fig_final_pipeline.png`에 있다.

"""
    draft=ROOT/"docs/REPORT_DRAFT.md";text=draft.read_text(encoding="utf-8")
    assert "### 2.6 오성민:" not in text and "### 4.5 오성민:" not in text
    text=text.replace("레짐 전환 앙상블(최종)","레짐 전환 앙상블(기존 baseline)")
    text=text.replace("### 2.3 결과 (MAE, 동일 조건)","### 2.3 기존 baseline 실험 기록 (최종 frozen 비교는 2.6절)")
    text=text.replace("### 2.5 최종 모델 선택 이유","### 2.5 기존 baseline의 설계 근거 (최종 선택은 2.6절)")
    text=text.replace("1. 검증·테스트·롤링 평가 모두에서 가장 낮은 오차(롤링 10주 중 6주 1위)","1. 기존 비교에서 낮은 오류를 보였던 champion baseline이다. 최종 모델은 두 VALID fold만으로 새로 선택했으며 TEST를 선택에 사용하지 않는다.")
    text=text.replace("예측 일최대(최종 모델)","예측 일최대(기존 baseline)")
    text=text.replace("### 3.3 모델이 잘 작동하는 조건과 실패하는 조건 (전력 MAE, 최종 모델)","### 3.3 기존 baseline이 잘 작동하는 조건과 실패하는 조건 (전력 MAE)")
    text=text.replace("### 4.3 기대 효과 (실측 고정 평가, 모델 기반 추정)","### 4.3 기존 시스템의 기대 효과 기록 (신규 TEST 통합 결과는 4.5절)")
    text=text.replace("확률 MILP(최종)","확률 MILP(기존 실험)")
    old="- 생산계획으로 가동일/휴무일을 먼저 나누는 **레짐 전환 앙상블**로 하루 전 시간별 전력을 예측했다. 휴무 주간이 포함된 검증 구간에서 단일 LightGBM 대비 MAE 29.9 → 9.8, 평상 운전 테스트에서 전주 동일시각 대비 8.3 → 6.2이다. 주 단위 롤링 평가 10회 중 6회 1위."
    text=text.replace(old,f"- 최종 forecaster는 **Regime-TabPFN**이다. 동일 VALID 두 fold·두 target primary MAE가 기존 champion10.241→8.376(18.21%) 개선되어 TEST 전에 선정했다. 동결한 TEST primary 상대 개선은 {improvement:.2f}%이며 결과·CI와 한계는2.6절에 분리했다.")
    text=text.replace("무작위 교차검증은 오차를 약 32% 과소평가하므로 원본 구간 시간순 검증만 사용했다.","무작위 교차검증은 오차를 약 32% 과소평가했다. 시간순 검증과 원본-only sensitivity를 사용했고, Phase2 primary는 기존 harness와 동일하게 복제일을 포함했다.")
    text=text.replace("평가는 원본 구간만. 학습에서는 복제일을 제외하거나 낮은 가중치", "새 residual·decision 평가는 원본만, Phase2 forecasting primary는 legacy copy 포함(원본-only 보조). 학습에서는 복제일을 제외하거나 낮은 가중치")
    text=text.replace("| `【오성민】` 가이드북 기준 모델(SimpleRNN), SARIMAX, TabPFN(·Chronos-2) | 같은 분할·같은 표로 추가 |","| **Regime-TabPFN(VALID 선정 최종)** | 가동 TabPFN-v2/휴무 original median,46개 동일 피처. SARIMAX는 baseline, 일반 TabPFN은 제외. SimpleRNN/Chronos는 미구현 |")
    text=text.replace("## 제3장.",forecast_section+"## 제3장.",1)
    text=text.replace("## 제5장.",decision_section+"## 제5장.",1)
    text=text.replace("## 한계","최종 재현은 `docs/PHASE3_FINAL_SYSTEM.md`와 `experiments/final_system_config.json`을 따른다. 전체 pytest35 passed. TEST 실행 전 integration/config commit과 audit를 보존했다. 기존 run_all.py는 historical baseline 재현용이며 새 최종 시스템 entry point는 experiments.final_system이다.\n\n## 한계",1)
    text=text.replace("## 부록","추가 한계: TabPFN pretrained prior/초기 weight 다운로드/CPU 비용, 작은 original residual sample, copy history, rare ratchet scarcity, 과거 repo의 TEST 재확인 기록, 신규 frozen TEST1회, surrogate counterfactual 및 실제 intervention 부재.\n\n## 부록",1)
    draft.write_text(text,encoding="utf-8")
    progress=ROOT/"docs/PROGRESS.md"
    progress.write_text(progress.read_text(encoding="utf-8")+"\n\n## 13. Phase 3 frozen final evaluation\n\n"+forecast_section+decision_section,encoding="utf-8")
    full=f"""# Phase 3 최종 통합 결과

## A. Git / freeze state

Branch seongmin. Phase2 commit `{config['phase2_commit']}`. Pre-TEST integration/config commit `{ledger['pretest_commit']}`. Main `{config['main_sha']}` 유지. TEST후 source/config 변경 없음. 문서 결과 반영은 미커밋 상태이며 push/merge하지 않았다.

## B. Pre-test audit

{table(pd.DataFrame(list(audit['checks'].items()),columns=['check','passed']))}

Leakage violations={audit['leakage_violations']}. 전체 pytest35 passed. TEST 실행 ledger status={ledger['status']}, successful evaluation count=1, config hash=`{ledger['config_sha256']}`. TEST 기반 선택/온라인 residual 갱신 없음. D-1 history의 lag/floor 사용은 기존 문제 정의에 따라 허용된다.

## C–E. Final TEST forecast / generalization / CI

{forecast_section}

## F. New residual pool / eligibility

{table(rp)}

{table(excluded)}

두 모델의 residual 날짜는 동일하다. 각 target24h 완전 벡터, training_end<forecast_date, pool/fit cutoff Aug16 전. 원본14일 warmup,14calendar-day origin cadence. TEST actual을 fitting/pool에 사용하지 않았다. Origin hyperparameters는 VALID선정이므로 label-OOS와 hyperparameter selection independence는 다르다. Residual covariance/correlation/hour bias/tails/daily-max error 및 Phase1 matched-date sanity 비교를 CSV로 저장했다.

## G–I. Probabilistic / decision / forecast-to-decision

{decision_section}

{table(status)}

## J. Final system

VALID-selected Regime-TabPFN + its own fixed empirical full-path K30/lambda0 + unchanged stochastic MILP. TEST outcome으로 모델을 다시 선택하지 않았다. 선택 모델/uncertainty/policy는 frozen config 그대로다.

## K. Important results

- VALID primary18.21% 개선, 두 fold 개선/rolling4승1동률.
- TEST primary relative improvement {improvement:.2f}% (same hourly mask).
- Stochastic B−A saving {delta:,.1f}won/day, CI[{ci.ci_low:,.1f},{ci.ci_high:,.1f}].
- Ratchet exceedance events {int(events.ratchet_events.max())}/{int(events.n_days.max())} decision days; ordinary TOU and rare-event value separate.
- Conclusion: {conclusion}.

## L. Limits

Small original sample; copied dates; VALID-selected TabPFN pretrained prior; observed-weather forecast assumption; surrogate counterfactual; scarce rare ratchet events; single new frozen TEST evaluation; historical TEST metrics previously inspected elsewhere; no actual factory intervention. Date bootstrap assumes exchangeable days and does not establish external generalization. If CI spans zero, do not claim a confirmed improvement. July19 remains VALID and was not retuned.

## M. REPORT_DRAFT changes

Renamed legacy final ensemble as baseline; historical tables/SHAP/P90/decision experiments labeled as such. Added2.6(VALID selection and final TEST/CI) and4.5(residual/uncertainty/decision/ablation/pipeline), corrected copy evaluation statement and selection rationale; added reproducibility/limits. Figures: final_system fig_test_daily_errors, fig_test_decision_savings, fig_final_pipeline.

## N. Exact reproduction

See docs/PHASE3_FINAL_SYSTEM.md. Checked-in frozen config is the source of truth; no tuning/re-freeze required.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-phase2.txt
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m experiments.final_system --stage prepare
.venv/Scripts/python.exe -m experiments.final_system --stage audit
.venv/Scripts/python.exe -m experiments.final_system --stage test
.venv/Scripts/python.exe -m experiments.final_system --stage summarize
```

Use the pre-TEST commit, clean working tree and fresh outputs for an independent replication. No existing completed ledger may be overwritten. The original run executed freeze once before prepare and committed config/code before audit/test. Offline set KAMP_TABPFN_MODEL_PATH to the pinned checksum checkpoint. TabPFN9.1.0, publicv2 revision `{config['tabpfn']['revision']}`, hash `{config['tabpfn']['sha256']}`, CPU4 threads/4estimators/median/seed42. Empirical seeds0/1/2 each K30. No family/K/lambda/calibration selection or second final TEST run.
"""
    (OUT/"final_system_summary.md").write_text(full,encoding="utf-8")
    if delivery:
        dest=Path(delivery);dest.mkdir(parents=True,exist_ok=True)
        shutil.copyfile(OUT/"final_system_summary.md",dest/"phase3_final_system_report.md")
        with zipfile.ZipFile(dest/"phase3_final_results.zip","w",zipfile.ZIP_DEFLATED) as z:
            for p in OUT.iterdir():
                if p.is_file():z.write(p,"final_system/"+p.name)
            for name in ("experiments/final_system_config.json","docs/PHASE3_FINAL_SYSTEM.md","docs/REPORT_DRAFT.md","requirements-phase2.txt"):
                z.write(ROOT/name,name)
    print(comparison.to_string(index=False));print(ds.to_string(index=False));print(conclusion)


if __name__=="__main__":
    import argparse
    parser=argparse.ArgumentParser();parser.add_argument("--delivery")
    run(parser.parse_args().delivery)
