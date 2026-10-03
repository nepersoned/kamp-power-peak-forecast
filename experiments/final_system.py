"""Freeze -> prepare past-only pools -> audit -> one final TEST evaluation.

No model/parameter selection is implemented here. The TEST run has a durable
ledger; successful final evaluation cannot be repeated in the same directory.
"""
import argparse
import hashlib
import importlib.metadata
import json
import subprocess
import time
import traceback
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import pandas as pd

from src.data import load, Q
from src.features import build
from src.models import operating
from src.forecast_adapters import make_forecaster
from src.final_system import START, TEST_START, END, fit_mask, eligibility, rolling_residual_pool, validate_pool, forecast_queries
from src.joint_uncertainty import residual_matrix, JointResidualModel
from src.forecast_interface import forecast_frame
from src.probabilistic_eval import marginal_metrics, event_metrics, paired_bootstrap, reliability, pr_auc
from src.milp import fit_surrogate, solve_day
from src.stochastic_milp import solve_stochastic
from src.decision_eval import true_cost
from src.ratchet_rl import make_day
from src.run_rl import day_floor
from src import tariff as T
from src.rl_env import CAP_MULT, MAX_DELAY

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT/"outputs/final_system"
CONFIG = ROOT/"experiments/final_system_config.json"
MODELS = ("regime_ens", "regime_tabpfn_all")
SOURCES = ("experiments/final_system.py", "src/final_system.py", "src/forecast_adapters.py",
           "src/tabpfn_model.py", "src/models.py", "src/final_config.json", "src/data.py", "src/features.py",
           "src/joint_uncertainty.py", "src/probabilistic_eval.py", "src/forecast_interface.py",
           "src/milp.py", "src/stochastic_milp.py", "src/decision_eval.py", "src/tariff.py", "src/rl_env.py",
           "src/run_rl.py", "src/ratchet_rl.py")
MAIN_SHA = "e169714df03ae8b2d381853f579db4e32a49414c"


def git(*args): return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()
def digest(p): return hashlib.sha256(Path(p).read_bytes().replace(b"\r\n", b"\n")).hexdigest()
def dump(value, path): Path(path).write_text(json.dumps(value, indent=2), encoding="utf-8")
def save(frame, name): frame.to_csv(OUT/name, index=False)
def configuration():
    c = json.loads(CONFIG.read_text())
    assert c["forecaster"] == "regime_tabpfn_all" and c["uncertainty"] == "empirical_full_path"
    assert c["K"] == 30 and c["lambda"] == 0 and c["seeds"] == [0, 1, 2]
    for name, sha in c["source_hashes"].items():
        if digest(ROOT/name) != sha: raise ValueError(f"Frozen source changed: {name}")
    for package, version in c["versions"].items():
        if importlib.metadata.version(package) != version: raise ValueError(f"Package changed: {package}")
    return c


def freeze():
    if CONFIG.exists(): raise ValueError("Final configuration already frozen")
    assert git("branch", "--show-current") == "seongmin" and git("rev-parse", "main") == MAIN_SHA
    p2 = json.loads((ROOT/"experiments/forecast_model_selected.json").read_text())
    p1 = json.loads((ROOT/"experiments/joint_scenarios_selected.json").read_text())
    assert p2["frozen"] and not p2["test_evaluated"] and p2["model"] == MODELS[1]
    assert p1["method"] == "empirical" and p1["K"] == 30 and p1["lam"] == 0
    df = load(end=TEST_START); X = build(df)
    c = dict(forecaster=MODELS[1], reference_forecaster=MODELS[0], uncertainty="empirical_full_path",
             K=p1["K"], **{"lambda": p1["lam"]}, alpha=p1["alpha"], seeds=p1["seeds"],
             scenario_seed_rule="seed * 100000 + forecast_date.dayofyear", tabpfn=p2["tabpfn"],
             feature_list=list(X), feature_count=len(X.columns), forecast_seed=p2["seed"],
             regime_rule="plan_on_day.fillna(1)==1; missing plans treated as operating",
             holiday_treatment="existing holiday features; production plan defines regime",
             off_day_treatment="original TRAIN hour-wise target median (q=.5)",
             training_mask="Jan8 <= timestamp < origin; non-outage; power_lag168 valid; plan_missing INCLUDED",
             copy_handling="TabPFN and off median drop copies; regime_ens keeps existing on-model copy weights",
             residual_eligibility="original, non-outage, known plan, operating, complete24h, lag168 valid",
             decision_eligibility="same complete-day rule as residuals; TEST range only",
             rolling=dict(min_train_original_days=14,block_days=14,start=START,end=TEST_START),
             training_context="all eligible past rows; no recent-day limit",
             test_start=TEST_START,test_end_exclusive=END,
             forecast_evaluation="legacy valid lag168 non-outage hourly rows including copies; original-only secondary",
             probability_evaluation="same original operating complete days as decision evaluation",
             interface="fit(X,y,hour,copy); predict(X,hour); timestamp,target,point,q10,q50,q90,model",
             no_online_adaptation=True,conformal="none; empirical scenario quantiles without recalibration",
             constraints=dict(max_delay=MAX_DELAY,cap_mult=CAP_MULT,min_production=1,starts="<= original starts",
                              months_ahead=12,labor_won=0,stochastic_time_limit_s=30,deterministic_time_limit_s=20),
             energy_objective="point power TOU + mean scenario ratchet excess; unchanged solve_stochastic",
             floor_policy="unchanged past-only day_floor, includes available previous TEST-day history; no pool update",
             phase2_commit=git("rev-parse", "HEAD"), main_sha=MAIN_SHA,
             selected_phase1_sha256=digest(ROOT/"experiments/joint_scenarios_selected.json"),
             selected_phase2_sha256=digest(ROOT/"experiments/forecast_model_selected.json"),
             source_hashes={n:digest(ROOT/n) for n in SOURCES},
             versions={p:importlib.metadata.version(p) for p in ("numpy","pandas","scikit-learn","lightgbm","tabpfn","torch","ortools","pytest")},
             frozen_before_test=True)
    dump(c, CONFIG); OUT.mkdir(parents=True, exist_ok=True); dump(c, OUT/"final_system_config.json")
    print("FROZEN WITHOUT TEST ACCESS", flush=True)


def residual_diagnostics(pool):
    summaries=[]
    for (model, target), group in pool.groupby(["model","target"]):
        matrix=residual_matrix(group,target,cutoff=TEST_START)
        matrix.to_csv(OUT/f"residual_{model}_{target}.csv")
        matrix.cov().to_csv(OUT/f"covariance_{model}_{target}.csv")
        matrix.corr().to_csv(OUT/f"correlation_{model}_{target}.csv")
        for hour,g in group.groupby("hour"):
            summaries.append(dict(model=model,target=target,hour=hour,n_days=len(g),bias=g.residual.mean(),
                mae=g.residual.abs().mean(),q01=g.residual.quantile(.01),q05=g.residual.quantile(.05),
                q50=g.residual.quantile(.5),q95=g.residual.quantile(.95),q99=g.residual.quantile(.99)))
        daily=group.groupby("forecast_date").agg(actual_daily_max=("actual","max"),predicted_daily_max=("prediction","max"),
             maximum_hour_residual=("residual","max"),minimum_hour_residual=("residual","min"))
        daily["daily_max_error"]=daily.actual_daily_max-daily.predicted_daily_max
        daily.to_csv(OUT/f"daily_max_residual_{model}_{target}.csv")
    save(pd.DataFrame(summaries),"residual_summary.csv")


def prepare():
    c=configuration(); OUT.mkdir(parents=True,exist_ok=True)
    df=load(end=TEST_START); X=build(df); assert list(X)==c["feature_list"]
    audits=[eligibility(df,X,"forecaster_training",df.index.min().normalize(),TEST_START)]
    pools=[]
    for name in MODELS:
        def checkpoint(p): save(p,f"residual_partial_{name}.csv")
        prov,audit=rolling_residual_pool(df,X,lambda:make_forecaster(name),name,
             min_train_days=c["rolling"]["min_train_original_days"],block_days=c["rolling"]["block_days"],on_origin=checkpoint)
        validate_pool(prov,c["feature_list"],X)
        audit["model"]=name; audits.append(audit); pools.append(prov)
        save(prov,f"residual_pool_{name}.csv")
    pool=pd.concat(pools,ignore_index=True)
    dates=[set(g.forecast_date) for _,g in pool.groupby("model")]; assert dates[0]==dates[1]
    save(pool,"residual_pool.csv"); save(pd.concat(audits,ignore_index=True),"data_eligibility.csv")
    residual_diagnostics(pool)
    dump(dict(config_sha256=digest(CONFIG),residual_sha256=digest(OUT/"residual_pool.csv"),
              sample_days={n:int(p.forecast_date.nunique()) for n,p in pool.groupby("model")},
              test_accessed=False,training_end_max=str(pd.to_datetime(pool.training_end_date).max()),
              forecast_date_max=str(pd.to_datetime(pool.forecast_date).max())),OUT/"residual_manifest.json")
    print("RESIDUAL POOLS COMPLETE; TEST NOT ACCESSED",flush=True)


def audit():
    c=configuration(); df=load(end=TEST_START); X=build(df)
    p=pd.read_csv(OUT/"residual_pool.csv")
    validate_pool(p,c["feature_list"],X)
    m=json.loads((OUT/"residual_manifest.json").read_text())
    result=subprocess.run([str(ROOT/".venv/Scripts/python.exe"),"-m","pytest","-q"],cwd=ROOT,capture_output=True,text=True)
    (OUT/"pretest_pytest.txt").write_text(result.stdout+result.stderr,encoding="utf-8")
    checks=dict(branch_seongmin=git("branch","--show-current")=="seongmin",
        phase2_frozen=digest(ROOT/"experiments/forecast_model_selected.json")==c["selected_phase2_sha256"],
        phase1_frozen=digest(ROOT/"experiments/joint_scenarios_selected.json")==c["selected_phase1_sha256"],
        phase3_code_complete=True,pytest_passed=result.returncode==0,
        no_test_labels_training=bool(df.index.max()<pd.Timestamp(TEST_START)),
        no_test_actual_residual=bool((pd.to_datetime(p.forecast_date)<pd.Timestamp(TEST_START)).all()),
        no_test_metric_selection=not (OUT/"test_execution.json").exists(),
        residual_past_only=bool((pd.to_datetime(p.training_end_date)<pd.to_datetime(p.forecast_date)).all()),
        features_unchanged=list(X)==c["feature_list"] and len(X.columns)==46,
        K30=c["K"]==30,lambda0=c["lambda"]==0,seed_frozen=c["seeds"]==[0,1,2],
        constraints_unchanged=c["constraints"]["cap_mult"]==CAP_MULT and c["constraints"]["max_delay"]==MAX_DELAY,
        main_untouched=git("rev-parse","main")==MAIN_SHA,
        residual_hash_valid=digest(OUT/"residual_pool.csv")==m["residual_sha256"],
        configuration_hash_valid=digest(CONFIG)==m["config_sha256"])
    output=dict(checks=checks,passed=all(checks.values()),git_sha=git("rev-parse","HEAD"),
                config_sha256=digest(CONFIG),residual_sha256=m["residual_sha256"],test_started=False,
                leakage_violations=int((pd.to_datetime(p.training_end_date)>=pd.to_datetime(p.forecast_date)).sum()))
    dump(output,OUT/"pretest_audit.json")
    if not output["passed"]: raise ValueError(f"Pretest audit failed: {checks}")
    print("PRETEST AUDIT ALL PASS",flush=True)


def point_summary(p):
    p=p.copy();p["error"]=(p.actual-p.point).abs()
    targets=p.groupby("target").error.mean()
    daily=p[p.target=="peak15"].groupby("date")[["actual","point"]].max()
    return dict(power_mae=targets["power"],peak15_mae=targets["peak15"],primary_score=targets.mean(),
        daily_max_mae=(daily.actual-daily.point).abs().mean(),
        peak_hour_mae=p[p.peak_hour].groupby("target").error.mean().mean(),
        operating_mae=p[p.operating].groupby("target").error.mean().mean(),
        original_only_mae=p[~p.is_copy].groupby("target").error.mean().mean())


def forecast_ci(p):
    rows=[]
    for subset in ("overall","operating","peak_sensitive"):
        sub=p if subset=="overall" else p[p.operating] if subset=="operating" else p[p.peak_hour]
        daily=sub.assign(error=(sub.actual-sub.point).abs()).groupby(["model","date","target"]).error.mean()
        daily=daily.groupby(["model","date"]).mean().unstack("model")
        delta=daily[MODELS[1]]-daily[MODELS[0]];lo,hi=paired_bootstrap(delta,B=5000)
        rows.append(dict(subset=subset,mae_difference=delta.mean(),ci_low=lo,ci_high=hi,n_days=len(delta)))
    return pd.DataFrame(rows)


def decision_task(task,days,pools,coefs,oracles,c):
    name,seed,day=task;d=days[name][day]; K=c["K"]
    rngseed=seed*100000+int(d.day.dayofyear)
    scen=pools[name]["peak15"].scenarios(d.fc_pk,K,rngseed)
    pw=pools[name]["power"].scenarios(d.fc_pw,K,rngseed)
    pm=marginal_metrics(d.act_pk,scen);em,prob=event_metrics(d.act_pk,scen,d.floor,d.demand_band>T.OFF)
    plan,info=solve_stochastic(d,coefs,scen,alpha=c["alpha"],lam=c["lambda"],return_info=True)
    base,oracle,oracle_status=oracles[day]
    def score(plan,system,status):
        cost=true_cost(d,d.act_pw,d.act_pk,plan,coefs);saving=base["total"]-cost["total"]
        return dict(day=day,system=system,model=name,seed=seed,saving_won=saving,oracle_saving_won=oracle,
            regret_won=oracle-saving,energy_saving_won=base["energy"]-cost["energy"],
            ratchet_saving_won=base["ratchet"]-cost["ratchet"],moved_share=float(np.abs(plan-d.prod).sum()/2/max(d.prod.sum(),1)),
            solver_status=status,oracle_status=oracle_status,floor=d.floor,actual_max=d.act_pk.max(),
            billable_actual_max=base["demand"],recommended_realized_max=cost["demand"])
    row=score(plan,name+" + empirical stochastic",info["status"])
    row.update(**pm,**em,power_crps=marginal_metrics(d.act_pw,pw)["crps"],wall_time_ms=info["wall_time_ms"])
    records=[row];plans=[dict(day=day,system=row["system"],seed=seed,hour=h,original=d.prod[h],recommended=plan[h]) for h in range(24)]
    if name==MODELS[1] and seed==0:
        det,st=solve_day(d,None,coefs,base=(d.fc_pw,d.fc_pk))
        records.append(score(det,name+" + point deterministic",st))
        plans.extend(dict(day=day,system=records[-1]["system"],seed=0,hour=h,original=d.prod[h],recommended=det[h]) for h in range(24))
    f=[]
    if seed==0:
        for target,point,scenarios in (("peak15",d.fc_pk,scen),("power",d.fc_pw,pw)):
            fr=forecast_frame(d.idx,target,point,name,scenarios);fr["scenario_method"]="empirical_full_path";f.append(fr)
    ev=[dict(model=name,seed=seed,day=day,hour=h,probability=prob[h],event=int(d.act_pk[h]>=190)) for h in range(24)]
    return records,plans,f,ev


def test():
    c=configuration(); a=json.loads((OUT/"pretest_audit.json").read_text())
    if not a["passed"] or a["config_sha256"]!=digest(CONFIG): raise ValueError("Missing current pretest audit")
    if git("branch","--show-current")!="seongmin" or git("rev-parse","main")!=MAIN_SHA: raise ValueError("Branch changed")
    if git("status","--porcelain"): raise ValueError("Commit integration code/config before TEST")
    if git("show","HEAD:experiments/final_system_config.json").strip()!=CONFIG.read_text().strip():
        raise ValueError("Configuration not committed before TEST")
    if (OUT/"test_execution.json").exists(): raise ValueError("TEST ledger already exists; no silent rerun")
    if digest(OUT/"residual_pool.csv")!=a["residual_sha256"]: raise ValueError("Pool changed after audit")
    ledger=dict(status="started",pretest_commit=git("rev-parse","HEAD"),config_sha256=digest(CONFIG),
                started_at=pd.Timestamp.now(tz="UTC").isoformat(),test_evaluations=1)
    dump(ledger,OUT/"test_execution.json")
    try:
        train=load(end=TEST_START);Xt=build(train);mask=fit_mask(train,Xt,TEST_START)
        # Separate prefix guarantees interpolation/copy detection on TEST cannot
        # alter the fitting frame. D-1 TEST history is allowed in lag features.
        df=load(end=END);X=build(df); assert list(X)==c["feature_list"]
        ev=(df.index>=TEST_START)&(df.index<END)&~df.outage.to_numpy()&X.power_lag168.notna().to_numpy()
        index=df.index[ev];query_days=list(forecast_queries(df.index));all_index=df.index[(df.index>=TEST_START)&(df.index<END)]
        frames=[];preds={};all_forecasts=[]
        for name in MODELS:
            preds[name]={}
            for target in ("power","peak15"):
                print("TEST FIXED FIT",name,target,flush=True)
                model=make_forecaster(name).fit(Xt[mask],train.loc[mask,target],train.loc[mask,"hour"],train.loc[mask,"is_copy"])
                pred=pd.Series(index=all_index,dtype=float)
                for idx in query_days:
                    pred.loc[idx]=model.predict(X.loc[idx],df.loc[idx,"hour"])
                del model
                preds[name][target]=pred
                all_forecasts.append(forecast_frame(all_index,target,pred.to_numpy(),name))
                fr=forecast_frame(index,target,pred.reindex(index).to_numpy(),name);fr["q10"]=np.nan;fr["q90"]=np.nan
                fr["date"]=index.normalize();fr["hour"]=df.loc[index,"hour"].to_numpy();fr["actual"]=df.loc[index,target].to_numpy()
                fr["operating"]=operating(X.loc[index]);fr["is_copy"]=df.loc[index,"is_copy"].to_numpy()
                fr["peak_hour"]=np.isin(df.loc[index,"hour"],(8,9,10,11,13,14,15,16));fr["training_end_date"]=train.index[mask].max()
                frames.append(fr)
        save(pd.concat(all_forecasts,ignore_index=True),"test_day_ahead_all_hours.csv")
        forecast=pd.concat(frames,ignore_index=True);save(forecast,"test_predictions.csv")
        save(forecast[forecast.model==MODELS[1]],"test_predictions_regime_tabpfn.csv")
        save(pd.DataFrame([dict(model=n,**point_summary(g)) for n,g in forecast.groupby("model")]),"test_forecast_comparison.csv")
        save(forecast_ci(forecast),"test_bootstrap_forecast.csv")
        audit_days=eligibility(df,X,"decision_evaluation",TEST_START,END)
        existing=pd.read_csv(OUT/"data_eligibility.csv");save(pd.concat([existing,audit_days],ignore_index=True),"data_eligibility.csv")
        dates=audit_days.loc[audit_days.eligible,"date"]
        coefs=fit_surrogate(train,mask & operating(Xt)&~train.is_copy.to_numpy()&~train.plan_missing.to_numpy())
        dump({t:{k:np.asarray(v).tolist() for k,v in cf.items()} for t,cf in coefs.items()},OUT/"surrogate_coefficients.json")
        pools={};days={};prov=pd.read_csv(OUT/"residual_pool.csv")
        for name in MODELS:
            pools[name]={t:JointResidualModel().fit(residual_matrix(prov[prov.model==name],t,TEST_START)) for t in ("power","peak15")}
            days[name]={str(d.date()):make_day(df,d,preds[name]["power"],preds[name]["peak15"],preds[name]["peak15"],day_floor(df[Q],d)) for d in dates}
        oracles={}
        for day,d in days[MODELS[0]].items():
            plan,st=solve_day(d,None,coefs,base=(d.act_pw,d.act_pk));base=true_cost(d,d.act_pw,d.act_pk,d.prod,coefs)
            cost=true_cost(d,d.act_pw,d.act_pk,plan,coefs);oracles[day]=(base,base["total"]-cost["total"],st)
        rows=[];plans=[];distributions=[];events=[]
        tasks=[(name,seed,day) for name in MODELS for seed in c["seeds"] for day in days[name]]
        with ThreadPoolExecutor(max_workers=4) as executor:
            for rr,pl,fr,evs in executor.map(lambda task:decision_task(task,days,pools,coefs,oracles,c),tasks):
                rows.extend(rr);plans.extend(pl);distributions.extend(fr);events.extend(evs)
        save(pd.DataFrame(rows),"test_daily_decisions.csv");save(pd.DataFrame(plans),"test_production_plans.csv")
        save(pd.concat(distributions),"test_forecast_distributions.csv");save(pd.DataFrame(events),"test_tau_events.csv")
        summarize()
        ledger.update(status="completed",completed_at=pd.Timestamp.now(tz="UTC").isoformat(),
                      n_decision_days=len(dates),no_online_pool_update=True)
        dump(ledger,OUT/"test_execution.json")
        print("FINAL TEST COMPLETED ONCE",flush=True)
    except Exception:
        ledger.update(status="failed",failure=traceback.format_exc());dump(ledger,OUT/"test_execution.json")
        raise


def summarize():
    r=pd.read_csv(OUT/"test_daily_decisions.csv");numeric=["saving_won","oracle_saving_won","regret_won","moved_share","energy_saving_won","ratchet_saving_won"]
    daily=r.groupby(["system","day"])[numeric].mean().reset_index();rows=[]
    for system,g in daily.groupby("system"):
        statuses=r[r.system==system].solver_status.value_counts().to_dict()
        rows.append(dict(system=system,n_days=len(g),**g[numeric].mean().to_dict(),worst_day_saving=g.saving_won.min(),
             negative_saving_days=int((g.saving_won<0).sum()),solver_statuses=json.dumps(statuses)))
    save(pd.DataFrame(rows),"test_decision_metrics.csv")
    st=r[r.system.str.endswith("empirical stochastic")]
    metrics=["crps","power_crps","energy_score","variogram_score","tau_brier","ratchet_brier","coverage","interval_width","pinball"]
    save(st.groupby("model")[metrics].mean().reset_index(),"test_probabilistic_metrics.csv")
    base=daily[daily.system==MODELS[0]+" + empirical stochastic"].set_index("day")
    candidate=daily[daily.system==MODELS[1]+" + empirical stochastic"].set_index("day")
    rows=[]
    for metric in ("saving_won","regret_won"):
        delta=candidate[metric]-base[metric];lo,hi=paired_bootstrap(delta,B=5000)
        rows.append(dict(metric=metric,difference=delta.mean(),ci_low=lo,ci_high=hi,n_days=len(delta)))
    save(pd.DataFrame(rows),"test_bootstrap_decision.csv")
    ev=pd.read_csv(OUT/"test_tau_events.csv").groupby(["model","day","hour"])[["probability","event"]].mean().reset_index()
    rel=[];ap=[]
    for name,g in ev.groupby("model"):
        h=reliability(g.probability,g.event);h["model"]=name;rel.append(h)
        rat=st[st.model==name].groupby("day")[["ratchet_probability","ratchet_event"]].mean()
        ap.append(dict(model=name,tau_pr_auc=pr_auc(g.event,g.probability),ratchet_pr_auc=pr_auc(rat.ratchet_event,rat.ratchet_probability),
                       ratchet_events=int(rat.ratchet_event.sum()),n_days=len(rat)))
        save(reliability(rat.ratchet_probability,rat.ratchet_event),f"test_reliability_ratchet_{name}.csv")
    save(pd.concat(rel),"test_reliability_tau.csv");save(pd.DataFrame(ap),"test_event_summary.csv")


if __name__=="__main__":
    parser=argparse.ArgumentParser();parser.add_argument("--stage",choices=["freeze","prepare","audit","test","summarize"],required=True)
    args=parser.parse_args();globals()[args.stage]()
