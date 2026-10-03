"""Phase 2 fixed two-fold model search. No TEST or decision-model imports.

Stages: base -> tabpfn -> analysis -> rolling. All run in the same frozen
protocol. Primary = mean of four fold/target hourly MAEs (legacy harness).
"""
import argparse
import hashlib
import itertools
import json
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
from src.data import load
from src.features import build
from src.models import operating
from src.forecast_adapters import make_forecaster
from src.forecast_interface import forecast_frame
from src.eval_conditions import condition_frame

ROOT=Path(__file__).resolve().parents[1]
OUT=ROOT/"outputs/forecast_models"
FOLDS=(("2021-07-02","2021-07-16"),("2021-07-16","2021-08-16"))
START="2021-01-08"
END="2021-08-16"
TARGETS=("power","peak15")
BASE_MODELS=("naive_24h","naive_168h","ridge","lgbm","regime_lgbm","regime_ens")


def save(d,name): d.to_csv(OUT/name,index=False)


def data():
    df=load(end=END); X=build(df)
    assert df.index.max()<pd.Timestamp(END)
    return df,X


def masks(df,X,a,b):
    if pd.Timestamp(b)>pd.Timestamp(END): raise ValueError("TEST period forbidden")
    ok=~df.outage.to_numpy() & X.power_lag168.notna().to_numpy()
    # Exact legacy harness masks: copies retained in evaluation and passed to
    # model-specific training logic, missing plans handled by existing features.
    tr=ok & (df.index>=START)&(df.index<a)
    ev=ok & (df.index>=a)&(df.index<b)
    return tr,ev


def rows_for(df,X,ev,p,target,name,fold,fit_end,runtime):
    idx=df.index[ev]; f=forecast_frame(idx,target,p,name)
    f["q10"]=np.nan; f["q90"]=np.nan # point-only, NOT calibrated intervals
    cf=condition_frame(df,X,ev)
    f["date"]=idx.normalize(); f["hour"]=df.loc[idx,"hour"].to_numpy()
    f["actual"]=df.loc[idx,target].to_numpy(); f["prediction"]=p
    f["fold"]=fold; f["operating"]=operating(X[ev]); f["is_copy"]=df.loc[idx,"is_copy"].to_numpy()
    f["transition"]=cf.transition.ne("평상").to_numpy()
    f["peak_hour"]=cf.peak_window.eq("주간 피크대").to_numpy()
    f["training_end_date"]=fit_end; f["fit_predict_seconds"]=runtime
    return f


def evaluate_model(name,df,X,folds=FOLDS):
    rows=[]; diagnostics=[]
    for fold,(a,b) in enumerate(folds,1):
        tr,ev=masks(df,X,a,b)
        for target in TARGETS:
            tic=time.perf_counter()
            if name=="sarimax":
                from src.sarimax_model import SARIMAXDaily
                m=SARIMAXDaily().fit(X[tr],df.loc[tr,target],copy=df.loc[tr,"is_copy"])
                fit_end=m.training_end; pred=pd.Series(index=df.index[ev],dtype=float)
                for day in pd.date_range(a,b,inclusive="left"):
                    idx=pd.date_range(day,periods=24,freq="h")
                    p=m.forecast_day(X.loc[idx]); pred.loc[idx]=p
                    # Update state only after all 24 forecast values are made.
                    obs=df.loc[idx,target].mask(df.loc[idx,"outage"])
                    m.observe_day(X.loc[idx],obs,copy=df.loc[idx,"is_copy"])
                p=pred.loc[df.index[ev]].to_numpy()
                diagnostics.append(dict(model=name,fold=fold,target=target,converged=m.converged,
                                        training_samples=m.n_observed,features=10))
            else:
                m=make_forecaster(name).fit(X[tr],df.loc[tr,target],df.loc[tr,"hour"],df.loc[tr,"is_copy"])
                fit_end=df.index[tr].max()
                if "tabpfn" in name:
                    pred=pd.Series(index=df.index[ev],dtype=float)
                    infer_seconds=0.
                    for day in df.index[ev].normalize().unique():
                        sel=ev & (df.index.normalize()==day)
                        q=time.perf_counter(); pred.loc[df.index[sel]]=m.predict(X[sel],df.loc[sel,"hour"])
                        infer_seconds+=time.perf_counter()-q
                        print("TabPFN day",name,fold,target,str(day.date()),"inference_total",round(infer_seconds,1),flush=True)
                    p=pred.to_numpy()
                    inner=m.on_model if name.startswith("regime_") else m
                    diagnostics.append(dict(model=name,fold=fold,target=target,features=inner.n_features,
                                            training_samples=inner.n_samples,fit_seconds=inner.fit_seconds,
                                            inference_seconds=infer_seconds,device="cpu",model_version="v2"))
                else: p=m.predict(X[ev],df.loc[ev,"hour"])
            runtime=time.perf_counter()-tic
            if not np.isfinite(p).all(): raise ValueError(f"Nonfinite predictions: {name}")
            rows.append(rows_for(df,X,ev,p,target,name,fold,fit_end,runtime))
            print(name,fold,target,"MAE",round(float(np.abs(df.loc[ev,target]-p).mean()),4),"seconds",round(runtime,2),flush=True)
    return pd.concat(rows,ignore_index=True),pd.DataFrame(diagnostics)


def stage_models(names):
    df,X=data()
    for name in names:
        try:
            p,diag=evaluate_model(name,df,X)
            save(p,f"valid_{name}.csv")
            if len(diag): save(diag,f"diagnostics_{name}.csv")
        except Exception:
            error=traceback.format_exc()
            (OUT/f"failure_{name}.txt").write_text(error,encoding="utf-8")
            print("FAILED",name,error,flush=True)
            if name in BASE_MODELS: raise


def primary(p):
    return float(p.assign(error=(p.actual-p.prediction).abs()).groupby(["fold","target"]).error.mean().mean())


def summary(p):
    r=p.copy(); r["error"]=(r.actual-r.prediction).abs()
    def mae(sub):
        return float(sub.groupby(["fold","target"]).error.mean().mean()) if len(sub) else np.nan
    daily=r.groupby(["fold","target","date"])[["actual","prediction"]].max()
    dm=(daily.actual-daily.prediction).abs().groupby(["fold","target"]).mean().mean()
    ft=r.groupby(["fold","target"]).error.mean()
    dm_target=(daily.actual-daily.prediction).abs().groupby(["fold","target"]).mean()
    return dict(model=r.model.iloc[0],power_mae=ft.xs("power",level="target").mean(),
                peak15_mae=ft.xs("peak15",level="target").mean(),primary_score=primary(r),
                daily_max_power_mae=dm_target.xs("power",level="target").mean(),
                daily_max_peak15_mae=dm_target.xs("peak15",level="target").mean(),
                daily_max_mae=dm,peak_hour_mae=mae(r[r.peak_hour]),transition_mae=mae(r[r.transition]),
                operating_mae=mae(r[r.operating]),original_only_score=mae(r[~r.is_copy]),
                runtime_seconds=r.groupby(["fold","target"]).fit_predict_seconds.first().sum())


def load_predictions():
    P={}
    for f in OUT.glob("valid_*.csv"):
        if f.stem.startswith("valid_blend"): continue
        p=pd.read_csv(f,parse_dates=["timestamp","date","training_end_date"])
        if (p.timestamp>=pd.Timestamp(END)).any(): raise ValueError("TEST contamination")
        P[p.model.iloc[0]]=p.sort_values(["fold","target","timestamp"]).reset_index(drop=True)
    base=P["regime_ens"]
    keys=["fold","target","timestamp","actual","is_copy"]
    for n,p in P.items():
        pd.testing.assert_frame_equal(p[keys],base[keys],check_dtype=False)
    return P


def blend(base,candidate,w,name):
    p=base.copy(); q=np.array([w[t] for t in p.target])
    p["prediction"]=q*base.prediction.to_numpy()+(1-q)*candidate.prediction.to_numpy()
    p["point"]=p.prediction; p["q50"]=p.prediction; p["model"]=name
    p["fit_predict_seconds"]=base.fit_predict_seconds+candidate.fit_predict_seconds
    return p


def diversity(P):
    rows=[]; daily=[]
    for a,b in itertools.combinations(P,2):
        pa,pb=P[a],P[b]
        for (fold,target),g in pa.groupby(["fold","target"]):
            ix=g.index; ea=g.actual-g.prediction; eb=pb.loc[ix,"actual"]-pb.loc[ix,"prediction"]
            rows.append(dict(a=a,b=b,fold=fold,target=target,pearson=ea.corr(eb),spearman=ea.corr(eb,method="spearman"),
                             error_covariance=np.cov(ea,eb)[0,1],disagreement=np.abs(g.prediction-pb.loc[ix,"prediction"]).mean()))
            v=pd.DataFrame(dict(date=g.date,error_a=ea.abs(),error_b=eb.abs())).groupby("date").mean()
            for day,row in v.iterrows(): daily.append(dict(a=a,b=b,fold=fold,target=target,date=day,
                                                          **row.to_dict(),winner=a if row.error_a<row.error_b else b))
    save(pd.DataFrame(rows),"residual_diversity_valid.csv"); save(pd.DataFrame(daily),"model_pair_daily_winners.csv")


def paired_ci(candidate,base,subset="all",B=5000):
    sel=np.ones(len(base),bool)
    if subset=="operating": sel=base.operating.to_numpy()
    if subset=="peak_hour": sel=base.peak_hour.to_numpy()
    if subset=="actual_peak":
        hot=base.loc[(base.target=="peak15")&(base.actual>=190),"timestamp"]
        sel=base.timestamp.isin(hot).to_numpy()
    d=base.loc[sel,["fold","target","date"]].copy()
    d["delta"]=np.abs(candidate.loc[sel,"actual"]-candidate.loc[sel,"prediction"])-np.abs(base.loc[sel,"actual"]-base.loc[sel,"prediction"])
    daily=d.groupby(["fold","date"]).delta.mean()
    rng=np.random.default_rng(42)
    boot=[]
    # Stratify dates by fold so the shorter fold retains equal primary weight.
    for fold in daily.index.get_level_values("fold").unique():
        v=daily.xs(fold).to_numpy(); boot.append(v[rng.integers(len(v),size=(B,len(v)))].mean(axis=1))
    samples=np.mean(boot,axis=0)
    return dict(model=candidate.model.iloc[0],subset=subset,mae_diff=daily.groupby(level="fold").mean().mean(),
                ci_low=np.quantile(samples,.025),ci_high=np.quantile(samples,.975),n_days=len(daily))


def analysis():
    P=load_predictions(); base=P["regime_ens"]
    # The legacy tuner used COPY-inclusive off medians while deployed
    # RegimeModel drops off copies. Diagnose without replacing the baseline.
    df,X=data(); on=operating(X); legacy=base.copy()
    for fold,(a,b) in enumerate(FOLDS,1):
        tr,_=masks(df,X,a,b)
        for t in TARGETS:
            prof=df.loc[tr & ~on,t].groupby(df.loc[tr & ~on,"hour"]).median()
            sel=(legacy.fold==fold)&(legacy.target==t)&~legacy.operating
            legacy.loc[sel,"prediction"]=legacy.loc[sel,"hour"].map(prof)
    save(pd.DataFrame([dict(canonical_score=primary(base),legacy_off_copies_score=primary(legacy))]),"legacy_off_profile_diagnostic.csv")
    singles=pd.DataFrame([summary(p) for p in P.values()]).sort_values("primary_score")
    save(singles,"single_model_comparison_valid.csv"); diversity(P)
    grids=[]; blends={}; specs={}
    # All eligible 2-model pairs with the champion, not brute-force stacking.
    for n,p in P.items():
        if n=="regime_ens" or n.startswith("naive"): continue
        trials=[]
        for i in range(11):
            w=i/10; weights={t:w for t in TARGETS}
            v=blend(base,p,weights,f"blend_{n}")
            row=dict(candidate=n,scheme="common",weight_power=w,weight_peak15=w,**summary(v))
            grids.append(row); trials.append((row["primary_score"],weights,v))
        _,weights,v=min(trials,key=lambda z:z[0]); name=f"blend_{n}"
        if 0<weights["power"]<1:
            blends[name]=v; specs[name]=dict(components=["regime_ens",n],weights={t:[weights[t],1-weights[t]] for t in TARGETS})
        # Target-specific optimum from the SAME coarse grid, no nonlinear meta learner.
        tw={}
        for t in TARGETS:
            tw[t]=min((r for r in grids if r["candidate"]==n),key=lambda r:r[f"{t}_mae"])["weight_power"]
        tv=blend(base,p,tw,f"blend_target_{n}")
        grids.append(dict(candidate=n,scheme="target_specific",weight_power=tw["power"],weight_peak15=tw["peak15"],**summary(tv)))
        # Extra target weight only retained if >=0.1 kW primary benefit over common.
        if primary(v)-primary(tv)>=.1:
            name=f"blend_target_{n}"; blends[name]=tv
            specs[name]=dict(components=["regime_ens",n],weights={t:[tw[t],1-tw[t]] for t in TARGETS})
    # A single justified third model; no exhaustive model-combination search.
    third_log=[]
    if blends:
        best_two=min(blends,key=lambda n:primary(blends[n]))
        members=specs[best_two]["components"]
        if primary(base)-primary(blends[best_two])>=.05:
            eligible_third=[]
            for n,p in P.items():
                if n in members or n.startswith("naive") or n=="sarimax": continue
                corr=[]
                for key,g in blends[best_two].groupby(["fold","target"]):
                    corr.append((g.actual-g.prediction).corr(p.loc[g.index,"actual"]-p.loc[g.index,"prediction"]))
                rho=float(np.mean(corr))
                ok=primary(p)<=1.2*primary(base) and rho<.95
                third_log.append(dict(candidate=n,residual_correlation=rho,primary=primary(p),eligible=ok))
                if ok: eligible_third.append(n)
            if eligible_third:
                third=min(eligible_third,key=lambda n:primary(P[n])); second=members[1]; trials=[]
                for i in range(11):
                    for j in range(11-i):
                        weights=[i/10,j/10,(10-i-j)/10]
                        v=base.copy(); v["prediction"]=sum(w*P[n].prediction for w,n in zip(weights,["regime_ens",second,third]))
                        v["point"]=v.prediction; v["q50"]=v.prediction; v["model"]=f"blend3_{second}_{third}"
                        v["fit_predict_seconds"]=sum(P[n].fit_predict_seconds for w,n in zip(weights,["regime_ens",second,third]) if w>0)
                        trials.append((primary(v),weights,v))
                        grids.append(dict(candidate=f"{second}+{third}",scheme="three_common",weights=json.dumps(weights),**summary(v)))
                _,weights,v=min(trials,key=lambda z:z[0])
                if primary(blends[best_two])-primary(v)>=.1:
                    name=v.model.iloc[0]; blends[name]=v
                    specs[name]=dict(components=["regime_ens",second,third],weights={t:weights for t in TARGETS})
    save(pd.DataFrame(third_log,columns=["candidate","residual_correlation","primary","eligible"]),"third_model_screening.csv")
    save(pd.DataFrame(grids),"ensemble_weight_grid_valid.csv")
    combined={**P,**blends}; ranking=pd.DataFrame([summary(p) for p in combined.values()]).sort_values("primary_score")
    save(ranking,"model_ensemble_comparison_valid.csv")
    folds=[]
    for n,p in combined.items():
        for fold,g in p.groupby("fold"):
            row=summary(g); row["fold"]=fold; folds.append(row)
    f=pd.DataFrame(folds); save(f,"fold_comparison_valid.csv")
    basefold=f[f.model=="regime_ens"].set_index("fold").primary_score
    eligible=[]
    for n,p in combined.items():
        v=f[f.model==n].set_index("fold").primary_score
        members=specs.get(n,dict(components=[n]))["components"]
        if "sarimax" in members:
            diag=pd.read_csv(OUT/"diagnostics_sarimax.csv")
            if not diag.converged.all(): continue # baseline retained, unstable fit not deployed
        if n=="regime_ens" or (primary(base)-primary(p)>=.05 and (v<=basefold+.2).all()): eligible.append(n)
    winner=min(eligible,key=lambda n:primary(combined[n]))
    finalists=["regime_ens"]+[n for n in ranking.model if n!="regime_ens" and n in eligible][:2]
    boots=[paired_ci(combined[n],base,s) for n in finalists if n!="regime_ens" for s in ("all","operating","peak_hour","actual_peak")]
    save(pd.DataFrame(boots,columns=["model","subset","mae_diff","ci_low","ci_high","n_days"]),"paired_bootstrap_valid.csv")
    for n in finalists: save(combined[n],f"finalist_{n}.csv")
    config=dict(phase=2,model=winner,preliminary=True,finalists=finalists,
                specification=specs.get(winner,dict(components=[winner],weights={t:[1.] for t in TARGETS})),
                finalist_specifications={n:specs.get(n,dict(components=[n],weights={t:[1.] for t in TARGETS})) for n in finalists},
                primary="mean over two folds and two target hourly MAEs",folds=FOLDS,
                adoption_rule="primary improvement >=0.05 kW; neither fold worsens >0.2 kW; rolling robustness check next",
                test_evaluated=False,phase1_unchanged=True)
    config["three_model_status"]="not run: no genuine two-model improvement" if not third_log else "screened at most one third model"
    config["ridge_stacking_status"]="not run: no additional complexity justified"
    (OUT/"selected_forecaster.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    print(ranking.to_string(index=False),flush=True); print("PRELIMINARY",winner,flush=True)


def rolling():
    config=json.loads((OUT/"selected_forecaster.json").read_text())
    df,X=data(); rows=[]; components={}
    # Same weekly origins as existing rolling evaluator, truncated before TEST.
    weeks=[(str(a.date()),str(min(a+pd.Timedelta(days=7),pd.Timestamp(END)).date()))
           for a in pd.date_range("2021-07-12","2021-08-09",freq="7D")]
    needed=set(n for spec in config["finalist_specifications"].values() for n in spec["components"])
    for component in sorted(needed):
        p,_=evaluate_model(component,df,X,folds=weeks)
        p=p[~p.is_copy].sort_values(["fold","target","timestamp"]).reset_index(drop=True)
        components[component]=p; save(p,f"rolling_component_{component}.csv")
    for name in config["finalists"]:
        spec=config["finalist_specifications"][name]
        p=components[spec["components"][0]].copy(); pred=np.zeros(len(p)); runtime=np.zeros(len(p))
        for i,n in enumerate(spec["components"]):
            w=np.array([spec["weights"][t][i] for t in p.target])
            pred+=w*components[n].prediction.to_numpy()
            runtime+=np.where(w>0,components[n].fit_predict_seconds.to_numpy(),0)
        p["prediction"]=pred; p["point"]=pred; p["q50"]=pred; p["model"]=name
        p["fit_predict_seconds"]=runtime; rows.append(p)
    allp=pd.concat(rows,ignore_index=True); save(allp,"rolling_finalist_predictions.csv")
    tab=pd.DataFrame([summary(g) for _,g in allp.groupby("model")]); save(tab,"rolling_finalist_summary.csv")
    wf=allp.assign(error=(allp.actual-allp.prediction).abs()).groupby(["model","fold","target"]).error.mean().reset_index()
    save(wf,"rolling_weekly_mae.csv")
    candidate=config["model"]
    if candidate!="regime_ens":
        scores=wf.groupby(["model","fold"]).error.mean().unstack("model")
        delta=scores[candidate]-scores.regime_ens
        config["rolling_delta_mean"]=float(delta.mean()); config["rolling_weeks_better"]=int((delta<0).sum())
        if delta.mean()>.2 or (delta<0).sum()<len(delta)/2:
            config["rejected_preliminary"]=candidate; config["model"]="regime_ens"
            config["specification"]=dict(components=["regime_ens"],weights={t:[1.] for t in TARGETS})
    config["preliminary"]=False
    config["frozen"]=True
    (OUT/"selected_forecaster.json").write_text(json.dumps(config,indent=2),encoding="utf-8")
    print("FROZEN",config["model"],"TEST NOT USED",flush=True)


def main(stage):
    OUT.mkdir(parents=True,exist_ok=True)
    phase1=(ROOT/"experiments/joint_scenarios_selected.json").read_bytes()
    if stage=="base":
        (OUT/"protocol.json").write_text(json.dumps(dict(folds=FOLDS,primary="(MAE_power_f1+MAE_peak15_f1+MAE_power_f2+MAE_peak15_f2)/4",
            evaluation="legacy OK including copies; original-only secondary",phase1_sha256=hashlib.sha256(phase1).hexdigest(),
            no_test=True,ensemble_grid=list(np.arange(11)/10),adoption=">=0.05 kW primary; each fold <=champion+0.2; target weights require extra0.1; rolling majority and mean<=champion+0.2"),indent=2),encoding="utf-8")
        stage_models([*BASE_MODELS,"sarimax"])
    if stage=="tabpfn":
        # Stop sensitivity after failure; an unavailable model cannot be blended.
        names=["tabpfn_all","regime_tabpfn_all","regime_tabpfn_28","regime_tabpfn_14"]
        for n in names:
            stage_models([n])
            if not (OUT/f"valid_{n}.csv").exists(): break
    if stage=="sarimax": stage_models(["sarimax"])
    if stage=="analysis": analysis()
    if stage=="rolling": rolling()
    assert phase1==(ROOT/"experiments/joint_scenarios_selected.json").read_bytes()


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--stage",choices=["base","sarimax","tabpfn","analysis","rolling"],required=True)
    main(parser.parse_args().stage)
