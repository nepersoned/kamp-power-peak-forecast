"""Phase 1 TRAIN -> VALID only. Never fit/select/evaluate on TEST.

Run: python -m experiments.joint_scenarios
Residual cache: --reuse-residuals (verified input/config hashes required).
"""
import argparse
import hashlib
import importlib.metadata
import json
import warnings
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data import load, RAW, Q
from src.features import build
from src.models import final_model, operating
from src.joint_uncertainty import (START, VALID_START, TEST_START, TARGETS, ScenarioConfig,
                                  JointResidualModel, training_mask, eligible_days,
                                  rolling_residuals, residual_matrix)
from src.probabilistic_eval import marginal_metrics, event_metrics, paired_bootstrap, reliability, pr_auc
from src.forecast_interface import forecast_frame
from src.milp import fit_surrogate, solve_day, _S_numeric
from src.decision_eval import true_cost
from src.stochastic_milp import solve_stochastic
from src.ratchet_rl import make_day
from src.run_rl import day_floor
from src import tariff as T

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs" / "joint_uncertainty"
CASE = "2021-07-19"
SEEDS = (0, 1, 2)
PRIMARY_K = 30


def save(frame, name):
    frame.to_csv(OUT / name, index=False)


def candidates():
    return ([ScenarioConfig("empirical"), ScenarioConfig("independent")]
            + [ScenarioConfig("cholesky", estimator=e) for e in ("empirical", "ledoit_wolf", "oas")]
            + [ScenarioConfig(m, rank=r) for m in ("pca_gaussian", "pca_bootstrap") for r in (2, 3, 4, 5, 6, 8)])


def analysis(matrix, target):
    m = JointResidualModel().fit(matrix)
    matrix.to_csv(OUT / f"residual_{target}.csv")
    matrix.corr().to_csv(OUT / f"residual_corr_{target}.csv")
    matrix.cov().to_csv(OUT / f"residual_cov_{target}.csv")
    ratio = m.eigenvalues / m.eigenvalues.sum()
    save(pd.DataFrame(dict(component=np.arange(1, len(ratio)+1), eigenvalue=m.eigenvalues,
                           explained_variance=ratio, cumulative=ratio.cumsum())), f"pca_explained_variance_{target}.csv")
    pd.DataFrame(m.loadings, columns=range(24)).to_csv(OUT / f"pca_loadings_{target}.csv", index_label="component_zero_based")
    pd.DataFrame(m.scores, index=matrix.index).to_csv(OUT / f"pca_scores_{target}.csv")
    rec = [dict(rank=r, explained_variance=float(ratio[:r].sum()),
                reconstruction_rmse=float(np.sqrt(np.mean((matrix.to_numpy()-m.reconstruct(m.scores, r))**2))))
           for r in (2,3,4,5,6,8)]
    save(pd.DataFrame(rec), f"pca_reconstruction_{target}.csv")
    diagnostics = []
    for e in ("empirical", "ledoit_wolf", "oas"):
        v = JointResidualModel(ScenarioConfig("cholesky", e)).fit(matrix)
        eig = np.linalg.eigvalsh(v.covariance)
        diagnostics.append(dict(estimator=e, n_days=len(matrix), min_raw_eigenvalue=v.raw_min_eigenvalue,
                                eigenvalue_floor=v.eigenvalue_floor, min_eigenvalue=eig.min(),
                                max_eigenvalue=eig.max(), condition=eig.max()/eig.min()))
        pd.DataFrame(v.covariance).to_csv(OUT / f"covariance_{target}_{e}.csv")
    save(pd.DataFrame(diagnostics), f"covariance_diagnostics_{target}.csv")
    fig, ax = plt.subplots(figsize=(7,6))
    im=ax.imshow(matrix.corr(), vmin=-1,vmax=1,cmap="coolwarm"); fig.colorbar(im,ax=ax)
    ax.set(xlabel="Hour",ylabel="Hour",title=f"{target}: past-only operating residual correlation (n={len(matrix)})")
    fig.tight_layout(); fig.savefig(OUT/f"fig_residual_corr_{target}.png"); plt.close(fig)
    fig,ax=plt.subplots(); ax.bar(np.arange(1,len(ratio)+1),ratio); ax.set(xlabel="Component",ylabel="Variance fraction")
    fig.tight_layout(); fig.savefig(OUT/f"fig_pca_spectrum_{target}.png"); plt.close(fig)
    fig,ax=plt.subplots()
    for i in range(4): ax.plot(range(24),m.loadings[i],label=f"PC{i+1}")
    ax.legend(); ax.set(xlabel="Hour",ylabel="Loading")
    fig.tight_layout(); fig.savefig(OUT/f"fig_pca_loadings_{target}.png"); plt.close(fig)


def summarize(metrics):
    numeric = ["mae", "power_mae", "peak_mae", "daily_max_mae", "crps", "power_crps", "energy_score", "variogram_score",
               "coverage", "interval_width", "pinball", "tau_brier", "ratchet_brier", "saving_won",
               "oracle_saving_won", "regret_won", "ratchet_saving_won", "energy_saving_won", "moved_share"]
    rows=[]
    for subset in ("all", "without_0719"):
        sub=metrics if subset=="all" else metrics[metrics.day!=CASE]
        daily=sub.groupby(["method","K","day"])[numeric].mean().reset_index()
        for (method,K),g in daily.groupby(["method","K"]):
            row=dict(method=method,K=K,subset=subset,n_days=len(g),**g[numeric].mean().to_dict())
            row["worst_day_saving_won"]=g.saving_won.min()
            repeated=sub[(sub.method==method)&(sub.K==K)]
            row["worst_seed_day_saving_won"]=repeated.saving_won.min()
            row["negative_seed_days"]=int((repeated.saving_won<0).sum())
            lo,hi=paired_bootstrap(g.saving_won)
            row.update(saving_ci_low=lo,saving_ci_high=hi)
            baseline=daily[(daily.method=="empirical")&(daily.K==K)].set_index("day")
            paired=g.set_index("day").saving_won-baseline.saving_won
            if paired.notna().all():
                lo,hi=paired_bootstrap(paired)
                row.update(delta_saving_empirical=paired.mean(),paired_ci_low=lo,paired_ci_high=hi)
            rows.append(row)
    return pd.DataFrame(rows)


def select(table):
    """Predeclared parsimonious rule: leave-case-out regret, positive paired CI.

    Adopt complexity only with positive lower 95% paired saving difference and
    no worse Energy Score or ratchet Brier. Otherwise retain empirical.
    """
    t=table[(table.subset=="without_0719")&(table.K==PRIMARY_K)]
    base=t[t.method=="empirical"].iloc[0]
    eligible=t[(t.method!="independent") & (t.paired_ci_low>0)
               & (t.energy_score<=base.energy_score) & (t.ratchet_brier<=base.ratchet_brier)]
    return "empirical" if eligible.empty else eligible.sort_values(["regret_won","energy_score","method"]).iloc[0].method


def finalize_tables(metrics):
    """Derive tables without refitting models or rerunning decisions."""
    table=summarize(metrics)
    save(table,"scenario_metrics_valid.csv"); save(table,"decision_metrics_valid.csv")
    save(table[table.method.str.startswith("pca")],"pca_dimension_sensitivity_valid.csv")
    save(table[table.method.str.startswith("cholesky")],"covariance_comparison_valid.csv")
    ranks=table.copy()
    for metric,ascending in (("mae",True),("crps",True),("energy_score",True),("saving_won",False),("regret_won",True)):
        ranks[f"rank_{metric}"]=ranks.groupby(["K","subset"])[metric].rank(method="min",ascending=ascending)
    save(ranks,"forecast_decision_ranks_valid.csv")
    seed_rows=[]
    for subset in ("all","without_0719"):
        sub=metrics if subset=="all" else metrics[metrics.day!=CASE]
        v=sub.groupby(["method","K","seed"])[["saving_won","regret_won","energy_score","crps"]].mean().reset_index()
        v["subset"]=subset; seed_rows.append(v)
    save(pd.concat(seed_rows),"seed_sensitivity_valid.csv")
    chosen=select(table)
    files=sorted(OUT.glob(f"forecast_dist_regime_ens_{chosen}_2021-*.csv"))
    if files: save(pd.concat([pd.read_csv(f) for f in files]),"forecast_dist_regime_ens.csv")
    return table


def evaluate_task(task, c, K, fitted, coefs, oracles):
    seed,d=task
    day=str(d.day.date()); rngseed=seed*100000+int(d.day.dayofyear)
    scen=fitted[c.name]["peak15"].scenarios(d.fc_pk,K,rngseed)
    pw=fitted[c.name]["power"].scenarios(d.fc_pw,K,rngseed)
    pm=marginal_metrics(d.act_pk,scen)
    em,tauprob=event_metrics(d.act_pk,scen,d.floor,d.demand_band>T.OFF)
    plan,info=solve_stochastic(d,coefs,scen,alpha=.9,lam=0,return_info=True)
    base,oracle,ost=oracles[day]; cost=true_cost(d,d.act_pw,d.act_pk,plan,coefs)
    saving=base["total"]-cost["total"]
    row=dict(day=day,method=c.name,K=K,seed=seed,**pm,**em,
             power_crps=marginal_metrics(d.act_pw,pw)["crps"],
             power_mae=float(np.abs(d.act_pw-d.fc_pw).mean()),
             mae=float(np.abs(d.act_pk-d.fc_pk).mean()),
             peak_mae=float(np.abs(d.act_pk[d.act_pk>=190]-d.fc_pk[d.act_pk>=190]).mean()) if (d.act_pk>=190).any() else np.nan,
             daily_max_mae=abs(d.act_pk.max()-d.fc_pk.max()),
             saving_won=saving,oracle_saving_won=oracle,regret_won=oracle-saving,
             energy_saving_won=base["energy"]-cost["energy"],ratchet_saving_won=base["ratchet"]-cost["ratchet"],
             moved_share=float(np.abs(plan-d.prod).sum()/2/max(d.prod.sum(),1)),
             solver_status=info["status"],oracle_status=ost,wall_time_ms=info["wall_time_ms"],
             objective=info.get("objective",np.nan),best_bound=info.get("best_bound",np.nan))
    plans=[dict(day=day,method=c.name,K=K,seed=seed,hour=h,original=d.prod[h],recommended=plan[h]) for h in range(24)]
    events=[dict(day=day,method=c.name,K=K,seed=seed,hour=h,probability=tauprob[h],event=int(d.act_pk[h]>=190)) for h in range(24)]
    case=None
    if day==CASE:
        mx=scen[:,d.demand_band>T.OFF].max(axis=1)
        case=dict(**row,floor=d.floor,actual_max=d.act_pk.max(),actual_billable_max=base["demand"],
                  realized_plan_max=cost["demand"],**{f"p{q}":np.quantile(mx,q/100) for q in (50,75,90,95)})
        if K==30 and seed==0:
            np.save(OUT/f"scenarios_0719_{c.name}.npy",scen)
    if K==30 and seed==0:
        f=pd.concat([forecast_frame(d.idx,"peak15",d.fc_pk,"regime_ens",scen),
                     forecast_frame(d.idx,"power",d.fc_pw,"regime_ens",pw)])
        f["scenario_method"]=c.name
        f.to_csv(OUT/f"forecast_dist_regime_ens_{c.name}_{day}.csv",index=False)
    return row,plans,events,case


def run(reuse=False, workers=4):
    warnings.filterwarnings("ignore", category=FutureWarning)
    OUT.mkdir(parents=True, exist_ok=True)
    fingerprint=hashlib.sha256(RAW.read_bytes()+b"".join((ROOT/f).read_bytes() for f in
                              ("src/final_config.json","src/joint_uncertainty.py","src/models.py",
                               "src/features.py","src/data.py"))).hexdigest()
    cache=OUT/"residual_provenance.csv"
    manifest=OUT/"manifest.json"
    # Load historical data with existing cleaning, then hard-truncate before
    # feature building, model fitting, scoring and decision evaluation.
    df=load(end=TEST_START)
    X=build(df)
    if reuse:
        if not manifest.exists() or json.loads(manifest.read_text())["fingerprint"]!=fingerprint:
            raise ValueError("Residual cache fingerprint mismatch; rerun without --reuse-residuals")
        prov=pd.read_csv(cache,parse_dates=["forecast_date","training_end_date"])
    else:
        print("Generating strictly past-only rolling TRAIN residuals",flush=True)
        prov,audit=rolling_residuals(df,X,final_model)
        save(prov,"residual_provenance.csv"); save(audit,"train_day_audit.csv")
        manifest.write_text(json.dumps(dict(fingerprint=fingerprint,complete=False)),encoding="utf-8")
    pools={t:residual_matrix(prov,t) for t in TARGETS}
    for t,E in pools.items(): analysis(E,t)
    mask=training_mask(df,X,VALID_START)
    preds={}
    vmask=(df.index>=VALID_START)&(df.index<TEST_START)
    for t in TARGETS:
        model=final_model().fit(X[mask],df.loc[mask,t],df.loc[mask,"hour"],df.loc[mask,"is_copy"])
        preds[t]=pd.Series(model.predict(X[vmask],df.loc[vmask,"hour"]),index=df.index[vmask])
    coefs=fit_surrogate(df,mask & operating(X) & ~df.is_copy.to_numpy())
    audit=eligible_days(df,X,VALID_START,TEST_START); save(audit,"valid_day_audit.csv")
    days=[make_day(df,d,preds["power"],preds["peak15"],preds["peak15"],day_floor(df[Q],d))
          for d in audit.loc[audit.reason=="","forecast_date"]]
    oracles={}
    for d in days:
        op,st=solve_day(d,None,coefs,base=(d.act_pw,d.act_pk))
        base=true_cost(d,d.act_pw,d.act_pk,d.prod,coefs)
        oc=true_cost(d,d.act_pw,d.act_pk,op,coefs)
        oracles[str(d.day.date())]=(base,base["total"]-oc["total"],st)
    configs=candidates(); fitted={c.name:{t:JointResidualModel(c).fit(E) for t,E in pools.items()} for c in configs}
    rows, plans, events, cases=[],[],[],[]
    # K sensitivity is TRAIN/VALID-only and uses every method: no outcome-based
    # screening before K=50. Three seeds quantify scenario instability.
    for K in (30,50):
        for c in configs:
            print(f"VALID K={K} {c.name}",flush=True)
            tasks=[(seed,d) for seed in SEEDS for d in days]
            with ThreadPoolExecutor(max_workers=workers) as pool:
                fn=lambda task:evaluate_task(task,c,K,fitted,coefs,oracles)
                for row,pl,ev,case in pool.map(fn,tasks):
                    rows.append(row); plans.extend(pl); events.extend(ev)
                    if case is not None: cases.append(case)
            # Incremental recovery files; no TEST artifacts are produced.
            save(pd.DataFrame(rows),"scenario_days_valid.csv")
    metrics=pd.DataFrame(rows); table=finalize_tables(metrics)
    save(pd.DataFrame(plans),"production_plans_valid.csv"); save(pd.DataFrame(cases),"rare_event_20210719.csv")
    ev=pd.DataFrame(events); save(ev,"tau_event_predictions_valid.csv")
    rel=[]; aps=[]
    for (method,K),g in ev.groupby(["method","K"]):
        # Average repeat-seed event probabilities before evaluating reliability.
        g=g.groupby(["day","hour"])[["probability","event"]].mean().reset_index()
        h=reliability(g.probability,g.event); h["method"]=method; h["K"]=K; rel.append(h)
        rat=metrics[(metrics.method==method)&(metrics.K==K)].groupby("day")[["ratchet_probability","ratchet_event"]].mean()
        aps.append(dict(method=method,K=K,tau_pr_auc=pr_auc(g.event,g.probability),
                        ratchet_pr_auc=pr_auc(rat.ratchet_event,rat.ratchet_probability)))
        h=reliability(rat.ratchet_probability,rat.ratchet_event); h["method"]=method; h["K"]=K
        save(h,f"reliability_ratchet_{method}_K{K}.csv")
    save(pd.concat(rel),"reliability_tau_valid.csv"); save(pd.DataFrame(aps),"event_pr_auc_valid.csv")
    chosen=select(table); selected=next(c for c in configs if c.name==chosen)
    settings=dict(phase=1,forecaster="regime_ens",forecaster_final=False,method=selected.method,
                  estimator=selected.estimator,rank=selected.rank,K=30,seeds=list(SEEDS),alpha=.9,lam=0,
                  rank_active=selected.method.startswith("pca"),covariance_active=selected.method=="cholesky",
                  selected_on="VALID excluding 2021-07-19",test_evaluated=False,
                  rule="positive paired 95% saving CI + no worse Energy Score and ratchet Brier; otherwise empirical",
                  fingerprint=fingerprint,residual_n_days=len(pools["peak15"]),train_end=str(df.index[mask].max()),
                  source_hashes={f:hashlib.sha256((ROOT/f).read_bytes()).hexdigest() for f in
                                 ("experiments/joint_scenarios.py","src/joint_uncertainty.py","src/probabilistic_eval.py",
                                  "src/stochastic_milp.py","src/decision_eval.py","src/models.py","src/features.py","src/data.py")},
                  versions={p:importlib.metadata.version(p) for p in ("numpy","pandas","scikit-learn","lightgbm","catboost","ortools","pytest")})
    manifest.write_text(json.dumps(settings,indent=2),encoding="utf-8")
    (OUT/"selected_config.json").write_text(json.dumps(settings,indent=2),encoding="utf-8")
    plot_case(configs,days)
    print(table[(table.K==30)&(table.subset=="without_0719")].to_string(index=False),flush=True)
    print("SELECTED",chosen,"TEST NOT EVALUATED",flush=True)


def plot_case(configs,days):
    d=next(d for d in days if str(d.day.date())==CASE)
    fig,ax=plt.subplots(figsize=(10,6))
    table=pd.read_csv(OUT/"scenario_metrics_valid.csv")
    valid=table[(table.K==30)&(table.subset=="without_0719")]
    representatives={"empirical","independent"}
    for prefix in ("cholesky","pca_gaussian","pca_bootstrap"):
        representatives.add(valid[valid.method.str.startswith(prefix)].sort_values("regret_won").iloc[0].method)
    for c in configs:
        if c.name in representatives:
            x=np.load(OUT/f"scenarios_0719_{c.name}.npy")[:,d.demand_band>T.OFF].max(axis=1)
            ax.step(np.sort(x),np.arange(1,len(x)+1)/len(x),where="post",label=c.name)
    ax.axvline(d.floor,color="black",linestyle="--",label="Past-only ratchet floor")
    ax.set(xlabel="Billable daily max (kW)",ylabel="Empirical CDF",title="2021-07-19, TRAIN-fitted scenarios (K=30, seed=0)")
    ax.legend(); fig.tight_layout(); fig.savefig(OUT/"fig_0719_scenarios.png"); plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(); parser.add_argument("--reuse-residuals",action="store_true")
    parser.add_argument("--workers",type=int,default=4)
    parser.add_argument("--summarize-only",action="store_true",help="Derive tables from existing VALID decisions; no fitting")
    args=parser.parse_args()
    if args.summarize_only:
        finalize_tables(pd.read_csv(OUT/"scenario_days_valid.csv"))
    else:
        run(args.reuse_residuals,args.workers)
