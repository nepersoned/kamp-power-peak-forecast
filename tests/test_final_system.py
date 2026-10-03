import numpy as np
import pandas as pd
import pytest
from src.final_system import fit_mask, eligibility, rolling_residual_pool, validate_pool, TEST_START, forecast_queries


def fixture_frames():
    idx=pd.date_range("2021-06-01",periods=6*24,freq="h")
    df=pd.DataFrame(dict(power=100.,peak15=110.,hour=idx.hour,is_copy=False,outage=False,plan_missing=False),index=idx)
    X=pd.DataFrame(dict(power_lag168=90.,plan_on_day=1.),index=idx)
    df.loc["2021-06-03","is_copy"]=True
    df.loc["2021-06-04","plan_missing"]=True;X.loc["2021-06-04","plan_on_day"]=np.nan
    X.loc["2021-06-05","plan_on_day"]=0.
    return df,X


def test_canonical_masks_are_explicitly_different():
    df,X=fixture_frames()
    mask=fit_mask(df,X,"2021-06-06")
    assert mask[df.index.get_loc("2021-06-04 12:00")]
    a=eligibility(df,X,"residual_pool","2021-06-01","2021-06-07").set_index("date")
    assert a.loc["2021-06-03","excluded_reason"]=="copy_day"
    assert a.loc["2021-06-04","excluded_reason"]=="plan_missing"
    assert a.loc["2021-06-05","excluded_reason"]=="shutdown"
    with pytest.raises(ValueError): fit_mask(df,X,"2021-08-17")
    df.loc["2021-06-02 01:00","outage"]=True
    assert not eligibility(df,X,"decision_evaluation","2021-06-02","2021-06-03").eligible.iloc[0]


def test_forecast_queries_include_future_outage_hours():
    index=pd.date_range(TEST_START,periods=48,freq="h")
    df=pd.DataFrame({"outage":False,"power":100.},index=index)
    df.iloc[3:10]=[True,0.]
    queries=list(forecast_queries(df.index))
    assert all(len(q)==24 for q in queries)
    assert all(t in queries[0] for t in df.index[3:10])
    with pytest.raises(ValueError):list(forecast_queries(index.delete(4)))


def test_rolling_pool_fit_uses_only_past_and_never_test():
    df,X=fixture_frames();fits=[]
    class Model:
        def fit(self,x,y,hour,copy):
            self.end=x.index.max();fits.append(x.index);return self
        def predict(self,x,hour):
            assert self.end<x.index.min();return np.full(len(x),95.)
    p,a=rolling_residual_pool(df,X,Model,"stub",min_train_days=1,block_days=2)
    validate_pool(p,list(X),X)
    assert set(pd.to_datetime(p.forecast_date).dt.day)=={2,6}
    assert all(len(g)==24 for _,g in p.groupby(["target","forecast_date"]))
    assert any(pd.Timestamp("2021-06-04 12:00") in index for index in fits)
    assert (pd.to_datetime(p.training_end_date)<pd.to_datetime(p.forecast_date)).all()
    held=df.copy();held.index=pd.date_range(TEST_START,periods=len(df),freq="h")
    with pytest.raises(ValueError,match="truncated"):rolling_residual_pool(held,X,Model,"stub")


def test_pool_rejects_future_copy_missing_incomplete_and_feature_change():
    df,X=fixture_frames()
    class Model:
        def fit(self,*args):return self
        def predict(self,x,h):return np.zeros(len(x))
    p,_=rolling_residual_pool(df,X,Model,"stub",min_train_days=1)
    for field,value in [("training_end_date",pd.Timestamp("2021-09-01")),("is_copy",True),("plan_missing",True)]:
        bad=p.copy();bad.loc[0,field]=value
        with pytest.raises(ValueError):validate_pool(bad,list(X),X)
    with pytest.raises(ValueError,match="Incomplete"):validate_pool(p.iloc[1:],list(X),X)
    with pytest.raises(ValueError,match="schema"):validate_pool(p,["workers"],X)


def test_fixed_pool_and_scenarios_do_not_depend_on_test_actual():
    from src.joint_uncertainty import JointResidualModel
    E=np.arange(40*24,dtype=float).reshape(40,24)/100
    m=JointResidualModel().fit(E);original=m.E.copy();point=np.ones(24)*100
    first=m.scenarios(point,30,228)
    # Observed TEST errors never enter a fitted pool; changed scoring labels
    # leave the scenario and optimizer inputs identical.
    from src.probabilistic_eval import marginal_metrics
    a=marginal_metrics(point,first);b=marginal_metrics(point+50,first)
    assert a["crps"]!=b["crps"]
    assert np.array_equal(original,m.E)
    assert np.array_equal(first,m.scenarios(point,30,228))
    assert first.shape==(30,24)


def test_decision_plan_uses_forecasts_and_not_heldout_actual():
    from types import SimpleNamespace
    from src.stochastic_milp import solve_stochastic
    from src.decision_eval import true_cost
    prod=np.zeros(24);prod[8:11]=10.
    cf=dict(gain=np.ones(24)*2,prev=0.,next=0.,start=0.,beta=.5)
    coefs={t:cf for t in ("power","peak15")}
    d=SimpleNamespace(prod=prod,rate=np.linspace(200,50,24),fc_pw=np.ones(24)*30,
         demand_band=np.ones(24,dtype=int),floor=100.,labor=np.ones(24),act_pw=np.ones(24)*30,act_pk=np.ones(24)*40)
    scenarios=np.ones((30,24))*40
    plan,info=solve_stochastic(d,coefs,scenarios,lam=0,return_info=True)
    baseline_cost=true_cost(d,d.act_pw,d.act_pk,plan,coefs)
    d.act_pw[:]=50.;d.act_pk[:]=250.
    changed,info2=solve_stochastic(d,coefs,scenarios,lam=0,return_info=True)
    assert info["status"]==info2["status"]=="optimal"
    assert np.allclose(plan,changed)
    assert np.isclose(plan.sum(),prod.sum())
    assert (plan<=prod.max()+1e-5).all()
    assert true_cost(d,d.act_pw,d.act_pk,plan,coefs)["total"]!=baseline_cost["total"]
