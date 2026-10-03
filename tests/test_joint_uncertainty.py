import numpy as np
import pandas as pd
import pytest

from src.joint_uncertainty import (JointResidualModel, ScenarioConfig, rolling_residuals,
                                   residual_matrix, training_mask)
from src.probabilistic_eval import crps, energy_score, variogram_score, event_metrics, paired_bootstrap
from src.forecast_interface import forecast_frame
from src.data import load


@pytest.fixture
def errors():
    rng=np.random.default_rng(42)
    return rng.normal(size=(35,3)) @ rng.normal(size=(3,24))


@pytest.mark.parametrize("method",["empirical","independent","cholesky","pca_gaussian","pca_bootstrap"])
def test_scenario_shape_seed(errors,method):
    m=JointResidualModel(ScenarioConfig(method,rank=3)).fit(errors)
    a=m.scenarios(np.ones(24)*10,50,7)
    assert a.shape==(50,24)
    assert np.isfinite(a).all() and (a>=0).all()
    np.testing.assert_array_equal(a,m.scenarios(np.ones(24)*10,50,7))
    assert not np.array_equal(a,m.scenarios(np.ones(24)*10,50,8))


@pytest.mark.parametrize("estimator",["empirical","ledoit_wolf","oas"])
def test_covariance_psd_stability(errors,estimator):
    # Exactly low-rank inputs deliberately stress Cholesky.
    m=JointResidualModel(ScenarioConfig("cholesky",estimator)).fit(errors)
    np.testing.assert_allclose(m.covariance,m.covariance.T,atol=1e-10)
    assert np.linalg.eigvalsh(m.covariance).min()>0
    np.testing.assert_allclose(m.cholesky@m.cholesky.T,m.covariance,atol=1e-8)


def test_pca_reconstruction(errors):
    m=JointResidualModel(ScenarioConfig("pca_bootstrap",rank=3)).fit(errors)
    np.testing.assert_allclose(m.reconstruct(m.scores),errors,atol=1e-10)
    assert np.mean((m.reconstruct(m.scores,2)-errors)**2)>1e-4


def test_score_sanity_and_dependence():
    y=np.zeros(24); perfect=np.zeros((10,24)); wrong=np.ones((10,24))
    assert crps(y,perfect)==energy_score(y,perfect)==variogram_score(y,perfect)==0
    assert crps(y,wrong)==1
    assert energy_score(y,wrong)==pytest.approx(np.sqrt(24))
    # Same hourly marginals but different hour dependence.
    a=np.array([[0.,0.],[2.,2.]])
    b=np.array([[0.,2.],[2.,0.]])
    assert crps(np.zeros(2),a)==crps(np.zeros(2),b)
    assert variogram_score(np.zeros(2),a)<variogram_score(np.zeros(2),b)
    assert energy_score(np.zeros(2),a)!=energy_score(np.zeros(2),b)
    em,_=event_metrics(np.ones(24)*200,np.ones((5,24))*200,190,np.ones(24,bool))
    assert em["ratchet_brier"]==em["tau_brier"]==0
    assert paired_bootstrap(np.ones(10))==(1.,1.)
    em,_=event_metrics(np.ones(24)*200,np.ones((5,24))*200,190,np.zeros(24,bool))
    assert em["ratchet_probability"]==em["ratchet_event"]==em["ratchet_brier"]==0


class PastMean:
    def fit(self,X,y,hour,copy):
        self.end=X.index.max(); self.value=y.mean()
        return self
    def predict(self,X,hour):
        assert self.end<X.index.min()
        return np.repeat(self.value,len(X))


def fake_data():
    idx=pd.date_range("2021-01-01", "2021-08-15 23:00",freq="h")
    df=pd.DataFrame(dict(power=np.arange(len(idx),dtype=float)+1,peak15=np.arange(len(idx),dtype=float)+2,
                         hour=idx.hour,prod=10.,is_copy=False,outage=False,plan_missing=False),index=idx)
    X=pd.DataFrame(dict(power_lag168=1.,plan_on_day=1.),index=idx)
    return df,X


def test_rolling_future_leakage_and_provenance():
    df,X=fake_data()
    p,_=rolling_residuals(df,X,PastMean,min_train_days=3,block_days=14)
    assert (pd.to_datetime(p.training_end_date)<pd.to_datetime(p.forecast_date)).all()
    assert (p.forecast_date<pd.Timestamp("2021-07-16")).all()
    E=residual_matrix(p,"peak15"); assert E.shape[1]==24
    # Alter all future labels: earlier residuals must be bitwise unchanged.
    altered=df.copy(); cutoff=pd.Timestamp("2021-05-01")
    altered.loc[altered.index>=cutoff,["power","peak15"]]+=100000
    q,_=rolling_residuals(altered,X,PastMean,min_train_days=3,block_days=14)
    pd.testing.assert_frame_equal(p[p.forecast_date<cutoff].reset_index(drop=True),
                                  q[q.forecast_date<cutoff].reset_index(drop=True))
    p.loc[0,"training_end_date"]=p.loc[0,"forecast_date"]
    with pytest.raises(ValueError): residual_matrix(p,"peak15")
    p.loc[0,"training_end_date"]=pd.Timestamp("2021-07-15 23:00")
    p.loc[0,"forecast_date"]=pd.Timestamp("2021-07-16")
    with pytest.raises(ValueError): residual_matrix(p,"peak15")


def test_valid_test_separation_and_cutoff():
    df,X=fake_data()
    mask=training_mask(df,X,"2021-07-16")
    assert df.index[mask].max()<pd.Timestamp("2021-07-16")
    valid_data=load(end="2021-08-16")
    assert valid_data.index.max()<pd.Timestamp("2021-08-16")
    from experiments.joint_scenarios import select
    t=pd.DataFrame([dict(method="empirical",K=30,subset="without_0719",paired_ci_low=0,energy_score=5,ratchet_brier=.1,regret_won=10),
                    dict(method="cholesky_oas",K=30,subset="without_0719",paired_ci_low=-1,energy_score=4,ratchet_brier=.1,regret_won=1),
                    dict(method="cholesky_oas",K=30,subset="all",paired_ci_low=100,energy_score=1,ratchet_brier=0,regret_won=0)])
    assert select(t)=="empirical" # 7/19 cannot override the selected configuration.


def test_distribution_interface():
    f=forecast_frame(pd.date_range("2021-07-16",periods=24,freq="h"),"peak15",np.ones(24),"dummy")
    assert list(f)==["timestamp","target","point","q10","q50","q90","model"]
    assert (f.point==f.q50).all()
    with pytest.raises(ValueError): forecast_frame(f.timestamp,"peak15",np.ones(24),"dummy",np.ones((4,23)))


def test_external_point_and_legacy_q50(tmp_path,monkeypatch):
    from src import decision_eval
    monkeypatch.setattr(decision_eval,"OUT",tmp_path)
    idx=pd.date_range("2021-07-16",periods=24,freq="h")
    f=pd.concat([forecast_frame(idx,t,np.ones(24)*10,"dummy") for t in ("power","peak15")])
    f["q50"]=20.
    f.to_csv(tmp_path/"forecast_dist_new.csv",index=False)
    f.drop(columns="point").to_csv(tmp_path/"forecast_dist_legacy.csv",index=False)
    forecasts=decision_eval.load_external(idx)
    assert (forecasts["new"][0]==10).all()
    assert (forecasts["legacy"][0]==20).all()


def test_solver_backward_compatible_and_scenario_independent():
    from types import SimpleNamespace
    from src.stochastic_milp import solve_stochastic
    d=SimpleNamespace(prod=np.zeros(24),rate=np.ones(24),fc_pw=np.ones(24)*10,
                      demand_band=np.ones(24),floor=20.)
    cf=dict(gain=np.zeros(24),prev=0.,next=0.,start=0.,beta=0.)
    coefs={"power":cf,"peak15":cf}
    a=solve_stochastic(d,coefs,np.ones((3,24))*10,lam=0)
    b,info=solve_stochastic(d,coefs,np.ones((3,24))*10,lam=0,return_info=True)
    np.testing.assert_array_equal(a,b); assert info["status"]=="optimal"
    with pytest.raises(ValueError): solve_stochastic(d,coefs,np.zeros((1,23)))
