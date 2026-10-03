import numpy as np
import pandas as pd
import pytest
from experiments.forecast_model_search import masks,primary,blend,paired_ci,FOLDS,data
from src.forecast_adapters import WeightedForecaster


def predictions(model,offset=0):
    rows=[]
    for fold,(a,b) in enumerate(FOLDS,1):
        for day in pd.date_range(a,b,inclusive="left"):
            for target in ("power","peak15"):
                for h in range(24):
                    rows.append(dict(timestamp=day+pd.Timedelta(hours=h),date=day,fold=fold,target=target,
                                     actual=10.,prediction=10.+offset,model=model,operating=True,peak_hour=8<=h<=11,
                                     fit_predict_seconds=1.,is_copy=False,transition=False))
    return pd.DataFrame(rows).sort_values(["fold","target","timestamp"]).reset_index(drop=True)


def test_primary_equal_fold_target_weights():
    p=predictions("a")
    p["prediction"]=p.actual+np.where(p.fold==1,1.,3.)
    assert primary(p)==2 # unequal duration must NOT reweight the primary objective


def test_blend_and_bootstrap_no_meta_training():
    a,b=predictions("a",1),predictions("b",-1)
    q=blend(a,b,{"power":.5,"peak15":.5},"blend")
    assert primary(q)==0
    ci=paired_ci(q,a)
    assert ci["mae_diff"]==ci["ci_low"]==ci["ci_high"]==-1
    # Changing actual values cannot alter the frozen blend predictions.
    a.actual+=1000; b.actual-=1000
    np.testing.assert_array_equal(q.prediction,blend(a,b,{"power":.5,"peak15":.5},"blend").prediction)


def test_protocol_no_test_and_no_workers():
    df,X=data(); assert df.index.max()<pd.Timestamp("2021-08-16")
    assert "workers" not in X
    for a,b in FOLDS:
        tr,ev=masks(df,X,a,b)
        assert df.index[tr].max()<df.index[ev].min()
        assert df.index[ev].max()<pd.Timestamp("2021-08-16")
    with pytest.raises(ValueError): masks(df,X,"2021-08-16","2021-09-15")


def test_adapter_weight_validation():
    with pytest.raises(ValueError): WeightedForecaster(["a"]*4,{})
    idx=pd.date_range("2021-01-08",periods=24,freq="h")
    X=pd.DataFrame({"power_lag168":1.},index=idx)
    y=pd.Series(np.ones(24),index=idx,name="power")
    with pytest.raises(ValueError): WeightedForecaster(["naive_168h"],{"power":[.5]}).fit(X,y,pd.Series(idx.hour,index=idx))


def test_sarimax_no_same_day_feedback(monkeypatch):
    import src.sarimax_model as mod
    class Result:
        mle_retvals={"converged":True}
        def __init__(self,y): self.value=float(np.nanmean(y))
        def forecast(self,K,exog): return np.ones(K)*self.value
        def extend(self,y,exog): return Result(y)
    class Fake:
        def __init__(self,y,exog,**kw):
            self.y=y; assert kw["seasonal_order"][-1]==24
            assert y.isna().sum()==24 # copied label day is missing, not repeated
        def fit(self,**kw): return Result(self.y)
    monkeypatch.setattr(mod,"SARIMAX",Fake)
    idx=pd.date_range("2021-01-08",periods=72,freq="h")
    X=pd.DataFrame({c:np.ones(72) for c in mod.EXOG},index=idx)
    y=pd.Series(np.ones(72)*10,index=idx)
    copies=pd.Series(False,index=idx); copies.iloc[:24]=True
    m=mod.SARIMAXDaily().fit(X.iloc[:48],y.iloc[:48],copy=copies.iloc[:48])
    with pytest.raises(ValueError): m.observe_day(X.iloc[48:],y.iloc[48:])
    a=m.forecast_day(X.iloc[48:]); future=y.iloc[48:]*100
    np.testing.assert_array_equal(a,np.ones(24)*10)
    m.observe_day(X.iloc[48:],future)
    assert m.training_end==idx[-1]


def test_tabpfn_context_is_past_original_only(monkeypatch):
    # Optional dependency-free stub tests the data route, not weight quality.
    import sys,types
    from src.tabpfn_model import TabPFNModel
    monkeypatch.setattr("src.tabpfn_model.resolve_checkpoint",lambda:"stub.ckpt")
    class Model:
        @classmethod
        def create_default_for_version(cls,v,**kw): return cls()
        def fit(self,X,y): self.rows=X.index; return self
        def predict(self,X,**kw): return np.ones(len(X))
    monkeypatch.setitem(sys.modules,"torch",types.SimpleNamespace(set_num_threads=lambda _:None))
    monkeypatch.setitem(sys.modules,"tabpfn",types.SimpleNamespace(TabPFNRegressor=Model))
    monkeypatch.setitem(sys.modules,"tabpfn.constants",types.SimpleNamespace(ModelVersion=lambda x:x))
    idx=pd.date_range("2021-06-01",periods=28*24,freq="h")
    X=pd.DataFrame({"power_lag24":1.},index=idx); y=pd.Series(1.,index=idx)
    copy=pd.Series(False,index=idx); copy.iloc[-24:]=True
    m=TabPFNModel(context_days=14).fit(X,y,copy=copy)
    assert m.n_samples==13*24
    assert m.model.rows.min()>=pd.Timestamp("2021-06-15")
    with pytest.raises(ValueError): m.predict(X.iloc[-48:-24])


def test_checkpoint_hash_guard(tmp_path,monkeypatch):
    import sys,types,hashlib
    import src.tabpfn_model as mod
    p=tmp_path/"weight.ckpt"; p.write_bytes(b"test weights")
    monkeypatch.setitem(sys.modules,"tabpfn.model_loading",types.SimpleNamespace(get_cache_dir=lambda:tmp_path))
    monkeypatch.setenv("KAMP_TABPFN_MODEL_PATH",str(p))
    with pytest.raises(ValueError,match="checksum"): mod.resolve_checkpoint()
    monkeypatch.setattr(mod,"WEIGHT_SHA256",hashlib.sha256(p.read_bytes()).hexdigest())
    assert mod.resolve_checkpoint()==p
