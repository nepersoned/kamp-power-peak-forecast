"""Factory adapters and a <=3 component weighted forecasting model."""
import numpy as np
from .models import candidates,RegimeModel


def make_forecaster(name):
    old=candidates()
    if name in old: return old[name]()
    if "tabpfn" in name:
        from .tabpfn_model import TabPFNModel
        days=None if name.endswith("all") else int(name.rsplit("_",1)[1])
        model=TabPFNModel(context_days=days)
        return RegimeModel(on_model=model) if name.startswith("regime_") else model
    raise ValueError(f"Unknown forecaster {name}")


class WeightedForecaster:
    def __init__(self,components,weights_by_target):
        if not 1<=len(components)<=3: raise ValueError("At most three components")
        self.components=components; self.weights_by_target=weights_by_target

    def fit(self,X,y,hour,copy=None):
        self.weights=np.asarray(self.weights_by_target[y.name],float)
        if len(self.weights)!=len(self.components) or (self.weights<0).any() or not np.isclose(self.weights.sum(),1):
            raise ValueError("Invalid ensemble weights")
        self.fitted=[make_forecaster(n).fit(X,y,hour,copy) if w>0 else None
                     for n,w in zip(self.components,self.weights)]
        return self

    def predict(self,X,hour):
        return sum(w*m.predict(X,hour) for w,m in zip(self.weights,self.fitted) if m is not None)
