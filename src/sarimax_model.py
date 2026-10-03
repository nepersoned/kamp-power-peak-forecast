"""Day-ahead seasonal state-space baseline; copied labels are missing observations.

Parameters fit only before the fold. Between forecast days the state may ingest
observations through D-1, never within the 24-hour forecast horizon.
"""
import numpy as np
import pandas as pd
from statsmodels.tsa.statespace.sarimax import SARIMAX

EXOG = ["log_prod", "plan_on_day", "hour_sin", "hour_cos", "is_holiday", "temp", "humid"]


class SARIMAXDaily:
    def __init__(self, maxiter=50):
        self.maxiter=maxiter

    def exog(self,X):
        z=X[EXOG].copy()
        z["week_sin"]=np.sin(2*np.pi*(X.index.dayofweek+X.index.hour/24)/7)
        z["week_cos"]=np.cos(2*np.pi*(X.index.dayofweek+X.index.hour/24)/7)
        return z

    def fit(self,X,y,hour=None,copy=None):
        if not X.index.equals(y.index) or not X.index.is_monotonic_increasing:
            raise ValueError("SARIMAX requires aligned chronological rows")
        idx=pd.date_range(X.index.min(),X.index.max(),freq="h")
        z=self.exog(X).reindex(idx)
        self.fill=z.median().fillna(0); self.scale=z.std().replace(0,1).fillna(1)
        ex=(z.fillna(self.fill)-self.fill)/self.scale
        ex["intercept"]=1.
        endog=y.astype(float).copy()
        if copy is not None: endog.loc[np.asarray(copy,bool)]=np.nan
        endog=endog.reindex(idx)
        self.n_observed=int(endog.notna().sum())
        self.result=SARIMAX(endog,exog=ex,order=(1,0,0),seasonal_order=(1,0,0,24),
                            trend="n",enforce_stationarity=True,enforce_invertibility=True).fit(
                                disp=False,maxiter=self.maxiter,cov_type="none")
        self.training_end=idx.max(); self.converged=bool(self.result.mle_retvals["converged"])
        self.forecasted_end=None
        return self

    def forecast_day(self,Xday):
        if len(Xday)!=24 or Xday.index.min()!=self.training_end+pd.Timedelta(hours=1):
            raise ValueError("Forecast must be next complete 24-hour day")
        ex=(self.exog(Xday).fillna(self.fill)-self.fill)/self.scale
        ex["intercept"]=1.
        self.forecasted_end=Xday.index.max()
        return np.asarray(self.result.forecast(24,exog=ex),float)

    def observe_day(self,Xday,yday,copy=None):
        if Xday.index.min()!=self.training_end+pd.Timedelta(hours=1):
            raise ValueError("Non-contiguous state update")
        if self.forecasted_end!=Xday.index.max():
            raise ValueError("Make the full day-ahead forecast before observing that day")
        ex=(self.exog(Xday).fillna(self.fill)-self.fill)/self.scale
        ex["intercept"]=1.
        y=yday.astype(float).copy()
        if copy is not None: y.loc[np.asarray(copy,bool)]=np.nan
        # extend filters just the new observations; parameters are not refitted.
        self.result=self.result.extend(y,exog=ex)
        self.training_end=Xday.index.max()
        self.forecasted_end=None
