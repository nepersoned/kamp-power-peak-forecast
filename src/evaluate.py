import numpy as np
import pandas as pd


def reg_metrics(y, p):
    y, p = np.asarray(y, float), np.asarray(p, float)
    e = y - p
    return dict(MAE=np.mean(np.abs(e)), RMSE=np.sqrt(np.mean(e ** 2)),
                WAPE=np.sum(np.abs(e)) / np.sum(np.abs(y)) * 100, Bias=np.mean(p - y))


def metrics_table(y, preds, mask=None):
    rows = {}
    for name, p in preds.items():
        yy, pp = (y, p) if mask is None else (y[mask], p[mask])
        rows[name] = reg_metrics(yy, pp)
    return pd.DataFrame(rows).T.round(2)


def daily_peak_metrics(y, p, index):
    """일 최대수요(요금 기준) 예측 오차."""
    d = pd.DataFrame({"y": np.asarray(y), "p": np.asarray(p)}, index=index).resample("D").max().dropna()
    return reg_metrics(d["y"], d["p"])


def error_by(frame, by, err="abs_err"):
    g = frame.groupby(by, observed=True)[err]
    return pd.DataFrame({"n": g.size(), "MAE": g.mean(), "bias": frame.groupby(by, observed=True)["err"].mean()}).round(2)
