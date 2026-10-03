"""Common point/distribution CSV schema for any fit/predict forecaster."""
import numpy as np
import pandas as pd


def forecast_frame(index, target, point, model, scenarios=None):
    point = np.asarray(point, float)
    if len(index) != len(point) or not np.isfinite(point).all():
        raise ValueError("Invalid point forecast")
    if scenarios is not None:
        scenarios = np.asarray(scenarios, float)
        if scenarios.ndim != 2 or scenarios.shape[1] != len(point) or not len(scenarios) or not np.isfinite(scenarios).all():
            raise ValueError("Invalid scenario distribution")
    q = np.repeat(point[None, :], 3, axis=0) if scenarios is None else np.quantile(scenarios, [.1, .5, .9], axis=0)
    return pd.DataFrame(dict(timestamp=index, target=target, point=point,
                             q10=q[0], q50=q[1], q90=q[2], model=model))
