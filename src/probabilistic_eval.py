"""Finite-ensemble marginal, event, and multivariate proper scores (kW)."""
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score


def _inputs(actual, scenarios):
    y, x = np.asarray(actual, float), np.asarray(scenarios, float)
    if y.ndim != 1 or x.ndim != 2 or x.shape[1] != len(y) or not len(x):
        raise ValueError("Expected actual (H,) and scenarios (K,H)")
    if not np.isfinite(y).all() or not np.isfinite(x).all():
        raise ValueError("Non-finite scoring input")
    return y, x


def crps(actual, scenarios):
    """CRPS of empirical predictive distribution, including self-pairs."""
    y, x = _inputs(actual, scenarios)
    return float(np.mean(np.abs(x - y)) - .5 * np.mean(np.abs(x[:, None] - x[None, :])))


def energy_score(actual, scenarios):
    y, x = _inputs(actual, scenarios)
    return float(np.linalg.norm(x - y, axis=1).mean()
                 - .5 * np.linalg.norm(x[:, None] - x[None, :], axis=2).mean())


def variogram_score(actual, scenarios, p=.5):
    """Uniform weights over unordered hour pairs; normalized by pair count."""
    y, x = _inputs(actual, scenarios)
    i, j = np.triu_indices(len(y), 1)
    obs = np.abs(y[i] - y[j]) ** p
    pred = (np.abs(x[:, i] - x[:, j]) ** p).mean(axis=0)
    return float(np.mean((obs - pred) ** 2))


def marginal_metrics(actual, scenarios):
    y, x = _inputs(actual, scenarios)
    q = np.quantile(x, [.1, .5, .9], axis=0)
    pinball = [np.maximum(a * (y - v), (a - 1) * (y - v)).mean()
               for a, v in zip((.1, .5, .9), q)]
    return dict(crps=crps(y, x), energy_score=energy_score(y, x),
                variogram_score=variogram_score(y, x),
                coverage=float(((y >= q[0]) & (y <= q[2])).mean()),
                interval_width=float((q[2] - q[0]).mean()), pinball=float(np.mean(pinball)))


def event_metrics(actual, scenarios, floor, demand_mask, tau=190):
    y, x = _inputs(actual, scenarios)
    mask = np.asarray(demand_mask, bool)
    tau_probability = (x >= tau).mean(axis=0)
    ratchet_probability = float((x[:, mask].max(axis=1, initial=0) > floor).mean())
    event = int(y[mask].max(initial=0) > floor)
    return dict(tau_brier=float(((tau_probability - (y >= tau)) ** 2).mean()),
                ratchet_probability=ratchet_probability, ratchet_event=event,
                ratchet_brier=(ratchet_probability - event) ** 2), tau_probability


def reliability(probability, event, bins=5):
    d = pd.DataFrame(dict(probability=probability, event=event))
    d["bin"] = np.minimum((d.probability * bins).astype(int), bins - 1)
    return d.groupby("bin").agg(n=("event", "size"), predicted=("probability", "mean"),
                               observed=("event", "mean")).reset_index()


def pr_auc(event, probability):
    # AP (stepwise PR-AUC); undefined if there are no positives.
    return float(average_precision_score(event, probability)) if np.any(event) else np.nan


def paired_bootstrap(delta, seed=42, B=2000):
    d = np.asarray(delta, float)
    if not len(d) or not np.isfinite(d).all():
        raise ValueError("Need finite paired daily differences")
    rng = np.random.default_rng(seed)
    means = d[rng.integers(len(d), size=(B, len(d)))].mean(axis=1)
    return tuple(np.quantile(means, [.025, .975]))
