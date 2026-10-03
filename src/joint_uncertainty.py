"""Past-only daily errors and interchangeable 24-hour scenario distributions.

Empirical full-path bootstrap already preserves within-day dependence. PCA is
applied only to residual curves, never to forecasting input features.
"""
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf, OAS

START, VALID_START, TEST_START = "2021-01-08", "2021-07-16", "2021-08-16"
TARGETS = ("peak15", "power")


def training_mask(df, X, cutoff):
    """Existing lag/outage rules; unavailable production plans are excluded."""
    return ((df.index >= START) & (df.index < pd.Timestamp(cutoff))
            & ~df.outage.to_numpy() & ~df.plan_missing.to_numpy()
            & X.power_lag168.notna().to_numpy())


def eligible_days(df, X, start, end):
    rows = []
    for day in pd.date_range(start, end, inclusive="left"):
        idx = pd.date_range(day, periods=24, freq="h")
        reason = ""
        if not idx.isin(df.index).all():
            reason = "incomplete"
        else:
            s = df.loc[idx]
            if s.is_copy.any():
                reason = "copy"
            elif s.outage.any():
                reason = "outage"
            elif s.plan_missing.any():
                reason = "missing_plan"
            elif X.loc[idx, "power_lag168"].isna().any():
                reason = "missing_lag"
            elif s["prod"].sum() <= 0:
                reason = "shutdown"
        rows.append(dict(forecast_date=day, reason=reason))
    return pd.DataFrame(rows)


def rolling_residuals(df, X, factory, min_train_days=14, block_days=14):
    """Expanding origins; every target day is strictly later than its fit rows.

    Block fitting reduces cost without using future labels. Features must obey
    the repository's day-ahead lag rules. Hyperparameters remain the legacy
    VALID-tuned configuration (retrospective limitation, documented separately).
    """
    audit = eligible_days(df, X, START, VALID_START)
    rows, models, origin, train_end = [], None, None, None
    for i, row in audit.iterrows():
        day = row.forecast_date
        if row.reason:
            continue
        if origin is None or (day - origin).days >= block_days:
            mask = training_mask(df, X, day)
            n = df.index[mask & ~df.is_copy.to_numpy()].normalize().nunique()
            if n < min_train_days:
                audit.loc[i, "reason"] = "rolling_warmup"
                continue
            models = {t: factory().fit(X[mask], df.loc[mask, t], df.loc[mask, "hour"],
                                     df.loc[mask, "is_copy"]) for t in TARGETS}
            origin, train_end = day, df.index[mask].max()
        idx = pd.date_range(day, periods=24, freq="h")
        assert train_end < day
        for target, model in models.items():
            pred = model.predict(X.loc[idx], df.loc[idx, "hour"])
            for hour, (ts, forecast) in enumerate(zip(idx, pred)):
                actual = float(df.loc[ts, target])
                rows.append(dict(forecast_date=day, training_end_date=train_end,
                                 target=target, hour=hour, actual=actual, forecast=forecast,
                                 residual=actual - forecast, regime="operating", is_copy=False))
    return pd.DataFrame(rows), audit


def residual_matrix(provenance, target, cutoff=VALID_START):
    p = provenance.loc[provenance.target == target].copy()
    forecast = pd.to_datetime(p.forecast_date)
    end = pd.to_datetime(p.training_end_date)
    if not (end < forecast).all() or not (forecast < pd.Timestamp(cutoff)).all():
        raise ValueError("Future/held-out residual in training pool")
    if p.is_copy.any() or not (p.regime == "operating").all():
        raise ValueError("Only original operating-day residuals are eligible")
    matrix = p.pivot(index="forecast_date", columns="hour", values="residual")
    if list(matrix.columns) != list(range(24)) or not np.isfinite(matrix).all().all():
        raise ValueError("Residuals must contain complete finite 24-hour paths")
    return matrix


@dataclass(frozen=True)
class ScenarioConfig:
    method: str = "empirical"
    estimator: str = "empirical"
    rank: int = 4

    @property
    def name(self):
        if self.method == "cholesky":
            return f"cholesky_{self.estimator}"
        if self.method.startswith("pca"):
            return f"{self.method}_r{self.rank}"
        return self.method


class JointResidualModel:
    def __init__(self, config=ScenarioConfig()):
        self.config = config

    def fit(self, residuals):
        E = np.asarray(residuals, dtype=float)
        if E.ndim != 2 or E.shape[1] != 24 or len(E) < 2 or not np.isfinite(E).all():
            raise ValueError("Need at least two complete 24-hour residuals")
        self.E, self.mean = E.copy(), E.mean(axis=0)
        centered = E - self.mean
        _, s, vt = np.linalg.svd(centered, full_matrices=False)
        self.loadings, self.eigenvalues = vt, s ** 2 / (len(E) - 1)
        self.scores = centered @ vt.T
        if self.config.method.startswith("pca") and not 1 <= self.config.rank <= len(vt):
            raise ValueError("PCA rank exceeds available components")
        estimator = self.config.estimator
        if estimator == "empirical":
            cov = np.cov(E, rowvar=False)
        elif estimator == "ledoit_wolf":
            cov = LedoitWolf().fit(E).covariance_
        elif estimator == "oas":
            cov = OAS().fit(E).covariance_
        else:
            raise ValueError(f"Unknown covariance estimator: {estimator}")
        cov = (cov + cov.T) / 2
        eig, vec = np.linalg.eigh(cov)
        self.raw_min_eigenvalue = float(eig.min())
        self.eigenvalue_floor = max(float(eig.max()) * 1e-8, 1e-8)
        self.covariance = (vec * np.maximum(eig, self.eigenvalue_floor)) @ vec.T
        self.cholesky = np.linalg.cholesky(self.covariance)
        return self

    def reconstruct(self, scores, rank=None):
        r = self.config.rank if rank is None else rank
        return self.mean + np.asarray(scores)[..., :r] @ self.loadings[:r]

    def sample(self, K, seed=0):
        if K <= 0:
            raise ValueError("K must be positive")
        rng = np.random.default_rng(seed)
        method = self.config.method
        if method == "empirical":
            # Exact K paths, including when pool is smaller than K.
            return self.E[rng.choice(len(self.E), K, replace=len(self.E) < K)]
        if method == "independent":
            return np.column_stack([rng.choice(self.E[:, h], K) for h in range(24)])
        if method == "cholesky":
            return self.mean + rng.normal(size=(K, 24)) @ self.cholesky.T
        r = self.config.rank
        if method == "pca_gaussian":
            scores = rng.normal(size=(K, r)) * np.sqrt(self.eigenvalues[:r])
        elif method == "pca_bootstrap":
            scores = self.scores[rng.choice(len(self.E), K, replace=len(self.E) < K), :r]
        else:
            raise ValueError(f"Unknown scenario method: {method}")
        return self.reconstruct(scores)

    def scenarios(self, point, K, seed=0):
        point = np.asarray(point, dtype=float)
        if point.shape != (24,) or not np.isfinite(point).all():
            raise ValueError("Point forecast must be a finite 24-hour vector")
        return np.maximum(0, point + self.sample(K, seed))
