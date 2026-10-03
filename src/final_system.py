"""Frozen final-system eligibility and causal residual generation.

Training follows the selected Phase-2 protocol; uncertainty/decisions require
original complete operating days with known production plans. No online pool
update occurs during TEST. Existing Phase-1 selection and solvers are unchanged.
"""
import numpy as np
import pandas as pd
from .models import operating

START = "2021-01-08"
TEST_START = "2021-08-16"
END = "2021-09-15"


def forecast_queries(index, start=TEST_START, end=END):
    """Queries depend only on calendar, never future outage/target labels."""
    idx = index[(index >= pd.Timestamp(start)) & (index < pd.Timestamp(end))]
    for day in idx.normalize().unique():
        daily = idx[idx.normalize() == day]
        if len(daily) != 24: raise ValueError("Forecast query must cover complete 24h")
        yield daily


def fit_mask(df, X, cutoff):
    if pd.Timestamp(cutoff) > pd.Timestamp(TEST_START):
        raise ValueError("No TEST labels may enter fitting")
    return ((df.index >= START) & (df.index < pd.Timestamp(cutoff))
            & ~df.outage.to_numpy() & X.power_lag168.notna().to_numpy())


def eligibility(df, X, stage, start, end):
    if stage not in ("forecaster_training", "residual_pool", "decision_evaluation"):
        raise ValueError("Unknown eligibility stage")
    rows = []
    for day in pd.date_range(start, end, inclusive="left"):
        idx = pd.date_range(day, periods=24, freq="h")
        reasons = []
        complete = idx.isin(df.index).all()
        if not complete:
            reasons.append("incomplete_24h")
            n_rows = 0
            copied = missing = off = outage = False
        else:
            sub = df.loc[idx]
            copied, missing = bool(sub.is_copy.any()), bool(sub.plan_missing.any())
            off, outage = not operating(X.loc[idx]).any(), bool(sub.outage.any())
            if day < pd.Timestamp(START): reasons.append("warmup_start")
            if X.loc[idx, "power_lag168"].isna().any(): reasons.append("warmup_lag168")
            if stage == "forecaster_training":
                # Row-level outage exclusion; copy behavior is model-specific.
                n_rows = int(fit_mask(df.loc[idx], X.loc[idx], TEST_START).sum())
                if n_rows == 0 and outage: reasons.append("outage_all_rows")
                if day >= pd.Timestamp(TEST_START): reasons.append("heldout_test")
            else:
                n_rows = 24
                if copied: reasons.append("copy_day")
                if outage: reasons.append("outage")
                if missing: reasons.append("plan_missing")
                if off: reasons.append("shutdown")
                if stage == "residual_pool" and day >= pd.Timestamp(TEST_START):
                    reasons.append("heldout_test")
        rows.append(dict(stage=stage, date=day, eligible=not reasons,
                         excluded_reason=";".join(reasons), eligible_rows=n_rows if not reasons else 0,
                         complete_24h=complete, is_copy=copied, plan_missing=missing,
                         shutdown=off, outage=outage))
    return pd.DataFrame(rows)


def rolling_residual_pool(df, X, factory, name, min_train_days=14, block_days=14, on_origin=None):
    """Reproduce Phase-1 origin cadence with explicit Phase-2 fit mask.

    Hyperparameters are VALID-selected: these are past-only label predictions,
    not an independent evaluation of hyperparameter selection.
    """
    if df.index.max() >= pd.Timestamp(TEST_START):
        raise ValueError("Residual input must be truncated before TEST")
    audit = eligibility(df, X, "residual_pool", START, TEST_START)
    origin, records, targets = None, [], ("peak15", "power")
    pending = []

    def flush(days, training_origin):
        if not days: return
        mask = fit_mask(df, X, training_origin)
        first, last = df.index[mask].min(), df.index[mask].max()
        origin_rows = []
        for target in targets:
            print("RESIDUAL FIT", name, target, str(training_origin.date()), int(mask.sum()), flush=True)
            model = factory().fit(X[mask], df.loc[mask, target], df.loc[mask, "hour"], df.loc[mask, "is_copy"])
            for day in days:
                idx = pd.date_range(day, periods=24, freq="h")
                pred = model.predict(X.loc[idx], df.loc[idx, "hour"])
                assert last < day
                for h, value in enumerate(pred):
                    actual = float(df.loc[idx[h], target])
                    origin_rows.append(dict(forecast_date=day, training_start_date=first, training_end_date=last,
                        target=target, hour=h, actual=actual, prediction=float(value), forecast=float(value),
                        residual=actual-value, operating=True, regime="operating", is_copy=False,
                        plan_missing=False, origin_id=str(training_origin.date()), model=name))
            del model
        records.extend(origin_rows)
        if on_origin is not None: on_origin(pd.DataFrame(records))

    for i, row in audit.iterrows():
        if not row.eligible: continue
        day = row.date
        if origin is None or (day-origin).days >= block_days:
            mask = fit_mask(df, X, day)
            original_days = df.index[mask & ~df.is_copy.to_numpy()].normalize().nunique()
            if original_days < min_train_days:
                audit.loc[i, ["eligible", "excluded_reason", "eligible_rows"]] = [False, "rolling_warmup", 0]
                continue
            flush(pending, origin)
            pending, origin = [], day
        pending.append(day)
    flush(pending, origin)
    return pd.DataFrame(records), audit


def validate_pool(provenance, feature_list, X):
    if list(X.columns) != feature_list or "workers" in feature_list:
        raise ValueError("Frozen feature schema changed")
    p = provenance
    dates, last = pd.to_datetime(p.forecast_date), pd.to_datetime(p.training_end_date)
    if not (last < dates).all() or not (dates < pd.Timestamp(TEST_START)).all():
        raise ValueError("Future/TEST residual detected")
    if p.is_copy.any() or p.plan_missing.any() or not p.operating.all():
        raise ValueError("Ineligible residual day")
    sizes = p.groupby(["model", "target", "forecast_date"]).hour.agg(["size", "nunique"])
    if not (sizes == 24).all().all() or not p.hour.between(0, 23).all():
        raise ValueError("Incomplete daily residual path")
    if not np.isfinite(p[["actual", "prediction", "residual"]]).all().all():
        raise ValueError("Nonfinite residual")
