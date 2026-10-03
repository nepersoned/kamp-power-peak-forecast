# Phase 3: frozen Regime-TabPFN → empirical uncertainty → decision evaluation

## Fixed choices and Git history

Phase 2 commit `bef07d0139145c4e78dfa0c800d6fd323111d082` preserves the VALID-selected `regime_tabpfn_all` (8.375723 vs deployed `regime_ens` 10.240550). Model selection ended before Phase 3. TEST does not change the selected model even if generalization fails.

`experiments/final_system_config.json` freezes feature order (46), TabPFN package9.1.0/public v2 checkpoint revision+SHA256, estimator4/median/cache/CPU4 threads/seed42/all eligible context, original hour-median shutdown prediction, empirical full-path K30, seeds0/1/2, alpha.9 and lambda0. No additional conformal calibration. Source hashes and dependency versions are part of this manifest. Phase1 covariance/PCA experiments remain auxiliary and are not reselected.

## Canonical eligibility

| Stage | Eligible | Copy | Missing plan | Shutdown | Outage / warm-up |
|---|---|---|---|---|---|
| Forecaster fitting | Jan8 <= row < forecast origin; lag168 valid | Passed to existing policy: TabPFN/off-median discard, baseline on-ensemble retains existing weights | Included as Phase2 selected protocol; regime defaults to operating | Original rows train off-median | Exclude outage rows; exclude before Jan8/missing lag |
| Residual pool | Complete original 24h operating day before Aug16 | Excluded | Excluded: scenario shape needs a known operating plan | Excluded: operating uncertainty pool | Exclude any outage/incomplete day; first14 original training days warm-up |
| Surrogate fitting | TRAIN+VALID original operating rows with known production plan | Excluded | Excluded: production response cannot be fit without a plan | Excluded | Same row-level lag/outage/start rules as fitting |
| Point TEST evaluation | Aug16–Sep14 non-outage rows, lag168 valid | Legacy mask retained; original-only secondary | Same conservative regime if present | Included | Exclude outage rows; daily-max uses available rows and this limitation is recorded |
| Probabilistic / decision TEST | Complete original 24h operating day, known plan, valid lag | Excluded | Excluded | Excluded | Exclude any outage/incomplete day |

`outputs/final_system/data_eligibility.csv` lists stage/date/eligible/excluded_reason and flags. Forecaster-training eligibility is the common row mask before model-specific copy handling. Other stages require whole days and list all applicable exclusion reasons. Different masks serve different purposes; none introduce future labels. The TRAIN `plan_missing` difference from Phase1 is resolved explicitly by Phase2's fitting mask; Phase1 code/selected config stay unchanged.

## Past-only residuals and reference system

Both `regime_ens` and selected Regime-TabPFN get freshly generated expanding-origin errors. They share the same eligible dates, 14-calendar-day refit cadence after at least14 original past training days, and pre-TEST cutoff. Each date's predictor fits only earlier rows. This matches Phase1 cadence but extends the pool through available VALID. These are label-out-of-sample predictions under VALID-selected hyperparameters, not a fresh independent validation of hyperparameter selection.

Reference System A uses `regime_ens` + the unchanged empirical scenario family on its own newly generated pre-TEST pool. This updates the pool's historical coverage in the same manner as System B, avoiding a confound from comparing TRAIN-only49-day coverage with extended TRAIN+VALID coverage. B uses Regime-TabPFN + its own errors. Old champion residuals are never used for B. C uses Regime-TabPFN point-only deterministic MILP as an ablation. Surrogate coefficients, oracle, tariffs, floor rules, constraints and evaluation days are shared across A/B/C.

The reference is the Phase2 deployed champion, not the older stochastic CLI's hardcoded `RegimeModel` center. This resolves the earlier forecaster inconsistency through injection without changing solver defaults or legacy entry points.

## TEST isolation and one-run ledger

Before TEST, `load(end=Aug16)` precedes interpolation/features/copy detection for all fitting/pool construction. During final execution the fit frame remains this independent prefix; TEST cannot alter its interpolation or duplicate classification. TEST features use the frozen feature builder and D-1-or-earlier observed lag values, day-ahead production plan and the existing weather-availability assumption.

Models and residual pools are fixed during TEST. D-1 TEST observations can enter future days' lag features and tariff floor history, as in the original forecasting problem; they never refit model parameters, re-estimate uncertainty or enter the residual pool. Scenario seed=`repeat_seed*100000+dayofyear`, repeats0/1/2, K30 for every method/day. Sampling full residual vectors preserves within-day dependence; no PCA/covariance selection occurs.

Pre-TEST inspection corrected the inference boundary: every calendar day gets a complete24h forecast first; only scoring then excludes outage rows. Future outage labels never select query batches. This was repaired before any TEST access, without changing model/uncertainty parameters, and recorded in the freeze manifest. `test_day_ahead_all_hours.csv` retains complete forecasts including excluded scoring hours.

`pretest_audit.json` records checks and pytest output. Integration code/config are committed before execution. The `test_execution.json` ledger is created before TEST data access, records the pre-TEST commit and marks completion/failure. The CLI refuses an existing ledger rather than silently repeating TEST. A failure requires a documented bug/environment repair; a poor metric never permits rerunning or tuning.

## Evaluation and limits

Point metrics use the legacy hourly mask with target-equal MAEs; daily max is peak15. Point bootstrap averages target errors within dates (including partial non-outage days), so its mean delta can differ slightly from the hourly-weighted aggregate. Probabilistic CRPS/coverage/pinball, Energy/Variogram and event metrics use decision-eligible complete operating days. Power CRPS is separate; other joint/event metrics are peak15. Q10/Q90 are empirical scenario quantiles; point-only exports keep them missing. Seed0 distribution exports are examples, while reported scores/decisions average all3 frozen repeats by day. Quantile scoring does not imply a conformal coverage guarantee.

Forecast paired CI uses5000 date resamples, seed42: overall/operating/peak-sensitive windows(08–11,13–16). Decision CI first averages3 seeds for each paired date, then resamples dates. Negative-saving days/worst day are calculated on seed-averaged days; seed-specific decisions are retained. No pseudo-replication of seeds as independent dates. Actual ratchet exceedance is the billable max > past-only floor, not unqualified daily max. No-event PR-AUC is undefined and saved as NaN.

Decisions call the existing `solve_stochastic`/`solve_day` and `true_cost`, without duplicating constraints or optimization. Costs use actual baseline + fitted linear surrogate change for a proposed plan. This is surrogate counterfactual evaluation, not actual factory intervention. Forecast MAE improvement need not improve saving/regret, especially when the TEST ratchet floor has already been established. July19 remains a separate VALID case study and its settings/results are not retuned.

Small original sample size, copied history, pretrained TabPFN prior, observed weather used as forecast proxy, rare ratchet events, historical TEST inspection elsewhere in the repo and a single new frozen evaluation limit generalization claims. CPU latency and initial network/cache requirements limit deployment. The chosen model remains VALID-selected regardless of TEST outcome.

## Reproduction

From the repository root on `seongmin`, Python3.12/Windows CPU:

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-phase2.txt
.venv/Scripts/python.exe -m pytest -q
# final_system_config.json is checked in: do not recreate or tune it.
.venv/Scripts/python.exe -m experiments.final_system --stage prepare
.venv/Scripts/python.exe -m experiments.final_system --stage audit
# The integration/config commit must already be present; working tree clean.
.venv/Scripts/python.exe -m experiments.final_system --stage test
```

Weight cache must contain the pinned hash (or initial network access is required); offline use `KAMP_TABPFN_MODEL_PATH`. Stage `freeze` is only for initial manifest creation before TEST and refuses an existing manifest. `summarize` only derives tables from saved final results; it does not run predictions or optimization. A fresh reproduction is a replication of the frozen evaluation, not new model selection. Each output directory permits a single successful final run.

Output: `outputs/final_system/` contains config/audit/ledger, eligibility, model-specific residual provenance/covariance/correlation/tails, point/distribution predictions, forecast/probabilistic/decision metrics, full production recommendations, paired CIs and reliability tables.
