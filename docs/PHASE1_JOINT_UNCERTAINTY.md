# Phase 1: 공동 오차구조 → 확률적 생산계획

최종 목표는 **BEST FORECASTER → BEST UNCERTAINTY MODEL → BEST DECISION / CONTROL POLICY**다. `regime_ens`는 현재 champion이며 Phase 1의 실험 통제다. 최종 forecasting model로 미리 확정하지 않는다. Phase 2에서는 forecasting 성능 탐색과 단순 앙상블을 우선한다.

## 고정한 설계

- 기존 구간: TRAIN 2021-01-08~07-15, VALID 07-16~08-15. TEST는 실행하지 않는다.
- 기존 `final_model()`을 주입해 peak15/power를 함께 예측한다. 공동 불확실성은 원본 가동일에 한정한다.
- TRAIN에서 원본 날짜 14일 이상이 확보된 후 expanding rolling origin을 시작한다. 모델은 최대 14일 단위로 다시 적합한다. 각 block 안의 모든 평가일은 적합 종료보다 뒤다. 날짜별 provenance에 실제 마지막 적합 timestamp를 기록한다.
- 각 TRAIN 날짜는 자기 날짜 및 이후 target을 학습에 쓰지 않는다. 기존 GroupKFold residual은 사용하지 않는다.
- 훈련에서 champion의 복제일 저가중치 방식을 유지하되, residual pool과 VALID 평가에서 복제일을 제외한다. 계획 누락/정전/불완전 시차/휴무일도 제외한다.
- VALID forecaster와 decision surrogate는 모두 TRAIN만 사용한다. `build_days()`/`build_all()`을 호출하지 않으며 TEST 모델/결정을 생성하지 않는다.
- 입력 데이터는 interpolation/feature 생성 전에 TEST 시작에서 자른다. 기존 weather/생산계획을 알고 있다는 가정은 유지한다. 이는 실서비스 기상예보 성능을 검증한 것이 아니다.

## 비교 방법

1. Empirical full-path: 실제 24시간 residual vector를 함께 추출한다. **기존 방식도 시간 내 dependence를 보존한다.**
2. Independent hour: 시간별 residual을 따로 추출해 dependence의 기여를 확인한다.
3. Cholesky: empirical/Ledoit–Wolf/OAS covariance를 Gaussian sampling에 사용한다.
4. PCA Gaussian: 중심화한 kW residual의 SVD, factor variance 기반 Gaussian sampling.
5. PCA bootstrap: 과거 factor score vector 전체를 함께 추출하고 rank-r 경로로 복원한다. MVP에서는 perturbation 또는 누락 성분용 추가 noise를 넣지 않는다. 따라서 새 tail을 생성한다고 주장할 수 없다.

공분산을 대칭화한 뒤 eigenvalue를 `max(최대 eigenvalue × 1e-8, 1e-8)` 이상으로 제한한다. 안정화 전/후 최소 eigenvalue와 condition number를 저장한다. PCA는 raw features를 압축하지 않는다. rank 후보는 2/3/4/5/6/8이다. 모든 방법의 forecast+residual을 0 이상으로 clip한다.

K=30/50, seed=0/1/2, CVaR alpha=.9, lambda=0, 12개월 래칫, 기존 동일 생산제약을 사용한다. Empirical/PCA bootstrap은 표본 수가 K보다 작을 때 replacement를 허용해 **정확히 K개**를 만든다. 그 외에는 기존 sampled baseline과 같이 비복원 추출한다. optimizer는 기존 `solve_stochastic()`을 그대로 사용한다. 과거 full-pool empirical 최종안의 숫자와 이번 sampled-K 결과는 별도 실험이다.

## 평가와 선택 규칙

각 seed/day에서 CRPS, 80% q10–q90 coverage/width, pinball, Energy Score, Variogram Score(p=.5, unordered pair 평균), 시간별 TAU=190 Brier 및 요금 산정 시간 daily-max ratchet Brier를 계산한다. Event PR-AUC는 average precision이다. Positive가 없으면 undefined로 기록한다. Reliability는 seed 평균 probability를 사용한다.

실현 비용은 기존 `true_cost()`의 actual+surrogate delta를 사용한다. Saving은 무조치 실제비용−계획 실제비용, oracle saving은 actual baseline MILP 계획의 절감, regret은 oracle saving−saving이다. 현장 실측 절감이 아닌 surrogate 기반 반사실적 실험이다. Oracle의 clipping/solver 한계 때문에 엄밀한 global optimal regret이라고 주장하지 않으며 음수 regret을 자르지 않는다. Solver status와 objective/bound를 보존한다.

방법 선택에는 **7/19를 제외한 VALID**만 사용한다. K=30을 사전 primary로 정하고 K=50은 sensitivity로 기록한다. 새 joint 방법은 empirical 대비 일별 paired saving bootstrap 95% CI 하한이 양수이고 Energy Score/ratchet Brier가 나빠지지 않을 때만 채택한다. 해당 후보 중 regret가 가장 작은 것을 선택한다. 근거가 부족하면 empirical을 유지한다. Reconstruction error만으로 rank를 선택하지 않는다. Paired bootstrap은 3개 seed를 날짜별 평균한 뒤 날짜를 2,000회 resampling하며 seed=42다.

표본/후보 수가 작고 많아 CI는 탐색적이다. 날짜 간 독립성, multiple comparisons를 엄밀히 보장하지 않는다. TEST로 선택을 보정하지 않는다. `regime_ens` hyperparameter 자체는 기존 VALID 튜닝 결과이므로 rolling provenance가 past-only라는 사실이 모든 hyperparameter도 당시 시점에 선택되었다는 뜻은 아니다.

## 인터페이스와 Phase 2

Forecaster는 기존 `fit(X, y, hour, copy)`/`predict(X, hour)` 규약을 사용한다. `rolling_residuals(..., factory)`에 forecaster factory를 주입할 수 있다. Uncertainty는 `JointResidualModel(config).fit(E).scenarios(point24, K, seed)`다. MILP는 모델명 대신 `Day.fc_pw`와 임의 `(K,24)` peak scenarios를 받는다.

CSV 공통 schema는 `timestamp,target,point,q10,q50,q90,model`이다. `point`는 scenario median과 다를 수 있다. 기존 `decision_eval.load_external()`은 point가 있으면 이를 사용하고 q50-only 파일도 계속 지원한다.

Phase 2에서는 기존 naive_24h/168h/Ridge/LightGBM/regime_lgbm/regime_ens와 SARIMAX/TabPFN(가능하면 Chronos)을 같은 validation fold 및 mask로 비교한다. Primary는 `experiments/harness.py`의 두 fold × 두 target MAE 동일 비중 평균을 유지한다. 일최대/피크/전환/가동일/rolling 성능을 secondary로 기록한다. 모델별 residual correlation과 diversity를 계산한 뒤 regime_ens+신규 모델 weight 0:.1:1, 최대 3개 단순 앙상블을 VALID에서만 탐색한다. 모든 설정을 freeze한 후 별도 승인을 받아 TEST를 최종 평가한다.

## 재현

저장소 root에서 Python 3.12로 실행한다. 전체 기존 run_all/stochastic CLI는 TEST를 실행할 수 있으므로 Phase 1에는 아래 전용 command만 사용한다.

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-phase1.txt
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m experiments.joint_scenarios --workers 4
```

Residual cache 재사용은 마지막 command에 `--reuse-residuals`를 추가한다. 데이터/config/forecasting/residual source hash가 다르면 거부한다. 병렬 job 수는 직원수 feature와 관계없는 계산 설정이며 forecasting에 workers feature를 쓰지 않는다. 결과는 `outputs/joint_uncertainty/`에 저장된다. TEST 실행 option은 제공하지 않는다.
