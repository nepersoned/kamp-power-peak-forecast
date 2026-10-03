# REPORT_DRAFT 수정 제안 — 본문은 아직 변경하지 않음

Phase-1 결과만으로 forecasting champion을 바꾸거나 synthetic covariance/PCA를 핵심 운영 방법으로 채택하지 않는다. TEST 결과도 아직 없다. 기존 REPORT_DRAFT 내용은 유지한다.

후속 결과 확정 후 수정 후보는 다음과 같다.

1. 오성민 담당 방법 절: `regime_ens`를 Phase-1 통제로 명시하고 최종 목표를 BEST FORECASTER → BEST UNCERTAINTY → BEST POLICY로 설명한다. TRAIN past-only residual provenance와 VALID TRAIN-only surrogate를 추가한다.
2. 확률예측 절: 기존 P90/conformal 옆에 24시간 residual dependence를 분석한다. 기존 empirical도 dependence를 보존한다는 사실을 정확하게 쓴다. Coverage/CRPS 외에 Energy/Variogram/event Brier를 추가한다. 이번 80% q10–q90 interval을 기존 one-sided P90 90% coverage와 혼동하지 않는다. 이번 Phase 1은 새로운 conformal/ACI tuning을 수행하지 않았다.
3. 표: PROGRESS §11의 7/19 제외 VALID scenario/decision 비교를 핵심 표 후보로 삼고 전체 포함 결과를 별도 제시한다. 17개 configuration/K/seed sensitivity는 부록으로 보낸다. Primary point MAE는 고정 forecaster라 모두 같다는 사실을 명시한다.
4. 그림: `fig_residual_corr_peak15.png`, `fig_pca_spectrum_peak15.png`, `fig_pca_loadings_peak15.png`를 구조 분석 후보로, `fig_0719_scenarios.png`를 case study 후보로 사용한다. PCA loadings를 재가동 원인으로 단정하지 않는다.
5. 결과 절: empirical이 probabilistic score에서 우수하고 일부 PCA가 decision regret에서 우수한 trade-off를 설명한다. Gaussian r3은 절감 차이 CI가 양수지만 Energy/ratchet Brier가 악화하므로 핵심 방법으로 채택하지 않는다. Bootstrap r2의 절감 CI는 0을 포함한다.
6. 7/19 절: floor206/실측222 kW, empirical 절감1,625,117원/oracle1,632,022원/regret6,905원 및 PCA bootstrap r4/r5 seed 실패를 설명한다. 이 날을 method selection에 쓰지 않았음을 명시한다.
7. 한계: 49개의 original operating paths, 과거 VALID-tuned champion configuration, 실측 기상 가정, TRAIN-only surrogate 기반 반사실적 비용, 단 하나의 VALID ratchet event, 여러 후보 탐색 및 재사용 VALID를 기록한다. TEST를 이미 확인한 과거 프로젝트 이력과 이번 TEST 미실행을 구분한다.

오성민 기여의 데이터 기반 문장은 “24시간 past-only 예측오차의 공동 구조를 추정하고, 공통 stochastic MILP에 연결해 확률 점수와 실현 decision regret를 함께 검증하였다”까지다. “PCA/covariance가 더 좋은 최종 운영 모델이다”는 이번 결과로 뒷받침되지 않는다. 최종 모델 성능 서사는 Phase 2 이후 작성한다.
