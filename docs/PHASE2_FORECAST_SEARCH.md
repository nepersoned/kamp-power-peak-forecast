# Phase 2: BEST FORECASTER (VALID only)

Phase 1은 commit `a8182b8b98b97ce56f03334240be071f7200da0b`에 보존했다. Empirical full-path, K30, lambda0 및 VALID 선택을 변경하지 않는다. 이번 Phase에서는 forecasting만 탐색하며 uncertainty/MILP를 실행하거나 수정하지 않는다.

## 고정한 protocol

**Primary selection metric = (MAE_power,F1 + MAE_peak15,F1 + MAE_power,F2 + MAE_peak15,F2) / 4.**

이는 기존 `experiments/harness.py`/`tune_et_ens.py` objective다. Fold1=[2021-07-02,07-16), Fold2=[07-16,08-16), TRAIN 시작=01-08. 각 fold 이전에만 parameter를 학습한다. MASK는 기존 harness와 같이 outage 제외 + lag168 유효이며 primary evaluation에는 copy/plan_missing을 포함한다. 원본만의 score는 sensitivity로 분리한다. 각 모델의 기존 copy 학습 정책을 유지한다(기존 champion은 저가중 학습, LGBM/TabPFN은 copy 제외).

기존 튜너의 score10.196435와 현재 deployed champion10.240550의 차이는 재현했다. 튜너 `off_pred`는 copy 포함 median이지만 deployed `RegimeModel`은 휴무 copy를 제거한다. 같은 operating predictions에 legacy off median을 적용하면 정확히10.196435가 된다. 기존 champion을 임의로 수정하지 않고 **현재 deployed 모델10.240550**을 공정한 공통 기준으로 삼는다.

TEST를 interpolation/feature 생성 전에 자른다. Legacy harness/run_all/rolling CLI를 import/실행하지 않는다(일부는 전체 데이터/TEST를 자동 처리한다). SARIMAX 및 TabPFN은 day24 단위로 예측한다. 미래 같은 날 actual을 입력하지 않으며 SARIMAX는 forecast24 이후에만 상태를 갱신한다. TabPFN query는 날짜별 분리하여 미래 query batch의 전처리 영향을 피한다. 실측 weather를 미래예보로 간주하는 기존 가정은 유지한다.

Model CSV는 공통 timestamp,target,point,q10,q50,q90,model에 date/hour/actual/prediction/fold/operating/is_copy/transition/peak_hour/training_end_date/runtime을 추가한다. q10/q90은 이번 point-only 단계에서는 NaN이며 calibrated distribution을 주장하지 않는다.

## 후보와 제한

- 기존 naive24/168, Ridge, LGBM, regime_lgbm, regime_ens.
- SARIMAX(1,0,0)×(1,0,0,24), stationary/invertible, 최대50 MLE iterations. Exog=log_prod, plan_on_day, hour sin/cos, holiday, temp, humid, weekly sin/cos, intercept. Exog scale/impute는 TRAIN만 사용한다. Copy label을 NaN observation으로 두어 복제에 의한 허위 seasonal likelihood를 줄인다. Calendar grid를 보존하며 하루24 direct horizon 뒤에만 D actual을 state에 넣는다. 이후 날짜는 갱신한 D-1 state를 이용한다. Parameter refit은 하지 않는다.
- statsmodels0.15.0은 Windows DLL load policy에서 실패했다. 공식0.14.6은 정상 import/fit한다. Day-level `extend`에서 constant exog와 constant trend가 충돌해 trend 대신 explicit exogenous intercept를 사용했다(모형적 intercept는 유지). 수렴 경고와 실패 log를 남겼다. Nonconverged 모델은 baseline 표에 남기되 deployment/ensemble 채택에서 제외한다.
- TabPFN9.1.0, public **TabPFN-v2** regression weight, n_estimators4, median output, seed42, CPU4 threads. 46개 기존 features를 유지하며 feature PCA/importance search를 하지 않는다. All/regime, regime context all/28/14일만 비교한다. 최근 calendar context의 original sample이 매우 적다는 사실을 저장한다.
- 공식 KV-cache `fit_with_cache`를 사용한다. 초기 `fit_preprocessors`는 날짜별 transformer 반복 때문에 CPU에서 지나치게 느려 중단했다. 동일 checkpoint/data/예측 문제를 유지했다. GPU는 없다(Intel UHD, 시스템 RAM 약16.9GB/논리CPU8). Core package versions는 requirements-phase2.txt에 기록한다.
- Chronos/SimpleRNN/Ridge stacking은 core 탐색 뒤 필요성이 정당화될 때만 수행한다. 이번 탐색이 성공하면 대규모 추가 모델 탐색은 하지 않는다. 가이드북 원형 파일이 repo에 없으며 168→1h 모델을 D-1→24h 동일 문제로 주장하지 않는다.

## TabPFN weight와 라이선스

Model ID=`Prior-Labs/TabPFN-v2-reg`, revision=`4972a65a1b30806315c6f92499959ffbfc69a673`, filename=`tabpfn-v2-regressor.ckpt`, SHA256=`2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736`, size44,390,977bytes. 최신 package default foundation model이라고 주장하지 않는다. CPU와 공개 weight 재현성을 위해 v2를 명시했다.

Weight license는 Prior Labs License1.1(2025-05), Apache2.0에서 attribution 요구를 추가한 형태다. 내려받은 LICENSE.txt를 outputs에 보존했다. [공식 model card](https://huggingface.co/Prior-Labs/TabPFN-v2-reg), [공식 구현/API](https://github.com/PriorLabs/TabPFN), [SARIMAX API](https://www.statsmodels.org/stable/generated/statsmodels.tsa.statespace.sarimax.SARIMAX.html)를 참고했다.

처음 weight 다운로드는 인터넷이 필요하다. Offline 제출 환경에서는 검증된 checkpoint를 미리 준비하고 `KAMP_TABPFN_MODEL_PATH`를 지정한다. Adapter는 hash mismatch를 거부하며, 파일이 없으면 pinned HF revision에서 다운로드한다. CPU에서도 실행 가능하지만 champion보다 fit 비용이 크다. v3 등 다른 weight의 성능은 이번 결과로 주장하지 않는다.

## 앙상블 및 채택 규칙

Pairwise residual Pearson/Spearman/covariance/disagreement와 날짜별 winner를 저장한다. Champion+각 non-naive 후보의 common w=0,.1,...,1을 비교한다. power/peak15별 w는 같은 grid에서 고르되 common 대비 primary 추가 개선>=.1kW일 때만 추가 구조로 남긴다. Endpoint w0/w1은 기존 single model이며 가짜 ensemble로 분류하지 않는다.

3-model은 genuine 2-model improvement가 있을 때만 검토한다. Third는 standalone primary<=champion×1.2이고 blended error correlation<.95여야 한다. 하나만 고르고 coarse simplex grid를 비교하며 two-model 대비 추가 개선>=.1kW일 때만 채택한다. 여러 model 조합 brute-force 또는 nonlinear stacking을 하지 않는다. Ridge stacking용 OOF를 새로 생성하지 않은 상태에서 VALID label로 meta learner를 적합하지 않는다.

Adoption gate는 primary 개선>=.05kW, 어느 fold도 champion보다>.2kW 악화하지 않음이다. 그 집합 중 primary 최소가 preliminary winner다. Finalists만 기존 weekly origins 07-12/19/26,08-02/09 (마지막은08-16에서 종료)에 재학습한다. 원본 row로 robustness를 평가하며 weights/context를 다시 튜닝하지 않는다. Winner가 주별 과반에서 개선하지 않거나 평균이 champion보다>.2kW 나쁘면 champion으로 돌아간다. 이 단계에서 다른 모델을 새로 골라 반복 평가하지 않는다.

Paired bootstrap은 날짜를 fold 내부에서 5,000회 resampling(seed42)하고 두 fold를 동일 비중으로 평균한다. 전체/operating/peak-window(08~11,13~16) CI를 기록한다. 날짜 간 독립성/여러 후보 탐색에 따른 CI의 탐색적 한계와 이미 공개된 과거 TEST 결과의 한계를 유지한다.

## 재현 및 Phase 3

```powershell
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements-phase2.txt
.venv/Scripts/python.exe -m pytest -q
.venv/Scripts/python.exe -m experiments.forecast_model_search --stage base
.venv/Scripts/python.exe -m experiments.forecast_model_search --stage tabpfn
.venv/Scripts/python.exe -m experiments.forecast_model_search --stage analysis
.venv/Scripts/python.exe -m experiments.forecast_model_search --stage rolling
```

출력은 outputs/forecast_models. 이전 설치 실패를 해결한 이번 실행은 base 다음 별도로 --stage sarimax를 실행했다. TEST option은 없다. Phase1 설정의 bytes 보존을 각 stage에서 검사한다.

Phase3는 frozen registry `make_forecaster(name)` 또는 `WeightedForecaster(components,weights)`를 이용한다. 선택된 모델로 **새로운** strictly past-only residual을 만들어야 한다. Phase1 `rolling_residuals`/`residual_matrix`/`JointResidualModel(empirical)`와 기존 MILP를 재사용할 수 있다. 다만 Phase1 기본 training_mask는 plan_missing을 제외하지만 이번 Phase2의 legacy protocol은 TRAIN plan_missing을 포함하므로 Phase3에서 fit-mask policy를 명시적으로 주입하는 최소 확장이 필요하다. 모델 factory만 바꾸고 훈련 protocol을 조용히 바꾸면 안 된다. Residual row의 original/operating/완전24h 필터는 유지한다. 이번에는 이 재구성이나 decision 평가를 실행하지 않는다.

## 동결한 결과

 오성민 Phase 2 — VALID BEST FORECASTER (2026-10-04)

Phase 1은 commit a8182b8에 보존했으며 empirical full-path K30/lambda0를 변경하지 않았다. Primary=(power F1 MAE+peak15 F1 MAE+power F2 MAE+peak15 F2 MAE)/4. 기존 tuning의 두 fold [07/02,07/16), [07/16,08/16)를 유지했다. TRAIN-only parameter fit, D-1 lag/plan, 원본-copy 정책 유지. TEST를 feature 생성 전 잘랐고 신규 TEST 예측/평가를 실행하지 않았다.

| model | power_mae | peak15_mae | primary_score | daily_max_peak15_mae | runtime_seconds |
|---|---|---|---|---|---|
| regime_tabpfn_all | 7.680 | 9.071 | 8.376 | 10.922 | 430.541 |
| regime_ens | 9.998 | 10.483 | 10.241 | 10.060 | 28.278 |
| regime_lgbm | 14.809 | 14.120 | 14.465 | 20.099 | 5.986 |
| tabpfn_all | 17.679 | 18.966 | 18.323 | 25.272 | 659.252 |
| lgbm | 22.165 | 24.429 | 23.297 | 30.168 | 7.175 |
| regime_tabpfn_28 | 27.342 | 21.176 | 24.259 | 23.622 | 75.895 |
| regime_tabpfn_14 | 27.503 | 21.144 | 24.324 | 23.821 | 71.412 |
| naive_168h | 26.275 | 28.822 | 27.549 | 38.256 | 0.020 |
| sarimax | 31.178 | 34.585 | 32.881 | 45.816 | 248.439 |
| ridge | 33.497 | 36.349 | 34.923 | 49.455 | 0.212 |
| naive_24h | 35.883 | 38.885 | 37.384 | 48.385 | 0.018 |

최종 선택은 **regime_tabpfn_all**. Regime-TabPFN은 휴무 기존 시간별 median + 가동 public TabPFN-v2(all original history)다. VALID primary는 8.375723 vs deployed regime_ens 10.240550으로 18.21% 개선했다. 두 fold primary는 9.4009/7.3505 vs 10.4046/10.0766으로 모두 개선했다. 기존 final_config10.196435는 copy 포함 휴무 median을 사용한 legacy 튜너와 deployed 코드 차이이며, 동일 operating 예측에 legacy median을 적용해 정확히 재현했다. Champion 코드를 수정하지 않았다.

Regime-TabPFN error Pearson은 fold/target별 .717–.820이다. 전체 paired date CI(candidate−champion)=[−3.752,−.295]kW, operating=[−5.062,−.861], peak-window=[−4.057,1.894]. CI는 탐색 후 비교/날짜 독립성의 한계가 있다. peak15 daily-max는10.922 vs10.060으로 악화했다. 일부 날짜는 더 나빠지므로 모든 조건의 개선을 주장하지 않는다.

2-model common weight grid 최적은 champion0/TabPFN1(단독)이다. Target별 power0/peak15 .1은 8.367842로 .007881kW만 추가 개선하여 복잡성 gate .1kW에 미달했다. genuine pair improvement가 없어 3-model/Ridge stacking은 수행하지 않았다. Chronos/SimpleRNN은 낮은 우선순위와 추가 필요성 부족으로 제외했다.

| week | regime_ens | regime_tabpfn_all | difference |
|---|---|---|---|
| 1 | 18.272 | 12.303 | -5.969 |
| 2 | 8.769 | 7.848 | -0.920 |
| 3 | 7.165 | 6.358 | -0.807 |
| 4 | 2.098 | 2.098 | 0.000 |
| 5 | 18.479 | 8.219 | -10.260 |

Frozen finalists만 weekly rolling origin/원본 rows로 검증했다. 평균차이 -3.591kW, 개선 4/5주로 사전 gate를 통과했다. Context/weight를 rolling 결과로 재튜닝하지 않았다.

TabPFN9.1.0, v2 pinned revision/checksum, CPU4 threads, 46 features, 4 estimators, median output/KV cache. 가동 TRAIN samples1224/1512, recent28/14는 첫 fold24개뿐이라 악화했다. Total VALID fit/predict430.5초 vs champion28.3초. SARIMAX(1,0,0)x(1,0,0,24)+10exog는4fit 모두50iteration내 nonconverged, primary32.881로 baseline만 남겼다. Package/weight 다운로드 및 offline 재현 조건은 PHASE2 문서에 기록했다.

전체 pytest **29 passed**, SWIG deprecation warning3개. Phase2의 신규7개 테스트는 future/context/copy/state/schema/score/hash guard를 검증한다. main, REPORT_DRAFT, Phase1 uncertainty/MILP는 수정하지 않았다. Phase2 변경은 미커밋이며 push/merge는 실행하지 않았다.

선택 명세는 experiments/forecast_model_selected.json, 상세 재현은 docs/PHASE2_FORECAST_SEARCH.md. Phase3는 make_forecaster 선택 모델로 strictly past-only residual을 새로 만들고 empirical K30/lambda0와 decision pipeline에 연결해야 한다. 기존 regime_ens residual 재사용 금지. Phase1 plan_missing 제외 fit-mask와 Phase2 legacy mask 차이를 명시적 policy로 연결해야 한다. 이번에는 Phase3/TEST를 실행하지 않았다.
