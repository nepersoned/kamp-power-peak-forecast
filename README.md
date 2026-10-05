# 생산계획 기반 공장 전력사용량·피크 예측과 래칫 인지 피크 관리

제6회 K-인공지능 제조데이터 분석 경진대회, 문제 ⑤ **제조 생산데이터 기반 전력사용량 예측 및 최대피크 위험조건 분석**.

하루 전(D-1 자정)에 다음 날 00~23시의 시간별 평균전력과 15분 최대수요를 예측하고(최종 모델 **Regime-TabPFN**), 과거 예측 오차 경로로 만든 시나리오로 피크·래칫 위험을 판정하며, 래칫 인지 확률 MILP로 생산 시점 조정 계획을 권고한다.

## 실행

Python 3.12, CPU. 환경은 `requirements.txt` 하나로 통일했다.

전체를 한 번에: `python run_pipeline.py` (테스트 → 비교 모델 → 최종 시스템 → 보고서 표·그림, CPU 약 45분). 단계별 실행은 아래와 같다.

```bash
python -m venv .venv
.venv/Scripts/python.exe -m pip install -r requirements.txt     # Linux/macOS: .venv/bin/python
.venv/Scripts/python.exe -m pytest -q                            # 누수 감사·예측·불확실성·최종 시스템 테스트

# 1) 비교 모델·오류분석·해석·그림 (outputs/, CPU 약 10분)
.venv/Scripts/python.exe run_all.py

# 2) 최종 시스템: Regime-TabPFN → 오차 시나리오 → 확률 MILP (outputs/final_system/)
.venv/Scripts/python.exe -m experiments.final_system --stage prepare   # 학습·오차 풀(테스트 이전 데이터만)
.venv/Scripts/python.exe -m experiments.final_system --stage audit     # 동결 설정·소스 해시·pytest 점검
.venv/Scripts/python.exe -m experiments.final_system --stage test      # 테스트 1회(같은 출력 폴더에서 재실행 거부)

# 3) 모델 탐색(검증 fold 2개) 예측 재생성과 보고서 보조 분석
.venv/Scripts/python.exe -m experiments.forecast_model_search --stage base
.venv/Scripts/python.exe -m experiments.forecast_model_search --stage tabpfn
.venv/Scripts/python.exe -m src.final_analysis --stage tables       # F1·조건별 오차 (outputs/final_analysis/)
.venv/Scripts/python.exe -m src.final_analysis --stage importance   # 피처 그룹 순열 중요도
.venv/Scripts/python.exe -m src.final_analysis --stage plan         # 생산계획 가정 민감도
```

- `experiments.final_system`은 `.venv/Scripts/python.exe`가 저장소 루트에 있다고 가정한다(audit 단계가 pytest를 그 경로로 실행).
- `audit`은 동결 설정(`experiments/final_system_config.json`)에 기록된 소스 파일 해시를 확인한다. 줄바꿈이 CRLF로 바뀌면 해시가 달라지므로 저장소는 `.gitattributes`로 LF를 고정했다.
- TabPFN 가중치(TabPFN-v2 회귀, 고정 revision·SHA256)는 첫 실행 때 내려받는다. 오프라인 환경은 같은 해시의 파일을 `KAMP_TABPFN_MODEL_PATH`로 지정한다.
- TabPFN은 CPU 연산 순서에 따라 예측이 소수점 수준에서 달라질 수 있다. 예측 MAE는 사실상 같지만(5.760 / 5.769) 확률 MILP의 하루 절감은 실행 환경에 따라 9.2~9.5천 원 범위로 달라졌다.

제출용 zip(소스·requirements·데이터·README·테스트 예측결과, 블라인드 검사 포함): `python tools/make_submission.py` → `dist/submission_source.zip`

추가 분석(각각 독립 실행, 결과는 `outputs/` 하위 폴더):

```bash
python -m src.peak_conditions      # 피크 발생 조건 (outputs/peak_conditions/)
python -m src.tariff               # 한전 요금·래칫 월별 계산 (outputs/tariff/)
python -m src.eval_conditions      # 조건별 오차, 모델 간 차이 부트스트랩 CI (outputs/conditions/)
python -m src.decision_eval test   # 결정 기반 평가: 예측기별 MILP 계획의 실제 절감 (outputs/decision/), valid도 가능
python -m src.run_rl --timesteps 200000   # 제어 정책 비교: 무조치·규칙·탐색·MILP·PPO (outputs/rl/), --load로 저장 모델 평가
python -m src.stochastic_milp      # 오차 시나리오 확률 MILP (outputs/stochastic/)
python -m src.ratchet_rl           # 래칫 합성일 강화학습(PPO), --bc로 MILP 시연 행동 복제 (outputs/ratchet_rl/)
python -m src.metaheuristics       # GA·타부 vs MILP, --split으로 연속 분할 GA (outputs/metaheuristics/)
python -m src.horizon              # 예측 구간(1~24시간 앞)별 성능 곡선 (outputs/horizon/)
python -m src.rolling_origin       # 주 단위 롤링 원점 평가 (outputs/rolling/)
python -m src.alert_threshold      # 비용 기준 래칫 위험 경보 임계값 (outputs/alert/)
python -m src.dashboard 2021-07-19 # 운영 대시보드 예시 (outputs/dashboard/)
```

## 데이터

`data/okm_augumented_2021.csv` — KAMP 자원 최적화 AI 데이터셋 (2021-01-01 ~ 09-14, 1시간 단위 6,168행).

진단 결과와 처리:

| 발견 | 처리 |
|---|---|
| 257일 중 115일이 다른 날과 15분 전력 96값이 완전히 같은 증강 복제일 (1~6월 대부분) | 오차 시나리오·의사결정 평가는 원본만. 학습에서는 제외(TabPFN)하거나 낮은 가중치(앙상블) |
| 무작위 K-fold는 복제일 누수로 오차를 약 32% 과소평가 | 시간순 분할만 사용 |
| `공장인원` = 생산량 / 시간 내 15분 전력 합 (오차 1e-9, 타깃 누수) | 피처에서 제외 |
| 7/13·7/15 48행의 시간 컬럼이 전력값으로 덮이고 생산량도 0으로 지워짐 | 시간은 일내 순번으로 복원, 계획은 결측 처리 |
| 8/28~29 전력 0 (정전·전면휴무) | 학습·평가에서 제외 |

## 구간

| 구간 | 기간 | 용도 |
|---|---|---|
| 학습 | 2021-01-08 ~ | 각 검증 fold 이전 데이터로만 학습 |
| 검증 fold1 / fold2 | 07-02 ~ 07-15 / 07-16 ~ 08-15 | 모델·하이퍼파라미터·임계값 선택 (fold2에 하계휴가 주간 포함) |
| 테스트 | 08-16 ~ 09-14 | 동결 설정으로 최종 1회 평가 (8/15까지로 재학습) |

## 모델

| 이름 | 설명 |
|---|---|
| `naive_24h`, `naive_168h` | 전일·전주 같은 시각 값 (베이스라인) |
| `ridge`, `lgbm`, `sarimax`, `tabpfn_all` | 단일 선형회귀, 단일 LightGBM, SARIMAX, 단일 TabPFN |
| `regime_lgbm` | 생산계획으로 가동일/휴무일을 먼저 나눈 뒤 가동일은 LightGBM(L1), 휴무일은 시간대별 중앙값 기저부하 |
| `regime_ens` | `regime_lgbm` + 직전 가동일 대비 잔차 학습 + 복제일 저가중 + LightGBM(L1) 0.3·ExtraTrees 0.7 앙상블 |
| **`regime_tabpfn_all` (최종)** | 같은 레짐 구조에서 가동일 모델을 TabPFN-v2(원본 가동일 전체 이력)로 교체 |

## 주요 결과

| 모델 | 검증 점수(2 fold × 2 타깃 MAE 평균) | 테스트 전력 MAE | 테스트 15분최대 MAE |
|---|---|---|---|
| 전일 같은 시각 | 37.38 | 36.66 | 40.04 |
| 전주 같은 시각 | 27.55 | 8.26 | 9.73 |
| LightGBM 단일 | 23.30 | 6.84 | 7.38 |
| TabPFN 단일 | 18.32 | — | — |
| 레짐 전환 LightGBM | 14.47 | 6.65 | 7.58 |
| 레짐 전환 앙상블 | 10.24 | 6.23 | 7.23 |
| **Regime-TabPFN** | **8.38** | **5.76** | **6.75** |

- 레짐 구조 없이 TabPFN만 쓰면 휴가 주간에서 무너진다(18.32). 레짐 구조 + 사전학습 모델 결합이 핵심이다.
- 테스트 30일(평상 운전)에서 앙상블 대비 개선의 95% CI는 [−1.17, +0.19]kW로 0을 포함한다.

## 구조

```
run_all.py                 비교 모델 파이프라인(오류분석·해석·그림)
experiments/final_system.py, final_system_config.json   최종 시스템 실행과 동결 설정
experiments/forecast_model_search.py   검증 fold 모델 탐색(TabPFN·SARIMAX 포함)
src/data.py                로드·정제·복제일 탐지
src/features.py            하루 전 예측용 피처 (시차, 생산계획, 운전 레짐, 기상)
src/models.py              비교 모델과 레짐 구조
src/tabpfn_model.py, src/sarimax_model.py   TabPFN·SARIMAX 어댑터
src/forecast_interface.py, src/forecast_adapters.py   예측-최적화 공통 인터페이스
src/joint_uncertainty.py, src/probabilistic_eval.py   오차 경로 시나리오, 분포 평가
src/final_system.py        최종 시스템 데이터 적격성·오차 풀
src/final_analysis.py      최종 모델 기준 F1·조건별 오차·중요도·계획 민감도
src/peak.py                피크 위험 판정, 피크 저감 시뮬레이션
src/explain.py             SHAP(전역·의존도·상호작용·개별사례), DiCE 반사실
src/evaluate.py, src/plots.py   평가 지표, 보고서용 그림
src/peak_conditions.py     피크 발생 조건 분석
src/tariff.py              한전 산업용(을) 고압A 요금: 계시별 전력량요금, 기본요금, 래칫
src/milp.py, src/stochastic_milp.py   래칫 인지 MILP, 오차 시나리오 확률 MILP(OR-Tools)
src/rl_env.py, src/run_rl.py, src/ratchet_rl.py, src/metaheuristics.py   강화학습·GA·타부 비교
src/decision_eval.py       결정 기반 평가(실측 고정), 인건비 할증 시나리오
src/eval_conditions.py, src/horizon.py, src/rolling_origin.py   조건별 오차, 예측 구간, 롤링 평가
src/alert_threshold.py, src/dashboard.py   비용 기준 경보 임계값, 운영 대시보드
tests/                     누수 감사·예측 모델·불확실성·최종 시스템 테스트
docs/                      보고서 초안, 진행 기록, 단계별 설계 문서
```

## 주요 출력

- `outputs/final_system/test_predictions_regime_tabpfn.csv` — **최종 모델 테스트 예측**(시각별 전력·15분 최대수요)
- `outputs/final_system/test_forecast_distributions.csv` — 오차 시나리오 분위수(Q10·Q50·Q90)
- `outputs/final_system/test_production_plans.csv`, `test_daily_decisions.csv` — 권고 생산계획과 날짜별 절감
- `outputs/final_system/test_forecast_comparison.csv`, `test_probabilistic_metrics.csv`, `test_decision_metrics.csv` — 최종 비교표
- `outputs/final_analysis/` — F1, 조건별 오차, 그룹 중요도, 계획 민감도
- `outputs/test_predictions.csv` — 비교 모델(레짐 앙상블) 예측, P90 상한(컨포멀 보정), 위험일 경보
- `outputs/model_compare_*.csv`, `outputs/peak_risk_*.csv`, `outputs/shap_*.csv`, `outputs/fig*.png` — 비교표·판정·해석·그림

## 오성민 Phase 1: joint uncertainty (TRAIN → VALID 전용)

최종 목표는 BEST FORECASTER → BEST UNCERTAINTY MODEL → BEST DECISION / CONTROL POLICY다.
Phase 1에서는 champion `regime_ens`를 실험 통제로 고정한다. 최종 forecasting model 확정은 Phase 2의 공정한 모델/앙상블 비교 후에 한다.

```bash
pip install -r requirements-phase1.txt
python -m pytest -q
python -m experiments.joint_scenarios --workers 4
```

이 command는 TEST를 평가하지 않는다. 결과는 `outputs/joint_uncertainty/`에 저장된다.
Phase-1 VALID 결론은 empirical full-path 유지(K30, lambda=0)이며 새 covariance/PCA의 우월성을 주장하지 않는다.
상세 설계·제약·재현은 [Phase-1 문서](docs/PHASE1_JOINT_UNCERTAINTY.md), 결과는 [PROGRESS §11](docs/PROGRESS.md)에 있다.

## 오성민 Phase 2: forecasting search (VALID 전용)

Phase 1 empirical K30/lambda0는 동결하고, 두 validation fold·두 target MAE의 동일 비중 평균으로 forecaster를 비교한다.

```bash
pip install -r requirements-phase2.txt
python -m pytest -q
python -m experiments.forecast_model_search --stage base
python -m experiments.forecast_model_search --stage tabpfn
python -m experiments.forecast_model_search --stage analysis
python -m experiments.forecast_model_search --stage rolling
```

결과는 `outputs/forecast_models/`, 동결한 명세는 `experiments/forecast_model_selected.json`이다. TEST 실행 option은 없다. TabPFN은 명시한 공개 v2 checkpoint를 사용하며 최초 다운로드에는 인터넷이 필요하다. Offline 환경은 검증된 파일에 `KAMP_TABPFN_MODEL_PATH`를 지정한다. [설계·라이선스·재현](docs/PHASE2_FORECAST_SEARCH.md)을 확인한다. Phase 3에서는 선택 forecaster의 past-only residual을 새로 적합해야 한다.

## Phase 3: frozen final system

최종 구성은 Regime-TabPFN → 새 past-only residual pool → empirical K30/lambda0 → 기존 stochastic MILP다. TEST 1회 실행 전 코드/config commit과 audit를 남긴다. 기존 `run_all.py`는 historical baseline이며 최종 시스템은 아래 entry point를 사용한다.

```bash
python -m experiments.final_system --stage prepare
python -m experiments.final_system --stage audit
python -m experiments.final_system --stage test
```

동결 config가 있는 pre-TEST commit과 clean working tree에서 실행한다. 완료한 TEST ledger가 있으면 재실행을 거부한다. `--stage summarize`는 저장된 표만 재계산한다. [최종 설계·mask·재현](docs/PHASE3_FINAL_SYSTEM.md), [보고서 최종 비교](docs/REPORT_DRAFT.md)를 참고한다. TEST에서 primary MAE는6.727→6.252였지만 forecast/decision 차이 CI가0을 포함했고 ratchet 사건은0건이었다. TEST 결과로 모델/uncertainty/policy를 재선택하지 않았다.

Forecast/decision 통합 계약과 전달 파일 설명은 [Forecast Model Handoff](docs/FORECAST_MODEL_HANDOFF.md)에 있다. Point/quantile CSV만으로 stochastic joint paths를 복원하지 않고, 별도 `handoff_joint_scenarios.csv`의 scenario×hour 배열을 전달한다.
