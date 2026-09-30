# 생산계획 기반 공장 전력사용량·피크 예측

제6회 K-인공지능 제조데이터 분석 경진대회, 문제 ⑤ **제조 생산데이터 기반 전력사용량 예측 및 최대피크 위험조건 분석**.

하루 전(D-1 자정)에 다음 날 00~23시의 시간별 평균전력과 15분 최대수요를 예측하고, 목표수요(190) 초과 위험을 판정하며, 생산계획 조정에 따른 피크 저감 효과를 시뮬레이션한다.

## 실행

```bash
pip install -r requirements.txt
python run_all.py
```

전처리 → 학습 → 검증/테스트 평가 → 오류분석 → 피크 저감 시뮬레이션 → SHAP·DiCE 해석 → 그림 생성까지 한 번에 실행되며, 결과는 `outputs/`에 저장된다. CPU 기준 약 10분.

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
pytest -q                          # 누수 감사 테스트
```

## 데이터

`data/okm_augumented_2021.csv` — KAMP 자원 최적화 AI 데이터셋 (2021-01-01 ~ 09-14, 1시간 단위 6,168행).

진단 결과와 처리:

| 발견 | 처리 |
|---|---|
| 257일 중 115일이 다른 날과 15분 전력 96값이 완전히 같은 증강 복제일 (1~6월 대부분) | 평가는 원본 구간(7월 이후)만. 학습에서는 제외하거나 낮은 가중치 |
| 무작위 K-fold는 복제일 누수로 오차를 약 32% 과소평가 | 시간순 분할만 사용 |
| `공장인원` = 생산량 / 시간 내 15분 전력 합 (오차 1e-9, 타깃 누수) | 피처에서 제외 |
| 7/13·7/15 48행의 시간 컬럼이 전력값으로 덮이고 생산량도 0으로 지워짐 | 시간은 일내 순번으로 복원, 계획은 결측 처리 |
| 8/28~29 전력 0 (정전·전면휴무) | 학습·평가에서 제외 |

## 구간

| 구간 | 기간 | 용도 |
|---|---|---|
| 학습 | 2021-01-08 ~ 07-15 | 모델 학습 |
| 검증 | 07-16 ~ 08-15 | 모델·임계값·보정량 선택 (하계휴가 주간 포함) |
| 테스트 | 08-16 ~ 09-14 | 최종 1회 평가 (학습+검증으로 재학습) |

## 모델

| 이름 | 설명 |
|---|---|
| `naive_24h`, `naive_168h` | 전일·전주 같은 시각 값 (베이스라인) |
| `ridge`, `lgbm` | 단일 선형회귀, 단일 LightGBM |
| `regime_lgbm` | 생산계획으로 가동일/휴무일을 먼저 나눈 뒤 가동일은 LightGBM(L1), 휴무일은 시간대별 중앙값 기저부하 |
| `regime_ens` | `regime_lgbm` + 직전 가동일 대비 잔차 학습 + 복제일 저가중 + LightGBM(L1) 0.3·ExtraTrees 0.7 앙상블 (하이퍼파라미터는 `experiments/`에서 검증 fold로만 결정) |

## 주요 결과 (전력 MAE)

| 모델 | 검증 (휴무 주간 포함) | 테스트 (평상 운전) |
|---|---|---|
| 전일 같은 시각 | 30.55 | 36.66 |
| 전주 같은 시각 | 43.11 | 8.26 |
| LightGBM 단일 | 29.87 | 6.84 |
| 레짐 전환 LightGBM | 12.60 | 6.65 |
| 레짐 전환 앙상블 | 9.77 | 6.23 |

- `공장인원`은 `생산량 / (시간 내 15분 전력 합)`으로 정확히 계산되는 값이라(타깃 누수) 피처에서 제외했다.
- 1시간 전 실측을 입력으로 줘도 테스트 MAE가 6.16이라, 평상 운전일의 하루 전 예측 오차는 바닥에 가깝다.

## 구조

```
run_all.py          전체 파이프라인
src/data.py         로드·정제·복제일 탐지
src/features.py     하루 전 예측용 피처 (시차, 생산계획, 운전 레짐, 기상)
src/models.py       비교 모델과 최종 모델
src/peak.py         피크 위험 판정, 피크 저감 시뮬레이션
src/explain.py      SHAP(전역·의존도·상호작용·개별사례), DiCE 반사실
src/evaluate.py     평가 지표
src/plots.py        보고서용 그림
src/peak_conditions.py  피크 발생 조건 분석
src/tariff.py       한전 산업용(을) 고압A 요금: 계시별 전력량요금, 기본요금, 래칫
src/rl_env.py       당일 수요 제어 강화학습 환경(gymnasium)과 기준 정책
src/milp.py         래칫 인지 MILP 부하이동(OR-Tools)
src/run_rl.py       제어 정책 비교 실행
src/decision_eval.py    결정 기반 평가(실측 고정), 인건비 할증 시나리오
src/eval_conditions.py  조건별 오차, 짝지은 부트스트랩
src/stochastic_milp.py  오차 시나리오 기반 확률 MILP
src/ratchet_rl.py   래칫 위험 강화학습(합성일, 행동 복제)
src/metaheuristics.py   GA·타부 탐색
src/horizon.py      예측 구간별 성능
src/rolling_origin.py   롤링 원점 평가
src/alert_threshold.py  비용 기준 경보 임계값
src/dashboard.py    운영 대시보드
tests/              누수 감사 테스트
docs/PROGRESS.md    진행 기록과 팀 계획
experiments/        튜닝·앙상블 실험 스크립트와 결과
```

## 주요 출력

- `outputs/test_predictions.csv` — 테스트 구간 시간별 예측, P90 상한(컨포멀 보정), 위험일 경보
- `outputs/model_compare_{valid,test}_{power,peak15}.csv` — 모델 비교표
- `outputs/peak_risk_*.csv` — 피크 위험 판정 성능, 사례
- `outputs/shaving_*.csv` — 피크 저감 시뮬레이션
- `outputs/shap_*.csv`, `outputs/dice_peak_actions.csv` — 해석 결과
- `outputs/fig*.png` — 그림 11종
