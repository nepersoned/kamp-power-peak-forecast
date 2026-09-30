"""누수 감사 테스트: 새 피처를 넣을 때마다 `pytest -q`로 확인한다.

1. 원자료 항등식: 공장인원 = 생산량 / (15분 값 합), 평균 = 15분 값 평균(반올림) → 이 열들은 피처 금지
2. 미래 교란 검사: D일 전력을 바꿔도 D일 피처는 변하지 않아야 한다(하루 전 예측 조건)
3. 시차 피처가 실제로 과거 값인지
4. 평가일이 학습 구간 날짜의 복사본이 아니어야 한다
"""
import numpy as np
import pandas as pd
import pytest

from src.data import Q, RAW, load
from src.features import build


@pytest.fixture(scope="module")
def data():
    df = load()
    return df, build(df)


def test_raw_identities_hold():
    raw = pd.read_csv(RAW)
    s = raw[["15분", "30분", "45분", "60분"]].sum(axis=1)
    ok = (s > 0) & raw["공장인원"].notna()
    assert np.abs(raw.loc[ok, "공장인원"] - raw.loc[ok, "생산량"] / s[ok]).max() < 1e-6
    assert (raw["평균"] == np.floor(raw[["15분", "30분", "45분", "60분"]].mean(axis=1) + 0.5)).all()


def test_leaky_columns_not_in_features(data):
    _, X = data
    banned = {"workers", "workers_day", "workers_roll3", "power", "peak15", *Q}
    assert not banned & set(X.columns), banned & set(X.columns)


def test_future_perturbation_does_not_change_same_day_features(data):
    df, X = data
    day = pd.Timestamp("2021-08-20")
    rows = pd.date_range(day, periods=24, freq="h")
    df2 = df.copy()
    df2.loc[rows, ["power", "peak15", *Q]] = df2.loc[rows, ["power", "peak15", *Q]] * 3 + 50
    X2 = build(df2)
    a, b = X.loc[rows], X2.loc[rows]
    changed = [c for c in X.columns if not np.allclose(a[c].to_numpy(float), b[c].to_numpy(float), equal_nan=True)]
    assert not changed, f"당일 실측 전력에 반응하는 피처: {changed}"


@pytest.mark.parametrize("lag", [24, 48, 168])
def test_lag_features_are_past_values(data, lag):
    df, X = data
    s = X[f"power_lag{lag}"].dropna()
    assert np.allclose(s.to_numpy(), df["power"].shift(lag).loc[s.index].to_numpy())


def test_evaluation_days_not_copies_of_training_days(data):
    """평가일(검증·테스트)의 전력 프로파일이 학습 구간 날짜의 복사본이면 안 된다.
    (7/30은 7/28의 복사본이지만 둘 다 검증 구간이라 학습→평가 누수는 아님)"""
    df, _ = data
    day = df.index.normalize()
    sig = df[Q].groupby(day).apply(lambda x: hash(tuple(x.to_numpy().ravel())))
    train_sigs = set(sig[sig.index < "2021-07-16"])
    ev = sig[(sig.index >= "2021-07-16") & (sig.index <= "2021-09-14")]
    leaked = [str(d.date()) for d, v in ev.items() if v in train_sigs]
    assert not leaked, leaked
