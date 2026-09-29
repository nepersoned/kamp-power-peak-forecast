"""하루 전(D-1 24시) 시점 예측을 위한 피처 생성.

예측 과제: D일 00~23시의 시간별 평균전력(power)과 15분 최대수요(peak15)를 D-1일 자정에 예측.
  - 전력 시차 피처는 24시간 이상만 사용 (D-1일까지 관측된 값만 사용 -> 누수 없음)
  - 생산량·인원: 전일 확정되는 생산계획으로 간주 (계획값 없이 예측하는 시나리오는 USE_PLAN=False로 비교)
  - 기상: 기상청 단기예보로 대체 가능하다고 가정 (보고서에 한계로 명시)
"""
import numpy as np
import pandas as pd

from .data import Q

# 2021 공휴일(대체공휴일 포함), 데이터 기간 내
HOLIDAYS = pd.to_datetime([
    "2021-01-01", "2021-02-11", "2021-02-12", "2021-02-13", "2021-03-01",
    "2021-05-05", "2021-05-19", "2021-06-06", "2021-08-15", "2021-08-16",
])

# 공장인원은 생산량 / (시간 내 15분 전력 합)으로 정확히 계산되는 값(오차 1e-9) -> 타깃 누수라 사용하지 않는다
PLAN = ["prod", "prod_day", "prod_share", "prod_prev_h", "prod_next_h"]
WEATHER = ["temp", "humid", "wind", "rain", "temp_day_max"]


def _day_lag(df, col, days, agg):
    daily = df[col].resample("D").agg(agg)
    return df.index.normalize().map(daily.shift(days))


def _last_operating_day_lags(df, plan_on):
    """레짐 조건부 시차: 달력상 어제가 아니라 '직전 가동일'의 같은 시각 값.
    휴무·연휴 직후에 전일 시차가 기저부하로 오염되는 문제를 막는다."""
    day = df.index.normalize()
    on_day = plan_on.groupby(day).first().astype(bool)
    op_days = on_day[on_day].index
    prev_op = pd.Series(op_days.searchsorted(on_day.index) - 1, index=on_day.index)
    prev_op = prev_op.map(lambda i: op_days[i] if i >= 0 else pd.NaT)   # 오늘보다 앞선 가동일
    src = df.index.map(lambda t: prev_op[t.normalize()]) + pd.to_timedelta(df["hour"].to_numpy(), unit="h")
    out = pd.DataFrame(index=df.index)
    out["op_power_lag"] = df["power"].reindex(src).to_numpy()
    out["op_peak_lag"] = df["peak15"].reindex(src).to_numpy()
    out["op_prod_lag"] = df["prod"].reindex(src).to_numpy()
    out["days_since_op"] = (day - src.normalize()).days.to_numpy()
    return out


def build(df, use_plan=True):
    X = pd.DataFrame(index=df.index)
    h, dow = df["hour"], df["dow"]
    X["hour"], X["dow"], X["month"] = h, dow, df["month"]
    X["hour_sin"], X["hour_cos"] = np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24)
    X["is_weekend"] = (dow >= 5).astype(int)
    X["is_holiday"] = df.index.normalize().isin(HOLIDAYS).astype(int)
    X["is_offday"] = X[["is_weekend", "is_holiday"]].max(axis=1)
    X["is_monday_am"] = ((dow == 0) & (h.between(7, 11))).astype(int)   # 주초 기동 피크
    X["tariff"], X["labor_rate"] = df["tariff"], df["labor_rate"]

    for k in (24, 48, 168):
        X[f"power_lag{k}"] = df["power"].shift(k)
        X[f"peak_lag{k}"] = df["peak15"].shift(k)
    X["power_prevday_mean"] = _day_lag(df, "power", 1, "mean")
    X["power_prevday_max"] = _day_lag(df, "peak15", 1, "max")
    X["power_prevweek_mean"] = df["power"].shift(24).rolling(168, min_periods=24).mean()
    X["intrahour_ramp_lag24"] = (df["q60"] - df["q15"]).shift(24)

    for c in ["temp", "humid", "wind", "rain"]:
        X[c] = df[c]
    X["temp_day_max"] = df["temp"].groupby(df.index.normalize()).transform("max")
    X["cdd"] = (df["temp"] - 24).clip(lower=0)      # 냉방부하
    X["hdd"] = (5 - df["temp"]).clip(lower=0)       # 난방부하

    if use_plan:
        day = df.index.normalize()
        X["prod"] = df["prod"]
        X["prod_day"] = df["prod"].groupby(day).transform("sum")
        X["prod_share"] = (df["prod"] / X["prod_day"].replace(0, np.nan)).fillna(0)
        X["prod_prev_h"] = df["prod"].groupby(day).shift(1).fillna(0)
        X["prod_next_h"] = df["prod"].groupby(day).shift(-1).fillna(0)   # 당일 계획이므로 다음 시간 값도 사용 가능
        X["log_prod"] = np.log1p(df["prod"])
        # 시간 단위 운전 스케줄: 가동 시작/종료 시각, 야간조 유무, 주변 3시간 계획량
        active = df["prod"].fillna(0) > 0
        hrs = df["hour"].where(active)
        X["first_active_h"] = hrs.groupby(day).transform("min")
        X["last_active_h"] = hrs.groupby(day).transform("max")
        X["before_start"] = (df["hour"] < X["first_active_h"]).astype(int)
        X["after_end"] = (df["hour"] > X["last_active_h"]).astype(int)
        X["night_shift"] = df["prod"].where(df["hour"] < 7, 0).groupby(day).transform("sum")
        X["prod_roll3"] = df["prod"].groupby(day).transform(lambda s: s.rolling(3, center=True, min_periods=1).mean())
        # 운전 레짐: 생산계획이 있으면 가동일(원본 데이터에서 예외 0건). 계획 누락일은 NaN 유지
        plan_on = (X["prod_day"] > 0).astype(float)
        X["plan_on_day"] = plan_on.mask(df["plan_missing"])
        X["planned_shutdown"] = ((plan_on == 0) & (X["is_offday"] == 0)).astype(float).mask(df["plan_missing"])
        X = X.join(_last_operating_day_lags(df, plan_on.mask(df["plan_missing"], 1.0)))
    return X
