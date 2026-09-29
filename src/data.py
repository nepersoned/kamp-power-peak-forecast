"""원천 CSV 로드 및 정제.

발견된 품질 이슈와 처리:
  1) 2021-07-13, 07-15 이틀(48행)은 '시간' 컬럼이 전력값으로 덮어써져 있음(상관 0.91)
     -> 모든 날짜가 정확히 24행이고, 정상 행에서 '시간'==일내 순번이 100% 일치하므로 순번으로 복원
  2) 풍속·강수량·공장인원 결측(3/1/17건) -> 시간순 선형보간, 인원은 0 (비가동 시간대에 집중)
  3) 2021-08-28~29 전력 0 (17시간) -> 정전/전면휴무로 판단, 학습에서 제외하되 플래그 보존
"""
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "okm_augumented_2021.csv"

RENAME = {
    "날짜": "date_raw", "시간": "hour_raw", "15분": "q15", "30분": "q30", "45분": "q45", "60분": "q60",
    "평균": "power", "생산량": "prod", "기온": "temp", "풍속": "wind", "습도": "humid",
    "강수량": "rain", "전기요금(계절)": "tariff", "day": "dow", "d": "dom", "m": "month",
    "공장인원": "workers", "인건비": "labor_rate",
}
Q = ["q15", "q30", "q45", "q60"]


def load(path=RAW):
    df = pd.read_csv(path).rename(columns=RENAME)
    df["hour"] = df.groupby("date_raw").cumcount()
    df["hour_corrupt"] = df["hour_raw"] != df["hour"]
    df["ts"] = pd.to_datetime(df["date_raw"].astype(str)) + pd.to_timedelta(df["hour"], unit="h")
    df = df.sort_values("ts").set_index("ts")
    assert df.index.is_unique and len(df) == 24 * df["date_raw"].nunique()

    # 시간 컬럼이 깨진 날은 생산량·인원도 전부 0으로 지워져 있음(정상 가동 전력인데 계획 0) -> 결측 처리
    bad_day = df.groupby("date_raw")["hour_corrupt"].transform("any")
    df.loc[bad_day, ["prod", "workers"]] = np.nan
    df["plan_missing"] = bad_day

    for c in ["wind", "rain"]:
        df[c] = df[c].interpolate(limit_direction="both")
    df.loc[~bad_day, "workers"] = df.loc[~bad_day, "workers"].fillna(0.0)

    df["peak15"] = df[Q].max(axis=1)          # 시간 내 15분 최대수요 (한전 최대수요전력 기준 단위)
    df["outage"] = df["power"] == 0
    df["dow"] = df["dow"] - 1                 # 0=월 ... 6=일 (원본 1=월 확인)
    df = df.drop(columns=["date_raw", "hour_raw"])
    df["is_copy"] = flag_copied_days(df)
    return df


def flag_copied_days(df):
    """전력 15분 프로파일(96값)이 앞선 날짜와 완전히 같은 날 = 증강으로 복제된 날."""
    day = df.index.normalize()
    sig = df[Q].groupby(day).apply(lambda x: hash(tuple(x.to_numpy().ravel())))
    first_seen = sig.groupby(sig).transform(lambda s: s.index.min())
    copied = first_seen != sig.index
    return day.map(copied).to_numpy()
