"""한전 산업용(을) 고압A 요금 계산 (전력량요금 + 기본요금 + 래칫).

출처: 한전 산업용(을) 고압A 요금표(2013-11-21 시행, 2021년 기본 단가 동일).
      기후환경요금·연료비조정요금·부가세·전력기금은 모든 kWh에 같은 단가로 붙으므로
      시간대 이동의 비용 차이에는 영향이 없어 제외한다.
가정:
  - 데이터의 15분 값(q15..q60)을 그 15분 동안의 평균 수요(kW)로 본다 → 15분 전력량 = 값 × 0.25 kWh
  - 선택요금은 Ⅱ(기본)
  - 기본요금 적용전력 = 검침 당월 포함 직전 12개월 중 동계(12·1·2월)·하계(7·8·9월)·당월의
    최대수요전력 중 최댓값(래칫). 2020년 12월 자료가 없어 데이터 안의 달만 쓴다.
  - 최대수요전력은 경부하 시간대를 제외한 15분 수요의 최댓값
  - 일요일·공휴일은 전일 경부하, 토요일은 최대부하 시간의 전력량요금을 중간부하 단가로
"""
import numpy as np
import pandas as pd

from .features import HOLIDAYS

BASE_RATE = {"I": 7220, "II": 8320, "III": 9810}          # 원/kW·월
ENERGY = {  # (경부하, 중간부하, 최대부하) 원/kWh
    "I": {"summer": (61.6, 114.5, 196.6), "springfall": (61.6, 84.1, 114.8), "winter": (68.6, 114.7, 172.2)},
    "II": {"summer": (56.1, 109.0, 191.1), "springfall": (56.1, 78.6, 109.3), "winter": (63.1, 109.2, 166.7)},
    "III": {"summer": (55.2, 108.4, 178.7), "springfall": (55.2, 77.3, 101.0), "winter": (62.5, 108.6, 155.5)},
}
RATCHET_MONTHS = (12, 1, 2, 7, 8, 9)
OFF, MID, PEAK = 0, 1, 2
QUARTERS = ["q15", "q30", "q45", "q60"]


def season(month):
    if 6 <= month <= 8:
        return "summer"
    return "winter" if month in (11, 12, 1, 2) else "springfall"


def band(ts, for_energy=True):
    """시각(시간 단위 Timestamp)의 부하 구간. 15분 슬롯은 모두 같은 시(hour)의 구간을 따른다."""
    day = ts.normalize()
    if day in HOLIDAYS or ts.dayofweek == 6 or ts.hour < 9 or ts.hour >= 23:
        return OFF
    h = ts.hour
    if season(ts.month) == "winter":
        peak = 10 <= h < 12 or 17 <= h < 20 or h == 22
    else:
        peak = 10 <= h < 12 or 13 <= h < 17
    if peak and not (for_energy and ts.dayofweek == 5):
        return PEAK
    return MID


def hourly_bands(index):
    return pd.DataFrame({"energy_band": [band(t, True) for t in index],
                         "demand_band": [band(t, False) for t in index]}, index=index)


def energy_rate(index, option="II"):
    b = hourly_bands(index)["energy_band"].to_numpy()
    return np.array([ENERGY[option][season(t.month)][k] for t, k in zip(index, b)])


def energy_cost(q, option="II"):
    """q: 시간 인덱스, 열 q15..q60 (kW). 반환: 시간별 전력량요금(원)."""
    kwh = q[QUARTERS].sum(axis=1) * 0.25
    return pd.Series(kwh.to_numpy() * energy_rate(q.index, option), index=q.index)


def demand_peak(q):
    """경부하를 제외한 시간의 15분 최대수요 (시간별 Series, 경부하 시간은 0)."""
    bands = hourly_bands(q.index)["demand_band"].to_numpy()
    return pd.Series(np.where(bands > OFF, q[QUARTERS].max(axis=1).to_numpy(), 0.0), index=q.index)


def monthly_bill(q, option="II"):
    """월별 전력량요금·당월 최대수요·래칫 적용전력·기본요금."""
    e = energy_cost(q, option).resample("MS").sum()
    dm = demand_peak(q).resample("MS").max()
    rows = []
    for m, cur in dm.items():
        window = dm[(dm.index > m - pd.DateOffset(months=12)) & (dm.index <= m)]
        ratchet = window[window.index.month.isin(RATCHET_MONTHS)]
        billed = max([cur] + list(ratchet.values))
        rows.append(dict(month=m.strftime("%Y-%m"), energy_kwh=float((q.loc[q.index.to_period("M") == m.to_period("M"), QUARTERS].sum().sum()) * 0.25),
                         energy_won=float(e[m]), month_peak_kw=float(cur), billing_kw=float(billed),
                         ratchet_binding=bool(billed > cur), base_won=float(billed * BASE_RATE[option])))
    out = pd.DataFrame(rows).set_index("month")
    out["total_won"] = out["energy_won"] + out["base_won"]
    return out


def ratchet_floor(q, upto_month):
    """해당 월 이전(당월 제외)까지 래칫 대상 월의 최대수요. 당월 피크가 이 값을 넘어야 기본요금이 오른다."""
    dm = demand_peak(q).resample("MS").max()
    m = pd.Timestamp(upto_month).to_period("M").to_timestamp()
    prev = dm[(dm.index < m) & (dm.index > m - pd.DateOffset(months=12)) & dm.index.month.isin(RATCHET_MONTHS)]
    return float(prev.max()) if len(prev) else 0.0


def day_cost(q_day, floor_kw, labor_rate=None, prod=None, option="II", months_ahead=12, labor_won=0.0):
    """하루 계획의 비용(원): 전력량요금 + 래칫 인상 비용 + (선택) 인건비 할증.

    래칫 인상 비용 = 기본요금 단가 × max(0, 당일 최대수요 − 기존 바닥) × 영향 개월 수.
    한 번 오른 적용전력은 최대 12개월 기본요금에 반영되므로 기본값 12개월.
    labor_won: 생산 1단위당 인건비(원). 할증 배수(labor_rate, 1.0/1.5)를 곱해 더한다.
    """
    e = float(energy_cost(q_day, option).sum())
    pk = float(demand_peak(q_day).max())
    ratchet = BASE_RATE[option] * max(0.0, pk - floor_kw) * months_ahead
    labor = 0.0
    if labor_won and prod is not None and labor_rate is not None:
        labor = float((np.asarray(prod) * np.asarray(labor_rate)).sum() * labor_won)
    return dict(energy_won=e, peak_kw=pk, ratchet_won=ratchet, labor_won=labor, total_won=e + ratchet + labor)


def main():
    from pathlib import Path

    from .data import load
    out = Path(__file__).resolve().parents[1] / "outputs" / "tariff"
    out.mkdir(parents=True, exist_ok=True)
    df = load()
    q = df[QUARTERS]
    bill = monthly_bill(q)
    bill.round(1).to_csv(out / "monthly_bill.csv", encoding="utf-8-sig")
    dp = demand_peak(q)
    hb = hourly_bands(q.index)
    # 15분 수요 190 이상 시간이 요금상 어느 구간에 있는지
    ev = df["peak15"] >= 190
    share = hb.loc[ev, "demand_band"].map({OFF: "경부하(기본요금 무관)", MID: "중간부하", PEAK: "최대부하"}).value_counts()
    share.to_csv(out / "tau_events_by_band.csv", encoding="utf-8-sig")
    print(bill.round(0).to_string())
    print("\n190 이상 시간의 요금 구간:", share.to_dict())
    print("요금상 최대수요 시각:", dp.idxmax(), dp.max())


if __name__ == "__main__":
    main()
