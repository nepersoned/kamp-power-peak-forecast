"""당일 실시간 수요 제어 강화학습 환경 (gymnasium).

에피소드 = 가동일 하루. 매 시각 h에서 "이번 시간 계획 물량 중 얼마를 지금 돌릴지" 고른다.
  행동 0/1/2 = 이번 시간 신규 물량의 0% / 50% / 100% 가동(미룬 물량은 대기열로).
  현실 제약: 미룬 물량은 최대 MAX_DELAY시간 안에 반드시 처리, 시간당 생산은 원래 계획 최대치 × CAP_MULT 이하,
            가동 시작 횟수가 원래 계획보다 늘면 큰 벌점(한 시간씩 켜고 끄는 펄스 운전 방지),
            23시에는 남은 물량 전부 처리(일 생산 총량 보존). → "가동 시각을 최대 2시간 늦추는 시차 기동"만 허용.
  (제약 없이 탐색하면 하루 생산을 한 시간에 몰아 시뮬레이터의 학습 범위 밖을 파고드는 것을 확인)
시뮬레이터: 학습된 예측 모델(시간별 평균전력·15분 최대수요)에 검증 가동일의 하루 잔차 경로를 통째로 표본 추출해 더한다.
보상: 에피소드 끝에 (원래 계획 비용 − 새 계획 비용) / 1000  [천 원]. 같은 잔차 표본으로 비교(공통 난수).
비용: 전력량요금(시간대별 단가) + 래칫 인상 비용(기본요금 × 초과분 × 12개월) + 인건비 할증(선택).

주의: 시뮬레이터는 관측 데이터로 학습한 연관 모델이다. 결과는 모델 기반 가상 추정이며 실측 효과가 아니다.
"""
import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces

from . import tariff as T

FRACTIONS = np.array([0.0, 0.5, 1.0])
MAX_DELAY = 2
CAP_MULT = 1.0
START_PENALTY_WON = 300_000   # 원래 계획보다 가동 시작(켜고 끄기)이 늘어날 때 1회당 벌점 — 사실상 금지


def n_starts(p):
    on = np.asarray(p) > 0
    return int((on & ~np.r_[False, on[:-1]]).sum())
PLAN_COLS = ["prod", "prod_day", "prod_share", "prod_prev_h", "prod_next_h", "log_prod", "first_active_h",
             "last_active_h", "before_start", "after_end", "night_shift", "prod_roll3"]


def plan_block(Xday, prod):
    """하루(24행) 피처에서 생산계획 파생 열만 새 계획으로 다시 계산 (features.build와 동일한 정의)."""
    X = Xday.copy()
    p = np.asarray(prod, float)
    h = np.arange(24)
    tot = p.sum()
    X["prod"] = p
    X["prod_day"] = tot
    X["prod_share"] = p / tot if tot > 0 else 0.0
    X["prod_prev_h"] = np.r_[0.0, p[:-1]]
    X["prod_next_h"] = np.r_[p[1:], 0.0]
    X["log_prod"] = np.log1p(p)
    act = h[p > 0]
    first, last = (act.min(), act.max()) if len(act) else (np.nan, np.nan)
    X["first_active_h"] = first
    X["last_active_h"] = last
    X["before_start"] = (h < first).astype(int)
    X["after_end"] = (h > last).astype(int)
    X["night_shift"] = p[h < 7].sum()
    X["prod_roll3"] = pd.Series(p).rolling(3, center=True, min_periods=1).mean().to_numpy()
    return X


class DayData:
    """에피소드에 필요한 하루치 고정 정보."""

    def __init__(self, df, X, day, floor_kw):
        idx = pd.date_range(day, periods=24, freq="h")
        self.day, self.idx = day, idx
        self.X = X.loc[idx]
        self.prod = df.loc[idx, "prod"].fillna(0).to_numpy(float)
        self.labor = df.loc[idx, "labor_rate"].to_numpy(float)
        self.rate = T.energy_rate(idx)
        self.demand_band = T.hourly_bands(idx)["demand_band"].to_numpy()
        self.energy_band = T.hourly_bands(idx)["energy_band"].to_numpy()
        self.floor = floor_kw
        self.hour = np.arange(24)


class Simulator:
    """계획(24시간 생산량) → 시간별 평균전력·15분 최대수요."""

    def __init__(self, power_model, peak_model, resid_power_days, resid_peak_days):
        """resid_*_days: (날 수, 24) 가동일 하루 잔차 경로."""
        self.pm, self.qm = power_model, peak_model
        self.rp, self.rq = np.asarray(resid_power_days), np.asarray(resid_peak_days)

    def expected(self, dd, prod):
        Xn = plan_block(dd.X, prod)
        hr = pd.Series(dd.hour, index=dd.idx)
        return self.pm.predict(Xn, hr), self.qm.predict(Xn, hr)

    def noise(self, rng):
        i = rng.integers(len(self.rp))
        return self.rp[i], self.rq[i]

    def cost(self, dd, prod, noise=None, labor_won=0.0, months_ahead=12):
        pw, pk = self.expected(dd, prod)
        if noise is not None:
            pw, pk = pw + noise[0], pk + noise[1]
        pw, pk = np.clip(pw, 0, None), np.clip(pk, 0, None)
        energy = float((pw * dd.rate).sum())
        demand = float(np.max(np.where(dd.demand_band > T.OFF, pk, 0.0)))
        ratchet = T.BASE_RATE["II"] * max(0.0, demand - dd.floor) * months_ahead
        labor = float((np.asarray(prod) * (dd.labor - 1.0)).sum() * labor_won)  # 할증분만
        extra = max(0, n_starts(prod) - n_starts(dd.prod))
        switch = START_PENALTY_WON * extra
        return dict(total=energy + ratchet + labor + switch, energy=energy, ratchet=ratchet, labor=labor,
                    switch=switch, extra_starts=extra, demand=demand, peak_all=float(pk.max()))


class PeakControlEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, days, sim, labor_won=0.0, seed=0, cap_mult=CAP_MULT, max_delay=MAX_DELAY, shaped=False):
        """shaped=True: 학습용. 노이즈 없는 기대 비용으로 매 시각 잠재함수 기반 보상
        r_t = Φ(s_{t+1}) − Φ(s_t),  Φ = −기대비용(지금까지 실행 + 남은 물량은 다음 시간/원래 계획대로) / 1000.
        합계는 하루 기대 절감액과 같다(telescoping). 평가(shaped=False)는 에피소드 끝 노이즈 경로 비용."""
        super().__init__()
        self.shaped = shaped
        self.days, self.sim, self.labor_won = days, sim, labor_won
        self.rng = np.random.default_rng(seed)
        self.cap_mult, self.max_delay = cap_mult, max_delay
        self.action_space = spaces.Discrete(len(FRACTIONS))
        self.observation_space = spaces.Box(-5, 5, shape=(19,), dtype=np.float32)

    # ----- 에피소드 -----
    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        i = options.get("day_index") if options and "day_index" in options else self.rng.integers(len(self.days))
        self.dd = self.days[i]
        self.noise = self.sim.noise(self.rng)
        self.plan = self.dd.prod.copy()
        self.done_prod = np.zeros(24)
        self.queue = []          # [(생성 시각, 물량)]
        self.h = 0
        self.cap = max(self.plan.max() * self.cap_mult, 1.0)
        self.scale = max(self.plan.max(), 1.0)
        self.base_pw, self.base_pk = self.sim.expected(self.dd, self.plan)
        if self.shaped:
            self.phi = -self.sim.cost(self.dd, self.plan, None, self.labor_won)["total"] / 1000
        return self._obs(), {}

    def _tentative(self):
        """h시까지 실행분 + 대기열은 다음 시간에 + 이후는 원래 계획."""
        p = self.done_prod.copy()
        if self.h < 24:
            p[self.h:] = self.plan[self.h:]
            p[self.h] += self.backlog
        return p

    def _obs(self):
        h, dd = self.h, self.dd
        nxt = lambda a, k: a[h + k] if h + k < 24 else 0.0
        o = [h / 23, self.backlog / self.scale, dd.prod[h] / self.scale,
             nxt(dd.prod, 1) / self.scale, nxt(dd.prod, 2) / self.scale, dd.prod[h + 1:].sum() / (24 * self.scale),
             *(np.eye(3)[dd.energy_band[h]]), float(nxt(dd.energy_band, 1) == T.PEAK),
             dd.rate[h] / 200, dd.labor[h] - 1.0, float(nxt(dd.labor, 1) - 1.0),
             self.base_pk[h] / 200, nxt(self.base_pk, 1) / 200, self.base_pk.max() / 200, dd.floor / 200,
             float(dd.demand_band[h] > T.OFF), float(h >= 18)]
        return np.asarray(o, dtype=np.float32)

    @property
    def backlog(self):
        return float(sum(a for _, a in self.queue))

    def step(self, action):
        h = self.h
        self.done_prod[h], self.queue = schedule_step(h, self.plan[h], self.queue, int(action), self.cap, self.max_delay)
        self.h += 1
        if self.shaped:
            phi = -self.sim.cost(self.dd, self._tentative(), None, self.labor_won)["total"] / 1000
            r, self.phi = phi - self.phi, phi
            if self.h < 24:
                return self._obs(), r, False, False, {}
            return np.zeros(self.observation_space.shape, np.float32), r, True, False, {}
        if self.h < 24:
            return self._obs(), 0.0, False, False, {}
        base = self.sim.cost(self.dd, self.plan, self.noise, self.labor_won)
        new = self.sim.cost(self.dd, self.done_prod, self.noise, self.labor_won)
        reward = (base["total"] - new["total"]) / 1000
        info = dict(base=base, new=new, moved=float(np.abs(self.done_prod - self.plan).sum() / 2),
                    day_prod=float(self.plan.sum()), day=str(self.dd.day.date()))
        return np.zeros(self.observation_space.shape, np.float32), reward, True, False, info


def schedule_step(h, new, queue, action, cap, max_delay):
    """한 시각의 실행량과 갱신된 대기열. 기한이 된 물량은 강제 처리, 나머지는 행동 비율만큼(오래된 것부터) 처리."""
    queue = list(queue)
    if new > 0:
        queue.append((h, float(new)))
    forced = sum(a for t, a in queue if h - t >= max_delay) if h < 23 else sum(a for _, a in queue)
    free = sum(a for t, a in queue if h - t < max_delay) if h < 23 else 0.0
    run = forced + FRACTIONS[action] * free
    run = max(forced, min(run, cap))            # 강제분은 상한을 넘어도 처리
    left = run
    out = []
    for t, a in sorted(queue):
        take = min(a, left)
        left -= take
        if a - take > 1e-9:
            out.append((t, a - take))
    return run, out


def rollout_plan(plan, acts, cap, max_delay=MAX_DELAY):
    done, queue = np.zeros(24), []
    for h in range(24):
        done[h], queue = schedule_step(h, plan[h], queue, int(acts[h]), cap, max_delay)
    return done


# ----- 기준 정책 -----
def policy_noop(obs, env):
    return 2


def policy_peak_rule(obs, env):
    """최대부하 시간에는 절반만 돌리고, 나머지는 다음 시간으로(다음 시간은 전량 가동)."""
    h = env.h
    return 1 if env.dd.energy_band[h] == T.PEAK else 2


def oracle_plan(env, dd, iters=2, labor_won=0.0):
    """완전 정보(노이즈 없는 시뮬레이터)로 시간별 행동을 좌표 하강 탐색 — 최적화 상한 기준선."""
    acts = np.full(24, 2)
    cap = max(dd.prod.max() * env.cap_mult, 1.0)

    def rollout(acts):
        return rollout_plan(dd.prod, acts, cap, env.max_delay)

    best = env.sim.cost(dd, rollout(acts), None, labor_won)["total"]
    for _ in range(iters):
        for h in range(23):
            if dd.prod[h] == 0:
                continue
            for a in (0, 1):
                trial = acts.copy(); trial[h] = a
                c = env.sim.cost(dd, rollout(trial), None, labor_won)["total"]
                if c < best - 1e-6:
                    best, acts = c, trial
    return acts


def run_policy(env, policy, n_days, seeds=(0, 1, 2), oracle=False, labor_won=0.0):
    rows = []
    for s in seeds:
        for i in range(n_days):
            obs, _ = env.reset(seed=1000 * s + i, options={"day_index": i})
            acts = oracle_plan(env, env.dd, labor_won=labor_won) if oracle else None
            done = False
            while not done:
                a = acts[env.h] if oracle else policy(obs, env)
                obs, r, done, _, info = env.step(a)
            rows.append(dict(seed=s, day=info["day"], reward_kwon=r, moved=info["moved"], day_prod=info["day_prod"],
                             base_total=info["base"]["total"], new_total=info["new"]["total"],
                             base_energy=info["base"]["energy"], new_energy=info["new"]["energy"],
                             base_demand=info["base"]["demand"], new_demand=info["new"]["demand"],
                             base_ratchet=info["base"]["ratchet"], new_ratchet=info["new"]["ratchet"],
                             new_labor=info["new"]["labor"], extra_starts=info["new"]["extra_starts"]))
    return pd.DataFrame(rows)
