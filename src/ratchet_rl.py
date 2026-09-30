"""래칫 갱신 위험을 다루는 당일 제어 강화학습 (드문 사건 합성 학습).

배경: 7월 완전 예측 절감의 77%가 7/19 한 번의 래칫 갱신 피크(약 163만 원)에서 나온다.
      이런 날은 데이터에 몇 번뿐이라 실제 데이터만으로는 강화학습이 배울 수 없다.
방법:
  - 세계 모형 = 결정 기반 평가와 동일: 실제 전력(새 계획) = 실측 + 대리모형 변화분
  - 학습용 날: 학습 구간 원본 가동일 + 날짜 단위 교차적합(out-of-fold) 예측(평균·P75)
  - 드문 사건 합성: 래칫 바닥을 그날 실제 최대수요 + U(−20, +15) kW로 무작위 → "넘을까 말까"인 날을 대량 생성
  - 관측: 시각, 대기 물량, 계획, 요금 구간, 예측·P75 상한과 바닥의 거리, 오늘 이미 찍힌 최대수요, 최근 예측 편향
  - 보상: 에피소드 끝 실제 절감 / 10만 원
평가: 실제 7월(검증, 실제 래칫 바닥)과 테스트 기간에서 무조치·MILP(평균/P75)·시간별 재최적화와 비교.
"""
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from sklearn.model_selection import GroupKFold

from . import tariff as T
from .decision_eval import true_cost
from .milp import _S_numeric, solve_day
from .rl_env import CAP_MULT, MAX_DELAY, schedule_step

OUT = Path(__file__).resolve().parents[1] / "outputs" / "ratchet_rl"
SCALE_WON = 100_000


@dataclass
class Day:
    day: pd.Timestamp
    idx: pd.DatetimeIndex
    prod: np.ndarray
    rate: np.ndarray
    labor: np.ndarray
    demand_band: np.ndarray
    energy_band: np.ndarray
    act_pw: np.ndarray
    act_pk: np.ndarray
    fc_pw: np.ndarray
    fc_pk: np.ndarray
    fc_pk75: np.ndarray
    floor: float


def make_day(df, d, fc_pw, fc_pk, fc_pk75, floor):
    idx = pd.date_range(d, periods=24, freq="h")
    b = T.hourly_bands(idx)
    return Day(d, idx, df.loc[idx, "prod"].fillna(0).to_numpy(float), T.energy_rate(idx),
               df.loc[idx, "labor_rate"].to_numpy(float), b["demand_band"].to_numpy(), b["energy_band"].to_numpy(),
               df.loc[idx, "power"].to_numpy(float), df.loc[idx, "peak15"].to_numpy(float),
               fc_pw.reindex(idx).to_numpy(float), fc_pk.reindex(idx).to_numpy(float),
               fc_pk75.reindex(idx).to_numpy(float), float(floor))


def realized_peak(day, plan, coefs, upto):
    """0..upto-1 시각의 실제 15분 최대수요(새 계획 기준), 기본요금 산정 시간만."""
    dpk = _S_numeric(plan, coefs["peak15"]) - _S_numeric(day.prod, coefs["peak15"])
    pk = day.act_pk + dpk
    m = [pk[t] for t in range(upto) if day.demand_band[t] > T.OFF]
    return max(m) if m else 0.0


class RatchetEnv(gym.Env):
    def __init__(self, days, coefs, seed=0, randomize_floor=True, floor_range=(-20.0, 15.0)):
        super().__init__()
        self.days, self.coefs = days, coefs
        self.rng = np.random.default_rng(seed)
        self.randomize_floor, self.floor_range = randomize_floor, floor_range
        self.action_space = spaces.Discrete(3)
        self.observation_space = spaces.Box(-10, 10, shape=(20,), dtype=np.float32)

    def reset(self, seed=None, options=None):
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        i = options["day_index"] if options and "day_index" in options else int(self.rng.integers(len(self.days)))
        d = self.days[i]
        if self.randomize_floor:
            true_max = realized_peak(d, d.prod, self.coefs, 24)
            d = Day(**{**d.__dict__, "floor": true_max + float(self.rng.uniform(*self.floor_range))})
        self.d = d
        self.plan = d.prod.copy()
        self.done = np.zeros(24)
        self.queue = []
        self.h = 0
        self.cap = max(d.prod.max() * CAP_MULT, 1.0)
        self.scale = max(d.prod.max(), 1.0)
        return self._obs(), {}

    def _obs(self):
        d, h = self.d, self.h
        rem = [t for t in range(h, 24) if d.demand_band[t] > T.OFF]
        fc_max = max((d.fc_pk[t] for t in rem), default=0.0)
        fc75_max = max((d.fc_pk75[t] for t in rem), default=0.0)
        tentative = self.done.copy()
        tentative[h:] = self.plan[h:]
        seen = realized_peak(d, tentative, self.coefs, max(h - 1, 0))       # h−1시까지 관측(다음 시각 가동 여부 반영)
        lo = max(0, h - 3)
        bias = float(np.mean(d.act_pk[lo:max(h - 1, lo + 1)] - d.fc_pk[lo:max(h - 1, lo + 1)])) if h >= 2 else 0.0
        nxt = lambda a, k: a[h + k] if h + k < 24 else 0.0
        backlog = sum(a for _, a in self.queue)
        o = [h / 23, backlog / self.scale, d.prod[h] / self.scale, nxt(d.prod, 1) / self.scale,
             *(np.eye(3)[d.energy_band[h]]), float(d.demand_band[h] > T.OFF), float(nxt(d.demand_band, 1) > T.OFF),
             d.rate[h] / 200, (d.fc_pk[h] - d.floor) / 20, (d.fc_pk75[h] - d.floor) / 20,
             (nxt(d.fc_pk75, 1) - d.floor) / 20, (fc_max - d.floor) / 20, (fc75_max - d.floor) / 20,
             (seen - d.floor) / 20 if seen > 0 else -2.0, bias / 20, d.floor / 200, float(d.labor[h] > 1), len(rem) / 24]
        return np.clip(np.asarray(o, np.float32), -10, 10)

    def mask(self, action):
        """행동 차단: 직전 시간에 가동했고 이번 시간 강제 처리분이 없으면 '0%(끄기)'를 50%로 바꾼다.
        가동 중 끊김(시작 횟수 증가)을 벌점이 아니라 구조적으로 막는다. 가동 시작을 늦추는 것은 허용."""
        h = self.h
        if int(action) != 0 or h == 0 or self.done[h - 1] <= 0:
            return int(action)
        queue = list(self.queue) + ([(h, float(self.plan[h]))] if self.plan[h] > 0 else [])
        forced = sum(a for t, a in queue if h - t >= MAX_DELAY)
        free = sum(a for t, a in queue if h - t < MAX_DELAY)
        return 1 if forced <= 0 and free > 0 else 0

    def step(self, action):
        h = self.h
        action = self.mask(action)
        self.done[h], self.queue = schedule_step(h, self.plan[h], self.queue, int(action), self.cap, MAX_DELAY)
        self.h += 1
        if self.h < 24:
            return self._obs(), 0.0, False, False, {}
        d = self.d
        base = true_cost(d, d.act_pw, d.act_pk, d.prod, self.coefs)
        new = true_cost(d, d.act_pw, d.act_pk, self.done, self.coefs)
        info = dict(saving=base["total"] - new["total"], energy=base["energy"] - new["energy"],
                    ratchet=base["ratchet"] - new["ratchet"], day=str(d.day.date()), floor=d.floor)
        return np.zeros(self.observation_space.shape, np.float32), info["saving"] / SCALE_WON, True, False, info


# ---------- 데이터 준비 ----------
def oof_forecasts(df, X, mask, n_splits=5):
    """학습 구간 날짜 단위 교차적합 예측(평균·P75): 학습용 날에 '틀릴 수 있는 예측'을 붙인다."""
    from .models import RegimeModel, quantile_upper
    idx = df.index[mask]
    groups = idx.normalize().factorize()[0]
    out = {k: pd.Series(np.nan, idx) for k in ("pw", "pk", "pk75")}
    copy = df["is_copy"].to_numpy()[mask]
    Xm, hm = X[mask], df["hour"][mask]
    for a, b in GroupKFold(n_splits).split(Xm, groups=groups):
        for key, tgt, mk in [("pw", "power", RegimeModel), ("pk", "peak15", RegimeModel),
                             ("pk75", "peak15", lambda: quantile_upper(0.75))]:
            y = df[tgt][mask]
            m = mk().fit(Xm.iloc[a], y.iloc[a], hm.iloc[a], copy[a])
            out[key].iloc[b] = m.predict(Xm.iloc[b], hm.iloc[b])
    return out


def build_days():
    import warnings
    warnings.filterwarnings("ignore")
    import run_all
    from .data import load
    from .features import build
    from .milp import fit_surrogate
    from .models import RegimeModel, operating, quantile_upper
    from .run_rl import END, START, TEST0, VAL0, day_floor

    df = load()
    X = build(df)
    run_all.COPY = df["is_copy"].to_numpy()
    tr, val, te = run_all.masks(df, X)
    t = df.index
    copy = df["is_copy"].to_numpy()
    sur_mask = operating(X) & ~copy & (~df["outage"]).to_numpy() & (t >= START) & (t < TEST0)
    coefs = fit_surrogate(df, sur_mask)
    q = df[T.QUARTERS]
    prod_day = df["prod"].fillna(0).groupby(t.normalize()).sum()

    def full_days(mask):
        ds = []
        for d in sorted(set(t[mask].normalize())):
            rows = t.get_indexer(pd.date_range(d, periods=24, freq="h"))
            if (rows < 0).any() or not mask[rows].all() or prod_day.get(d, 0) <= 0 or copy[rows].any():
                continue
            ds.append(d)
        return ds

    # 학습용: 학습 구간 원본 가동일 + 교차적합 예측
    oof = oof_forecasts(df, X, tr)
    train_days = [make_day(df, d, oof["pw"], oof["pk"], oof["pk75"], day_floor(q, d)) for d in full_days(tr)]

    def period_days(fit_mask, ev_mask):
        f = {}
        for key, tgt, mk in [("pw", "power", RegimeModel), ("pk", "peak15", RegimeModel),
                             ("pk75", "peak15", lambda: quantile_upper(0.75))]:
            p, _ = run_all.fit_predict(mk, X, df[tgt], df["hour"], fit_mask, ev_mask)
            f[key] = pd.Series(p, t[ev_mask])
        return [make_day(df, d, f["pw"], f["pk"], f["pk75"], day_floor(q, d)) for d in full_days(ev_mask)]

    val_days = period_days(tr, val)
    test_days = period_days(tr | val, te)
    return coefs, train_days, val_days, test_days


# ---------- 평가 ----------
def evaluate(days, coefs, policies):
    rows = []
    for i, d in enumerate(days):
        base = true_cost(d, d.act_pw, d.act_pk, d.prod, coefs)
        for name, pol in policies.items():
            plan = pol(i, d)
            c = true_cost(d, d.act_pw, d.act_pk, plan, coefs)
            rows.append(dict(day=str(d.day.date()), policy=name, floor=d.floor, saving_won=base["total"] - c["total"],
                             energy_saving_won=base["energy"] - c["energy"], ratchet_saving_won=base["ratchet"] - c["ratchet"],
                             moved_share=float(np.abs(plan - d.prod).sum() / 2 / max(d.prod.sum(), 1))))
    return pd.DataFrame(rows)


def protect_rule_plan(d, margin=5.0):
    """현장용 보호 규칙: P75 상한 예측이 (래칫 바닥 − margin)을 넘는 기본요금 시간에만
    그 시간 생산의 50%를 미룬다(다음 시간 이후 처리). 그 외에는 원래 계획."""
    risky = (d.demand_band > T.OFF) & (d.fc_pk75 >= d.floor - margin)
    queue, done = [], np.zeros(24)
    cap = max(d.prod.max() * CAP_MULT, 1.0)
    for h in range(24):
        done[h], queue = schedule_step(h, d.prod[h], queue, 1 if risky[h] else 2, cap, MAX_DELAY)
    return done


def rl_plan(model, env, i):
    obs, _ = env.reset(options={"day_index": i})
    done = False
    while not done:
        obs, _, done, _, _ = env.step(int(model.predict(obs, deterministic=True)[0]))
    return env.done.copy()


def summarize(res, n_boot=2000, seed=0):
    rng = np.random.default_rng(seed)
    rows = []
    for name, g in res.groupby("policy", sort=False):
        s = g["saving_won"].to_numpy()
        boot = [rng.choice(s, len(s)).mean() for _ in range(n_boot)]
        rows.append(dict(policy=name, days=len(g), saving_won_per_day=s.mean(),
                         ci95=f"[{np.percentile(boot, 2.5):,.0f}, {np.percentile(boot, 97.5):,.0f}]",
                         energy_won=g["energy_saving_won"].mean(), ratchet_won=g["ratchet_saving_won"].mean(),
                         worst_day_won=s.min(), moved_share=g["moved_share"].mean()))
    return pd.DataFrame(rows).round(1)


def main(timesteps=300_000, seed=0, eval_only=False):
    from stable_baselines3 import PPO
    OUT.mkdir(parents=True, exist_ok=True)
    coefs, train_days, val_days, test_days = build_days()
    print(f"train {len(train_days)} / valid {len(val_days)} / test {len(test_days)} days", flush=True)

    if eval_only:
        model = PPO.load(OUT / f"ppo_ratchet_seed{seed}")
    else:
        env = RatchetEnv(train_days, coefs, seed=seed)
        model = PPO("MlpPolicy", env, n_steps=24 * 128, batch_size=512, gamma=1.0, learning_rate=3e-4, ent_coef=0.01,
                    seed=seed, verbose=0)
        model.learn(total_timesteps=timesteps)
        model.save(OUT / f"ppo_ratchet_seed{seed}")

    results = []
    for period, days in [("valid", val_days), ("test", test_days)]:
        ev = RatchetEnv(days, coefs, randomize_floor=False)
        pols = {
            "무조치": lambda i, d: d.prod.copy(),
            "MILP 평균 예측": lambda i, d: solve_day(d, None, coefs, base=(d.fc_pw, d.fc_pk))[0],
            "MILP P75 상한": lambda i, d: solve_day(d, None, coefs, base=(d.fc_pw, d.fc_pk75))[0],
            "보호 규칙(위험 시간 50% 미루기)": lambda i, d: protect_rule_plan(d),
            "PPO(래칫 합성 학습)": lambda i, d: rl_plan(model, ev, i),
            "완전 예측 MILP(상한)": lambda i, d: solve_day(d, None, coefs, base=(d.act_pw, d.act_pk))[0],
        }
        res = evaluate(days, coefs, pols).assign(period=period)
        results.append(res)
        tab = summarize(res)
        tab.to_csv(OUT / f"compare_{period}_seed{seed}.csv", index=False, encoding="utf-8-sig")
        print(f"\n[{period}]\n{tab.to_string(index=False)}", flush=True)
        if period == "valid":
            print(res[res["day"] == "2021-07-19"][["policy", "saving_won", "ratchet_saving_won"]].to_string(index=False))
    pd.concat(results).to_csv(OUT / f"days_seed{seed}.csv", index=False, encoding="utf-8-sig")

    # 합성 날에서의 비교(학습 분포 안): 무작위 바닥 × 학습일
    syn = RatchetEnv(train_days, coefs, seed=999)
    rows = []
    for k in range(300):
        obs, _ = syn.reset()
        d = syn.d
        base = true_cost(d, d.act_pw, d.act_pk, d.prod, coefs)["total"]
        done = False
        while not done:
            obs, _, done, _, info = syn.step(int(model.predict(obs, deterministic=True)[0]))
        milp75 = base - true_cost(d, d.act_pw, d.act_pk, solve_day(d, None, coefs, base=(d.fc_pw, d.fc_pk75))[0], coefs)["total"]
        milp = base - true_cost(d, d.act_pw, d.act_pk, solve_day(d, None, coefs, base=(d.fc_pw, d.fc_pk))[0], coefs)["total"]
        rows.append(dict(ppo=info["saving"], milp_mean=milp, milp_p75=milp75))
    syn_tab = pd.DataFrame(rows).describe().round(0)
    syn_tab.to_csv(OUT / f"synthetic_seed{seed}.csv", encoding="utf-8-sig")
    print(f"\n[합성 래칫 위험일 300개]\n{syn_tab.loc[['mean', '25%', '50%', 'min']].to_string()}")


# ---------- MILP 시연 사전학습(행동 복제) + PPO 미세조정 ----------
def expert_actions(env, plan):
    """MILP 계획(시간별 실행량)을 환경 행동(0/50/100%)으로 근사해 시연 궤적을 만든다."""
    from .rl_env import FRACTIONS
    obs_list, act_list = [], []
    obs = env._obs()
    for h in range(24):
        queue = list(env.queue)
        if env.plan[h] > 0:
            queue.append((h, float(env.plan[h])))
        forced = sum(a for t, a in queue if h - t >= MAX_DELAY) if h < 23 else sum(a for _, a in queue)
        free = sum(a for t, a in queue if h - t < MAX_DELAY) if h < 23 else 0.0
        frac = (plan[h] - forced) / free if free > 1e-9 else 1.0
        a = env.mask(int(np.argmin(np.abs(FRACTIONS - np.clip(frac, 0, 1)))))
        obs_list.append(obs); act_list.append(a)
        obs, _, done, _, _ = env.step(a)
    return obs_list, act_list


def collect_demos(days, coefs, n_episodes=3000, seed=1):
    env = RatchetEnv(days, coefs, seed=seed)
    O, A = [], []
    for _ in range(n_episodes):
        env.reset()
        d = env.d
        plan, _ = solve_day(d, None, coefs, base=(d.fc_pw, d.fc_pk75))
        o, a = expert_actions(env, plan)
        O += o; A += a
    return np.asarray(O, np.float32), np.asarray(A, np.int64)


def behavior_clone(model, O, A, epochs=30, lr=1e-3, batch=1024, seed=0):
    import torch
    torch.manual_seed(seed)
    pol = model.policy
    opt = torch.optim.Adam(pol.parameters(), lr=lr)
    Ot, At = torch.as_tensor(O), torch.as_tensor(A)
    n = len(Ot)
    for ep in range(epochs):
        perm = torch.randperm(n)
        tot = 0.0
        for i in range(0, n, batch):
            idx = perm[i:i + batch]
            dist = pol.get_distribution(Ot[idx])
            loss = -dist.log_prob(At[idx]).mean()
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(idx)
    with torch.no_grad():
        acc = float((pol.get_distribution(Ot).distribution.probs.argmax(1) == At).float().mean())
    return tot / n, acc


def main_bc(finetune_steps=200_000, seed=0, n_demos=3000):
    from stable_baselines3 import PPO
    OUT.mkdir(parents=True, exist_ok=True)
    coefs, train_days, val_days, test_days = build_days()
    O, A = collect_demos(train_days, coefs, n_demos, seed=seed + 1)
    print(f"demos {len(A)} steps, action dist {np.bincount(A, minlength=3) / len(A)}", flush=True)
    env = RatchetEnv(train_days, coefs, seed=seed)
    model = PPO("MlpPolicy", env, n_steps=24 * 128, batch_size=512, gamma=1.0, learning_rate=1e-4, ent_coef=0.0,
                clip_range=0.1, seed=seed, verbose=0)
    loss, acc = behavior_clone(model, O, A)
    print(f"BC loss {loss:.3f} acc {acc:.3f}", flush=True)
    model.save(OUT / f"bc_seed{seed}")
    bc_model = PPO.load(OUT / f"bc_seed{seed}")
    if finetune_steps > 0:
        model.learn(total_timesteps=finetune_steps)
        model.save(OUT / f"bc_ppo_seed{seed}")

    for period, days in [("valid", val_days), ("test", test_days)]:
        ev = RatchetEnv(days, coefs, randomize_floor=False)
        pols = {
            "MILP 평균 예측": lambda i, d: solve_day(d, None, coefs, base=(d.fc_pw, d.fc_pk))[0],
            "MILP P75 상한(시연자)": lambda i, d: solve_day(d, None, coefs, base=(d.fc_pw, d.fc_pk75))[0],
            "행동 복제(BC)": lambda i, d: rl_plan(bc_model, ev, i),
            "BC + PPO 미세조정": lambda i, d: rl_plan(model, ev, i),
            "완전 예측 MILP(상한)": lambda i, d: solve_day(d, None, coefs, base=(d.act_pw, d.act_pk))[0],
        }
        res = evaluate(days, coefs, pols).assign(period=period)
        tab = summarize(res)
        tab.to_csv(OUT / f"bc_compare_{period}_seed{seed}.csv", index=False, encoding="utf-8-sig")
        print(f"\n[{period}]\n{tab.to_string(index=False)}", flush=True)
        if period == "valid":
            print(res[res["day"] == "2021-07-19"][["policy", "saving_won"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesteps", type=int, default=300_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--eval-only", action="store_true")
    ap.add_argument("--bc", action="store_true", help="MILP 시연 행동 복제 + PPO 미세조정")
    a = ap.parse_args()
    if a.bc:
        main_bc(a.timesteps, a.seed)
    else:
        main(a.timesteps, a.seed, a.eval_only)
