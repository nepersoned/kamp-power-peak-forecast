"""모델 해석: SHAP(전역·의존도·상호작용·개별사례) + DiCE 반사실(피크 저감 조치).

잔차 타깃 모델(y - op_peak_lag 학습)은 오프셋이 선형이므로,
전체 예측 = 오프셋 + f(x) 를 SHAP로 정확히 분해하기 위해 오프셋 기여를 해당 변수(op_*_lag)에 더한다.
"""
import matplotlib

matplotlib.use("Agg")
import dice_ml  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import shap  # noqa: E402

from .peak import TAU  # noqa: E402
from . import plots  # noqa: E402,F401  (한글 폰트 설정)

KOR = {"prod_roll3": "생산량(±1h 평균)", "prod": "생산량", "hour": "시각", "op_peak_lag": "직전가동일 동시각 피크",
       "op_power_lag": "직전가동일 동시각 전력", "workers_roll3": "인원(±1h 평균)", "workers": "인원",
       "peak_lag168": "전주 동시각 피크", "power_lag168": "전주 동시각 전력", "temp": "기온",
       "last_active_h": "가동 종료시각", "first_active_h": "가동 시작시각", "prod_share": "일 생산 중 비중",
       "power_prevweek_mean": "전주 평균전력", "prod_day": "일 생산량", "workers_day": "일 인원",
       "dow": "요일", "temp_day_max": "일 최고기온", "humid": "습도", "prod_next_h": "다음시간 생산량",
       "prod_prev_h": "이전시간 생산량", "night_shift": "야간조 생산량", "is_monday_am": "월요일 오전"}


def kor(c):
    return KOR.get(c, c)


def shap_values(booster, X, offset_col=None):
    """TreeSHAP 값. offset_col 이 있으면 오프셋 기여(중심화)를 그 열에 합산."""
    ex = shap.TreeExplainer(booster)
    sv = ex.shap_values(X)
    base = float(np.atleast_1d(ex.expected_value)[0])
    if offset_col is not None:
        off = X[offset_col].fillna(X[offset_col].median()).to_numpy()
        j = list(X.columns).index(offset_col)
        sv = sv.copy()
        sv[:, j] += off - off.mean()
        base += off.mean()
    return sv, base, ex


def shap_report(booster, X, out, offset_col=None, sample_inter=600, seed=0):
    sv, base, ex = shap_values(booster, X, offset_col)
    names = [kor(c) for c in X.columns]
    imp = pd.Series(np.abs(sv).mean(0), X.columns).sort_values(ascending=False)
    imp.round(3).to_csv(out / "shap_importance.csv", encoding="utf-8-sig")

    plt.figure()
    shap.summary_plot(sv, X, feature_names=names, max_display=15, show=False)
    plt.title("SHAP 요약 (15분 최대수요, 가동일, 테스트)")
    plt.tight_layout(); plt.savefig(out / "fig8_shap_beeswarm.png", bbox_inches="tight"); plt.close()

    top = [c for c in imp.index if c in X.columns][:4]
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6))
    for ax, c in zip(axes, top):
        j = list(X.columns).index(c)
        ax.scatter(X[c], sv[:, j], s=6, c=X["hour"], cmap="viridis", alpha=0.7)
        ax.axhline(0, color="k", lw=0.5)
        ax.set(xlabel=kor(c), ylabel="SHAP", title=kor(c))
    fig.suptitle("주요 변수 의존도 (색 = 시각)")
    fig.tight_layout(); fig.savefig(out / "fig9_shap_dependence.png", bbox_inches="tight"); plt.close(fig)

    # 상호작용 (표본)
    rs = np.random.default_rng(seed)
    idx = rs.choice(len(X), min(sample_inter, len(X)), replace=False)
    iv = ex.shap_interaction_values(X.iloc[idx])
    m = np.abs(iv).mean(0)
    np.fill_diagonal(m, 0)
    cols = list(X.columns)
    pairs = [(cols[i], cols[j], m[i, j] * 2) for i in range(len(cols)) for j in range(i + 1, len(cols))]
    inter = pd.DataFrame(pairs, columns=["feat_a", "feat_b", "mean_abs_interaction"]).sort_values(
        "mean_abs_interaction", ascending=False).head(15)
    inter.round(3).to_csv(out / "shap_interactions_top15.csv", index=False, encoding="utf-8-sig")
    a, b = inter.iloc[0, :2]
    ia, ib = cols.index(a), cols.index(b)
    fig, ax = plt.subplots(figsize=(6, 4))
    sc = ax.scatter(X.iloc[idx][a], iv[:, ia, ib] * 2, c=X.iloc[idx][b], cmap="coolwarm", s=10)
    plt.colorbar(sc, label=kor(b))
    ax.axhline(0, color="k", lw=0.5)
    ax.set(xlabel=kor(a), ylabel="상호작용 SHAP", title=f"최대 상호작용: {kor(a)} × {kor(b)}")
    fig.tight_layout(); fig.savefig(out / "fig10_shap_interaction.png", bbox_inches="tight"); plt.close(fig)
    return sv, base, imp, inter


def waterfall(sv_row, base, x_row, title, path, max_display=10):
    e = shap.Explanation(values=sv_row, base_values=base, data=x_row.to_numpy(),
                         feature_names=[kor(c) for c in x_row.index])
    plt.figure()
    shap.plots.waterfall(e, max_display=max_display, show=False)
    plt.title(title)
    plt.tight_layout(); plt.savefig(path, bbox_inches="tight"); plt.close()


# ---------------- DiCE: 피크 시간 생산계획 반사실 ----------------
CTRL = ["prod", "prod_prev_h", "prod_next_h"]


class ControlWrapper:
    """DiCE가 바꿀 수 있는 '의사결정 변수'만 입력받고, 나머지 피처는 해당 시각의 실제 값으로 고정.
    파생 계획 피처(±1h 평균, 로그, 일 비중)는 의사결정 변수로부터 일관되게 재계산."""

    def __init__(self, predict_fn, x_row, day_prod, cols):
        self.f, self.x, self.day_prod, self.cols = predict_fn, x_row, day_prod, cols

    def _expand(self, Z):
        X = pd.DataFrame(np.repeat(self.x.to_numpy()[None], len(Z), 0), columns=self.cols)
        for c in CTRL:
            X[c] = Z[c].to_numpy(float)
        X["prod_roll3"] = (X["prod_prev_h"] + X["prod"] + X["prod_next_h"]) / 3
        X["log_prod"] = np.log1p(X["prod"])
        dp = self.day_prod + (X["prod"] - self.x["prod"]) + (X["prod_prev_h"] - self.x["prod_prev_h"]) \
            + (X["prod_next_h"] - self.x["prod_next_h"])
        X["prod_day"] = dp
        X["prod_share"] = np.where(dp > 0, X["prod"] / dp, 0)
        return X.astype(float)

    def predict(self, Z):
        return self.f(self._expand(Z))


def dice_peak_actions(predict_fn, X_rows, day_prod, out, n_cf=3, tau=TAU, seed=0):
    """예측 피크 >= tau 인 시간에 대해 '예측 < tau' 가 되는 생산계획 반사실 생성.
    제약: 해당 시간 생산·인원은 감축만, 인접 시간은 증가 허용(이동) — 일 총량은 보고서에서 별도 점검."""
    rows = []
    for ts, x in X_rows.iterrows():
        dp = float(day_prod.loc[ts])
        wrapper = ControlWrapper(predict_fn, x, dp, X_rows.columns)
        base_pred = float(wrapper.predict(pd.DataFrame([x[CTRL]]))[0])
        if base_pred < tau:
            continue
        hi = {c: max(float(x[c]) * 2, 1.0) for c in CTRL}
        rng = {"prod": [0.0, float(x["prod"])],
               "prod_prev_h": [float(x["prod_prev_h"]), hi["prod_prev_h"] + float(x["prod"])],
               "prod_next_h": [float(x["prod_next_h"]), hi["prod_next_h"] + float(x["prod"])]}
        frame = pd.DataFrame([{c: v for c, v in zip(CTRL, vals)} for vals in
                              np.random.default_rng(seed).uniform([r[0] for r in rng.values()],
                                                                  [max(r[1], r[0] + 1e-6) for r in rng.values()],
                                                                  size=(300, len(CTRL)))])
        frame["y"] = wrapper.predict(frame)
        d = dice_ml.Data(dataframe=frame, continuous_features=CTRL, outcome_name="y")
        m = dice_ml.Model(model=wrapper, backend="sklearn", model_type="regressor")
        exp = dice_ml.Dice(d, m, method="random")
        try:
            cf = exp.generate_counterfactuals(pd.DataFrame([x[CTRL].astype(float)]), total_CFs=n_cf,
                                              desired_range=[0, tau - 0.5], features_to_vary=CTRL,
                                              permitted_range=rng, random_seed=seed)
            cfs = cf.cf_examples_list[0].final_cfs_df
        except Exception as e:  # 해가 없는 경우
            rows.append(dict(timestamp=ts, pred=base_pred, found=False, note=str(e)[:80]))
            continue
        if cfs is None or len(cfs) == 0:
            rows.append(dict(timestamp=ts, pred=base_pred, found=False))
            continue
        cfs = cfs.assign(cut_prod_pct=lambda c: (1 - c["prod"] / max(x["prod"], 1e-9)) * 100,
                         moved_to_neighbors=lambda c: (c["prod_prev_h"] - x["prod_prev_h"]) + (c["prod_next_h"] - x["prod_next_h"]))
        best = cfs.sort_values("cut_prod_pct").iloc[0]
        rows.append(dict(timestamp=ts, hour=int(x["hour"]), pred=round(base_pred, 1), found=True,
                         cf_pred=round(float(best["y"]), 1), prod=float(x["prod"]), cf_prod=round(float(best["prod"])),
                         cut_prod_pct=round(float(best["cut_prod_pct"]), 1),
                         moved_to_neighbors=round(float(best["moved_to_neighbors"]))))
    res = pd.DataFrame(rows)
    res.to_csv(out / "dice_peak_actions.csv", index=False, encoding="utf-8-sig")
    return res
