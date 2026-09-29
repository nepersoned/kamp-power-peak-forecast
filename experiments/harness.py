"""성능 개선 실험용 공통 틀. 선택은 2개 검증 fold 평균으로만 하고, 테스트는 최종 1회."""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
warnings.filterwarnings("ignore")

from src.data import load  # noqa: E402
from src.features import build  # noqa: E402
from src.models import operating  # noqa: E402

START = "2021-01-08"
FOLDS = [("2021-07-02", "2021-07-16"), ("2021-07-16", "2021-08-16")]
TEST = ("2021-08-16", "2021-09-15")

df = load()
X = build(df)
OK = (~df["outage"]).to_numpy() & X["power_lag168"].notna().to_numpy()
COPY = df["is_copy"].to_numpy()
T = df.index
H = df["hour"]
ON = operating(X)


def split(a, b, keep_copies=False):
    tr = OK & (T >= START) & (T < a) & (keep_copies | ~COPY)
    te = OK & (T >= a) & (T < b)
    return tr, te


def off_pred(y, tr, te):
    prof = y[tr & ~ON].groupby(H[tr & ~ON]).median()
    return H[te & ~ON].map(prof).to_numpy()


def evaluate(fit_on, targets=("power", "peak15"), folds=FOLDS, keep_copies=False, Xm=None, return_preds=False):
    """fit_on(Xtr, ytr, Xte, meta) -> 가동일 예측. 휴무일은 중앙값 고정.
    반환: {target: [fold MAE ...]} 및 평균."""
    Xm = X if Xm is None else Xm
    res, preds = {}, {}
    for tgt in targets:
        y = df[tgt]
        maes = []
        for a, b in folds:
            tr, te = split(a, b, keep_copies)
            p = np.empty(te.sum())
            on_te = ON[te]
            meta = dict(target=tgt, idx_tr=T[tr & ON], idx_te=T[te & ON], copy_tr=COPY[tr & ON])
            p[on_te] = fit_on(Xm[tr & ON], y[tr & ON], Xm[te & ON], meta)
            p[~on_te] = off_pred(y, tr, te)
            maes.append(float(np.mean(np.abs(y[te].to_numpy() - p))))
            preds[(tgt, a)] = pd.Series(p, T[te])
        res[tgt] = maes
    score = float(np.mean([np.mean(v) for v in res.values()]))
    return (score, res, preds) if return_preds else (score, res)


def fmt(score, res):
    return f"score {score:.3f} | " + " | ".join(f"{k}: " + "/".join(f"{m:.2f}" for m in v) for k, v in res.items())
