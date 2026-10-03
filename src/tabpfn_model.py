"""Optional public TabPFN-v2 regressor on the same day-ahead tabular features."""
import time
import hashlib
import os
from pathlib import Path
import numpy as np

MODEL_ID="Prior-Labs/TabPFN-v2-reg"
REVISION="4972a65a1b30806315c6f92499959ffbfc69a673"
FILENAME="tabpfn-v2-regressor.ckpt"
WEIGHT_SHA256="2ab5a07d5c41dfe6db9aa7ae106fc6de898326c2765be66505a07e2868c10736"


def resolve_checkpoint():
    from tabpfn.model_loading import get_cache_dir
    path=Path(os.environ.get("KAMP_TABPFN_MODEL_PATH",str(get_cache_dir()/FILENAME)))
    if not path.exists():
        from huggingface_hub import hf_hub_download
        path=Path(hf_hub_download(MODEL_ID,FILENAME,revision=REVISION))
    if hashlib.sha256(path.read_bytes()).hexdigest()!=WEIGHT_SHA256:
        raise ValueError("TabPFN checkpoint checksum mismatch; do not silently change weights")
    return path


class TabPFNModel:
    def __init__(self,context_days=None,n_estimators=4,model_version="v2"):
        if model_version!="v2": raise ValueError("Phase 2 adapter supports the pinned v2 checkpoint only")
        self.context_days=context_days; self.n_estimators=n_estimators; self.model_version=model_version

    def fit(self,X,y,hour=None,copy=None):
        import torch
        from tabpfn import TabPFNRegressor
        from tabpfn.constants import ModelVersion
        torch.set_num_threads(4)
        keep=np.ones(len(X),bool) if copy is None else ~np.asarray(copy,bool)
        if self.context_days is not None:
            keep &= X.index>=X.index.max().normalize()-np.timedelta64(self.context_days-1,"D")
        X,y=X.loc[keep],y.loc[keep]
        if len(X)<24: raise ValueError("Insufficient original context")
        # Column order and NaN handling are owned by TabPFN; no feature selection
        # or imputation using held-out rows.
        self.columns=list(X); self.n_samples=len(X); self.n_features=X.shape[1]
        self.training_end=X.index.max()
        version=ModelVersion(self.model_version)
        self.model=TabPFNRegressor.create_default_for_version(version,n_estimators=self.n_estimators,
                  model_path=str(resolve_checkpoint()),
                  device="cpu",ignore_pretraining_limits=True,random_state=42,
                  n_preprocessing_jobs=1,fit_mode="fit_with_cache")
        tic=time.perf_counter(); self.model.fit(X,y.to_numpy()); self.fit_seconds=time.perf_counter()-tic
        return self

    def predict(self,X,hour=None):
        if list(X)!=self.columns: raise ValueError("Feature schema changed")
        if X.index.min()<=self.training_end: raise ValueError("Prediction must follow fit data")
        tic=time.perf_counter()
        out=np.asarray(self.model.predict(X,output_type="median"),float)
        self.predict_seconds=time.perf_counter()-tic
        return out
