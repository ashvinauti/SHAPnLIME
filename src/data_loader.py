"""
Data loading, cleaning, feature engineering, and preprocessing pipeline.
Supports: Parquet, CSV. Designed for CIC-IDS-2017/2018 and generic netflow datasets.
"""
import os
import time
import pickle
import hashlib
import warnings
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import VarianceThreshold

from .logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger("xai_ids.data_loader")


class DataLoader:
    """End-to-end data loading and preprocessing pipeline."""

    # Known label mappings for common IDS datasets
    BENIGN_ALIASES = {"benign", "normal", "legitimate", "0", "safe"}
    INFINITE_REPLACEMENTS = [np.inf, -np.inf]

    def __init__(self, config):
        self.cfg = config
        self.data_cfg = config.data
        self.scaler: Optional[StandardScaler] = None
        self.feature_names: List[str] = []
        self.dropped_features: List[str] = []
        self._cache: Dict[str, Any] = {}

    # -------------------------------------------------------------------------
    # Public API
    # -------------------------------------------------------------------------
    def load_and_prepare(
        self,
        path: Optional[str] = None,
        fit_scaler: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, pd.Series]:
        """
        Full pipeline: load → clean → feature select → scale → split.

        Returns:
            X_train  : benign-only scaled features (for autoencoder training)
            X_test   : all scaled features
            y_test   : binary labels (0=benign, 1=attack)
            df_raw   : cleaned unscaled DataFrame (for LIME/SHAP feature names)
        """
        data_path = path or self.data_cfg.path
        t0 = time.time()
        logger.info(f"📂 Loading data from: {data_path}")

        df, labels = self._load_files(data_path)
        logger.info(f"✅ Raw shape: {df.shape} | Elapsed: {time.time()-t0:.1f}s")

        df, labels = self._clean(df, labels)
        logger.info(f"✅ Cleaned shape: {df.shape}")

        if self.data_cfg.sample_size:
            df, labels = self._sample(df, labels, self.data_cfg.sample_size)
            logger.info(f"✅ Sampled shape: {df.shape}")

        df = self._feature_select(df)
        logger.info(f"✅ After feature selection: {df.shape}")

        self.feature_names = df.columns.tolist()
        X = df.values.astype(np.float32)

        if fit_scaler:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(X)
        else:
            if self.scaler is None:
                raise RuntimeError("Scaler not fitted. Call load_and_prepare with fit_scaler=True first.")
            X_scaled = self.scaler.transform(X)

        y_binary = self._encode_labels(labels)
        benign_mask = (y_binary == 0)

        X_train = X_scaled[benign_mask]
        X_test = X_scaled
        y_test = y_binary

        logger.info(f"✅ Train (benign): {X_train.shape} | Test (all): {X_test.shape}")
        logger.info(f"   Benign: {benign_mask.sum()} | Attacks: {(~benign_mask).sum()}")
        return X_train, X_test, y_test, df

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        """Transform new data using fitted scaler (for inference)."""
        if self.scaler is None:
            raise RuntimeError("Scaler not fitted.")
        df = df[self.feature_names] if self.feature_names else df
        df = df.select_dtypes(include=[np.number])
        df.replace(self.INFINITE_REPLACEMENTS, np.nan, inplace=True)
        df.fillna(df.median(), inplace=True)
        return self.scaler.transform(df.values.astype(np.float32))

    def save(self, path: str = "models/preprocessor.pkl") -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"scaler": self.scaler,
                         "feature_names": self.feature_names,
                         "dropped_features": self.dropped_features}, f)
        logger.info(f"💾 Preprocessor saved → {path}")

    @classmethod
    def load_preprocessor(cls, config, path: str = "models/preprocessor.pkl") -> "DataLoader":
        loader = cls(config)
        with open(path, "rb") as f:
            data = pickle.load(f)
        loader.scaler = data["scaler"]
        loader.feature_names = data["feature_names"]
        loader.dropped_features = data.get("dropped_features", [])
        logger.info(f"✅ Preprocessor loaded from {path}")
        return loader

    # -------------------------------------------------------------------------
    # Private helpers
    # -------------------------------------------------------------------------
    def _load_files(self, path: str) -> Tuple[pd.DataFrame, pd.Series]:
        p = Path(path)
        ext = self.data_cfg.file_format.lower()

        if p.is_file():
            files = [p]
        elif p.is_dir():
            files = sorted(p.glob(f"*.{ext}"))
            if not files:
                files = sorted(p.glob("*.csv"))
        else:
            raise FileNotFoundError(f"Data path not found: {path}")

        if not files:
            raise FileNotFoundError(f"No {ext} files found in {path}")

        frames = []
        for f in files:
            logger.info(f"  Loading {f.name} …")
            if f.suffix == ".parquet":
                df_tmp = pd.read_parquet(f, engine="pyarrow")
            else:
                df_tmp = pd.read_csv(f, low_memory=False)
            frames.append(df_tmp)

        df = pd.concat(frames, ignore_index=True)
        label_col = self.data_cfg.label_column

        # Try to auto-detect label column if not found exactly
        if label_col not in df.columns:
            candidates = [c for c in df.columns if "label" in c.lower()]
            if candidates:
                label_col = candidates[0]
                logger.warning(f"⚠️  Label column '{self.data_cfg.label_column}' not found; using '{label_col}'")
            else:
                raise ValueError(f"Label column '{label_col}' not found in dataset. "
                                  f"Columns: {df.columns.tolist()[:10]}")

        labels = df[label_col].astype(str).str.strip().str.lower()
        df = df.drop(columns=[label_col])
        return df, labels

    def _clean(self, df: pd.DataFrame, labels: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
        # Keep only numeric features
        df = df.select_dtypes(include=[np.number])

        # Replace inf values
        df.replace(self.INFINITE_REPLACEMENTS, np.nan, inplace=True)

        # Drop columns with > 50% NaN
        thresh = int(0.5 * len(df))
        df = df.dropna(axis=1, thresh=thresh)

        # Fill remaining NaN with column median
        df.fillna(df.median(), inplace=True)

        # Drop constant columns
        non_constant = df.columns[df.nunique() > 1]
        dropped_const = list(set(df.columns) - set(non_constant))
        if dropped_const:
            logger.debug(f"   Dropped {len(dropped_const)} constant columns")
        df = df[non_constant]

        # Align labels index
        labels = labels.loc[df.index]
        return df, labels

    def _sample(self, df: pd.DataFrame, labels: pd.Series, n: int) -> Tuple[pd.DataFrame, pd.Series]:
        if len(df) <= n:
            return df, labels
        # Stratified-like sampling: preserve class ratio
        benign_frac = (labels.str.contains(self.data_cfg.benign_label)).mean()
        n_benign = min(int(n * benign_frac), (labels.str.contains(self.data_cfg.benign_label)).sum())
        n_attack = n - n_benign

        benign_idx = df[labels.str.contains(self.data_cfg.benign_label)].index
        attack_idx = df[~labels.str.contains(self.data_cfg.benign_label)].index

        rng = np.random.default_rng(self.data_cfg.random_state)
        sel_benign = rng.choice(benign_idx, min(n_benign, len(benign_idx)), replace=False)
        sel_attack = rng.choice(attack_idx, min(n_attack, len(attack_idx)), replace=False)
        sel = np.concatenate([sel_benign, sel_attack])
        rng.shuffle(sel)

        return df.loc[sel], labels.loc[sel]

    def _feature_select(self, df: pd.DataFrame) -> pd.DataFrame:
        fc = self.data_cfg.feature_selection
        if not fc.enabled:
            return df

        original_cols = set(df.columns)

        # 1. Variance threshold
        sel = VarianceThreshold(threshold=fc.variance_threshold)
        sel.fit(df)
        kept = df.columns[sel.get_support()].tolist()
        df = df[kept]
        logger.debug(f"   Variance filter: kept {len(kept)} / {len(original_cols)} features")

        # 2. Correlation filter (drop one of highly correlated pairs)
        corr_matrix = df.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > fc.correlation_threshold)]
        self.dropped_features = to_drop
        df = df.drop(columns=to_drop)
        if to_drop:
            logger.debug(f"   Correlation filter: dropped {len(to_drop)} features")

        return df

    def _encode_labels(self, labels: pd.Series) -> np.ndarray:
        benign_keyword = self.data_cfg.benign_label.lower()
        y = np.where(labels.str.contains(benign_keyword), 0, 1)
        return y

    # -------------------------------------------------------------------------
    # Dataset summary
    # -------------------------------------------------------------------------
    def summary(self, df: pd.DataFrame, y: np.ndarray) -> Dict[str, Any]:
        return {
            "num_samples": len(df),
            "num_features": df.shape[1],
            "benign_count": int((y == 0).sum()),
            "attack_count": int((y == 1).sum()),
            "benign_pct": f"{100*(y==0).mean():.1f}%",
            "attack_pct": f"{100*(y==1).mean():.1f}%",
            "features": self.feature_names,
        }
