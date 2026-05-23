"""
Anomaly detection model:
  • Deep Autoencoder (primary) with dropout, batch norm, early stopping
  • Isolation Forest (secondary, optional)
  • Adaptive threshold strategies
  • Model persistence (save / load)
"""
import os
import json
import pickle
import time
from pathlib import Path
from typing import Optional, Tuple, Dict, Any

import numpy as np
from sklearn.ensemble import IsolationForest

from .logger import get_logger

logger = get_logger("xai_ids.model")

# ---------------------------------------------------------------------------
# Lazy TensorFlow import to avoid long startup when not needed
# ---------------------------------------------------------------------------
def _import_tf():
    import tensorflow as tf
    from tensorflow.keras import layers, models, callbacks, optimizers
    return tf, layers, models, callbacks, optimizers


class AnomalyDetector:
    """
    Hybrid anomaly detector: Autoencoder + (optional) Isolation Forest ensemble.
    """

    def __init__(self, config):
        self.cfg = config
        self.ae_cfg = config.model.autoencoder
        self.thr_cfg = config.model.threshold
        self.ens_cfg = config.model.ensemble
        self.pers_cfg = config.model.persistence

        self.autoencoder = None
        self.isolation_forest: Optional[IsolationForest] = None
        self.threshold: float = 0.0
        self.input_dim: int = 0
        self.history = None
        self._train_errors: Optional[np.ndarray] = None

    # -----------------------------------------------------------------------
    # Build
    # -----------------------------------------------------------------------
    def _build_autoencoder(self, input_dim: int):
        tf, layers, models, callbacks, optimizers = _import_tf()
        self.input_dim = input_dim
        cfg = self.ae_cfg

        def encoder_block(x, units):
            x = layers.Dense(units)(x)
            if cfg.use_batch_norm:
                x = layers.BatchNormalization()(x)
            x = layers.Activation(cfg.activation)(x)
            x = layers.Dropout(cfg.dropout_rate)(x)
            return x

        inp = layers.Input(shape=(input_dim,), name="input")
        x = inp
        for units in cfg.encoder_dims:
            x = encoder_block(x, units)
        latent = layers.Dense(cfg.latent_dim, name="latent")(x)

        for units in cfg.decoder_dims:
            x = encoder_block(latent if x is latent else x, units)
            if x is latent:
                x = encoder_block(latent, units)
        out = layers.Dense(input_dim, activation="linear", name="output")(x)

        self.autoencoder = models.Model(inp, out, name="xai_ids_autoencoder")
        opt = optimizers.Adam(learning_rate=cfg.learning_rate)
        self.autoencoder.compile(optimizer=opt, loss="mse")
        logger.info(f"✅ Autoencoder built | params: {self.autoencoder.count_params():,}")
        self.autoencoder.summary(print_fn=logger.debug)

    # -----------------------------------------------------------------------
    # Training
    # -----------------------------------------------------------------------
    def fit(self, X_train: np.ndarray, X_val: Optional[np.ndarray] = None) -> "AnomalyDetector":
        """
        Train the autoencoder on benign-only data.
        Optionally trains Isolation Forest on the same data.
        """
        tf, layers, models, callbacks, optimizers = _import_tf()
        t0 = time.time()
        logger.info("🚀 Training Autoencoder …")

        self._build_autoencoder(X_train.shape[1])
        cfg = self.ae_cfg

        cbs = [
            callbacks.EarlyStopping(
                monitor="val_loss" if X_val is not None else "loss",
                patience=cfg.early_stopping_patience,
                restore_best_weights=True,
                verbose=0
            ),
            callbacks.ReduceLROnPlateau(
                monitor="val_loss" if X_val is not None else "loss",
                factor=0.5,
                patience=cfg.reduce_lr_patience,
                min_lr=1e-6,
                verbose=0
            ),
        ]

        val_data = (X_val, X_val) if X_val is not None else None
        self.history = self.autoencoder.fit(
            X_train, X_train,
            epochs=cfg.epochs,
            batch_size=cfg.batch_size,
            validation_split=cfg.validation_split if val_data is None else 0.0,
            validation_data=val_data,
            callbacks=cbs,
            verbose=0
        )

        stopped_epoch = len(self.history.history["loss"])
        logger.info(f"✅ Autoencoder trained ({stopped_epoch} epochs) | {time.time()-t0:.1f}s")

        # Compute training errors for threshold calibration
        self._train_errors = self._reconstruction_error(X_train)
        self._fit_threshold(self._train_errors)

        if self.ens_cfg.use_isolation_forest:
            logger.info("🌲 Training Isolation Forest …")
            self.isolation_forest = IsolationForest(
                contamination=self.ens_cfg.isolation_contamination,
                random_state=self.cfg.data.random_state,
                n_jobs=-1,
            )
            self.isolation_forest.fit(X_train)
            logger.info("✅ Isolation Forest trained")

        return self

    # -----------------------------------------------------------------------
    # Prediction
    # -----------------------------------------------------------------------
    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Returns:
            predictions : binary array (0=normal, 1=anomaly)
            scores      : continuous anomaly scores (normalized 0–1)
            details     : dict with ae_errors, if_scores (optional)
        """
        ae_errors = self._reconstruction_error(X)
        ae_scores = self._normalize_scores(ae_errors)
        details = {"ae_errors": ae_errors, "ae_scores": ae_scores}

        if self.isolation_forest is not None:
            # IF returns -1 (anomaly) / +1 (normal) and a score
            if_raw = self.isolation_forest.decision_function(X)  # higher = more normal
            if_scores = 1.0 - self._normalize_scores(if_raw)     # invert: higher = more anomalous
            details["if_scores"] = if_scores
            scores = (self.ens_cfg.ensemble_weight_ae * ae_scores +
                      self.ens_cfg.ensemble_weight_if * if_scores)
        else:
            scores = ae_scores

        predictions = (ae_errors > self.threshold).astype(int)
        return predictions, scores, details

    def anomaly_score_fn(self, X: np.ndarray) -> np.ndarray:
        """Callable for SHAP/LIME: returns per-sample anomaly score (AE error)."""
        return self._reconstruction_error(X)

    # -----------------------------------------------------------------------
    # Threshold strategies
    # -----------------------------------------------------------------------
    def _fit_threshold(self, train_errors: np.ndarray) -> float:
        strategy = self.thr_cfg.strategy
        if strategy == "percentile":
            self.threshold = float(np.percentile(train_errors, self.thr_cfg.percentile))
        elif strategy == "iqr":
            q1, q3 = np.percentile(train_errors, [25, 75])
            iqr = q3 - q1
            self.threshold = float(q3 + self.thr_cfg.iqr_multiplier * iqr)
        elif strategy == "zscore":
            mean, std = train_errors.mean(), train_errors.std()
            self.threshold = float(mean + self.thr_cfg.zscore_threshold * std)
        elif strategy == "mad":
            median = np.median(train_errors)
            mad = np.median(np.abs(train_errors - median))
            self.threshold = float(median + self.thr_cfg.zscore_threshold * 1.4826 * mad)
        else:
            raise ValueError(f"Unknown threshold strategy: {strategy}")
        logger.info(f"📏 Threshold ({strategy}): {self.threshold:.6f}")
        return self.threshold

    def set_threshold(self, value: float) -> None:
        """Manually override the threshold (for fine-tuning)."""
        self.threshold = value
        logger.info(f"📏 Threshold set manually: {value:.6f}")

    # -----------------------------------------------------------------------
    # Helpers
    # -----------------------------------------------------------------------
    def _reconstruction_error(self, X: np.ndarray) -> np.ndarray:
        recon = self.autoencoder.predict(X, verbose=0)
        return np.mean((X - recon) ** 2, axis=1)

    @staticmethod
    def _normalize_scores(scores: np.ndarray) -> np.ndarray:
        mn, mx = scores.min(), scores.max()
        if mx == mn:
            return np.zeros_like(scores)
        return (scores - mn) / (mx - mn)

    def training_summary(self) -> Dict[str, Any]:
        if self.history is None:
            return {}
        return {
            "final_train_loss": float(self.history.history["loss"][-1]),
            "final_val_loss": float(self.history.history.get("val_loss", [0])[-1]),
            "epochs_run": len(self.history.history["loss"]),
            "threshold": self.threshold,
        }

    # -----------------------------------------------------------------------
    # Persistence
    # -----------------------------------------------------------------------
    def save(self, run_id: str = "latest") -> str:
        model_dir = Path(self.pers_cfg.model_dir) / run_id
        model_dir.mkdir(parents=True, exist_ok=True)

        # Save Keras autoencoder
        ae_path = str(model_dir / "autoencoder.keras")
        self.autoencoder.save(ae_path)

        # Save meta (threshold, config, history)
        meta = {
            "threshold": self.threshold,
            "input_dim": self.input_dim,
            "run_id": run_id,
        }
        if self.history:
            meta["training_history"] = {k: [float(v) for v in vals]
                                         for k, vals in self.history.history.items()}

        with open(model_dir / "meta.json", "w") as f:
            json.dump(meta, f, indent=2)

        # Save Isolation Forest
        if self.isolation_forest is not None:
            with open(model_dir / "isolation_forest.pkl", "wb") as f:
                pickle.dump(self.isolation_forest, f)

        logger.info(f"💾 Model saved → {model_dir}")
        return str(model_dir)

    @classmethod
    def load(cls, config, run_id: str = "latest") -> "AnomalyDetector":
        tf, layers, models, callbacks, optimizers = _import_tf()
        detector = cls(config)
        model_dir = Path(config.model.persistence.model_dir) / run_id

        ae_path = model_dir / "autoencoder.keras"
        detector.autoencoder = models.load_model(str(ae_path))
        detector.input_dim = detector.autoencoder.input_shape[1]

        with open(model_dir / "meta.json") as f:
            meta = json.load(f)
        detector.threshold = meta["threshold"]

        if_path = model_dir / "isolation_forest.pkl"
        if if_path.exists():
            with open(if_path, "rb") as f:
                detector.isolation_forest = pickle.load(f)

        logger.info(f"✅ Model loaded from {model_dir} | threshold={detector.threshold:.6f}")
        return detector
