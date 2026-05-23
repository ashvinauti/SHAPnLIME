"""
XAI Engine: SHAP (global) + LIME (local) explanations for the anomaly detector.
Handles batch processing, caching, and export.
"""
import json
import warnings
from pathlib import Path
from typing import Optional, List, Dict, Any, Callable

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt

from .logger import get_logger

warnings.filterwarnings("ignore")
logger = get_logger("xai_ids.explainer")


class XAIExplainer:
    """
    Wraps SHAP + LIME explanations for tabular anomaly detection.
    """

    def __init__(self, config, score_fn: Callable, feature_names: List[str]):
        """
        Args:
            config       : App configuration object
            score_fn     : Callable[np.ndarray] → np.ndarray of anomaly scores
            feature_names: List of feature column names
        """
        self.cfg = config.explainability
        self.score_fn = score_fn
        self.feature_names = feature_names
        self._shap_explainer = None
        self._lime_explainer = None
        self._shap_values: Optional[np.ndarray] = None
        self._shap_data: Optional[np.ndarray] = None

    # -----------------------------------------------------------------------
    # SHAP — Global Explanations
    # -----------------------------------------------------------------------
    def fit_shap(self, X_background: np.ndarray) -> None:
        """Fit SHAP KernelExplainer on background (benign) samples."""
        import shap
        shap_cfg = self.cfg.shap
        n_bg = min(shap_cfg.num_background_samples, X_background.shape[0])

        rng = np.random.default_rng(42)
        bg_idx = rng.choice(X_background.shape[0], n_bg, replace=False)
        background = X_background[bg_idx]

        logger.info(f"🔍 Fitting SHAP KernelExplainer on {n_bg} background samples …")
        self._shap_explainer = shap.KernelExplainer(self.score_fn, background)
        logger.info("✅ SHAP explainer ready")

    def compute_shap(self, X_explain: np.ndarray) -> np.ndarray:
        """
        Compute SHAP values for a set of samples.
        Returns ndarray of shape (n_samples, n_features).
        """
        if self._shap_explainer is None:
            raise RuntimeError("Call fit_shap() first.")
        shap_cfg = self.cfg.shap
        n = min(shap_cfg.num_explain_samples, X_explain.shape[0])
        logger.info(f"🔍 Computing SHAP values for {n} samples …")
        raw = self._shap_explainer.shap_values(X_explain[:n])
        vals = np.array(raw)
        if vals.ndim == 1:
            vals = vals.reshape(1, -1)
        self._shap_values = vals
        self._shap_data = X_explain[:n]
        logger.info(f"✅ SHAP values computed: {vals.shape}")
        return vals

    def shap_feature_importance(self) -> pd.DataFrame:
        """Returns global feature importances (mean |SHAP|) sorted descending."""
        if self._shap_values is None:
            raise RuntimeError("Run compute_shap() first.")
        mean_abs = np.abs(self._shap_values).mean(axis=0)
        df = pd.DataFrame({
            "feature": self.feature_names[:len(mean_abs)],
            "importance": mean_abs
        }).sort_values("importance", ascending=False).reset_index(drop=True)
        return df

    def plot_shap_summary(self, save_path: Optional[str] = None) -> str:
        """SHAP beeswarm summary plot."""
        import shap
        if self._shap_values is None:
            raise RuntimeError("Run compute_shap() first.")
        fig, ax = plt.subplots(figsize=(12, 8))
        shap.summary_plot(
            self._shap_values,
            self._shap_data,
            feature_names=self.feature_names[:self._shap_values.shape[1]],
            max_display=self.cfg.shap.plot_top_features,
            show=False
        )
        plt.tight_layout()
        out = save_path or "reports/shap_summary.png"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"📊 SHAP summary saved → {out}")
        return out

    def plot_shap_bar(self, save_path: Optional[str] = None) -> str:
        """SHAP bar chart of global feature importances."""
        import shap
        if self._shap_values is None:
            raise RuntimeError("Run compute_shap() first.")
        fig, ax = plt.subplots(figsize=(10, 7))
        shap.summary_plot(
            self._shap_values,
            self._shap_data,
            feature_names=self.feature_names[:self._shap_values.shape[1]],
            plot_type="bar",
            max_display=self.cfg.shap.plot_top_features,
            show=False
        )
        plt.tight_layout()
        out = save_path or "reports/shap_bar.png"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"📊 SHAP bar chart saved → {out}")
        return out

    def plot_shap_waterfall(self, sample_idx: int = 0,
                             save_path: Optional[str] = None) -> str:
        """SHAP waterfall plot for a single sample."""
        import shap
        if self._shap_values is None or self._shap_data is None:
            raise RuntimeError("Run compute_shap() first.")

        expected = float(self.score_fn(self._shap_data).mean())
        shap_val = self._shap_values[sample_idx]
        feat_names = self.feature_names[:len(shap_val)]

        explanation = shap.Explanation(
            values=shap_val,
            base_values=expected,
            data=self._shap_data[sample_idx],
            feature_names=feat_names,
        )
        fig = plt.figure(figsize=(12, 6))
        shap.plots.waterfall(explanation, max_display=15, show=False)
        plt.tight_layout()
        out = save_path or f"reports/shap_waterfall_sample{sample_idx}.png"
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(out, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"📊 SHAP waterfall saved → {out}")
        return out

    def export_shap_json(self, path: str = "reports/shap_values.json") -> str:
        """Export SHAP values and feature importance to JSON."""
        if self._shap_values is None:
            raise RuntimeError("Run compute_shap() first.")
        fi = self.shap_feature_importance()
        data = {
            "feature_importance": fi.to_dict(orient="records"),
            "shap_values_sample": self._shap_values[:5].tolist(),
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"📄 SHAP JSON exported → {path}")
        return path

    # -----------------------------------------------------------------------
    # LIME — Local Explanations
    # -----------------------------------------------------------------------
    def fit_lime(self, X_train: np.ndarray) -> None:
        """Fit LIME LimeTabularExplainer."""
        from lime.lime_tabular import LimeTabularExplainer
        lime_cfg = self.cfg.lime
        logger.info("🔍 Fitting LIME LimeTabularExplainer …")
        self._lime_explainer = LimeTabularExplainer(
            training_data=X_train,
            feature_names=self.feature_names,
            mode="regression",
            kernel_width=lime_cfg.kernel_width,
        )
        logger.info("✅ LIME explainer ready")

    def explain_lime(self, x: np.ndarray, sample_label: str = "sample") -> Dict[str, Any]:
        """
        LIME explanation for a single sample.

        Args:
            x            : 1-D array (single sample)
            sample_label : Label for saving plot

        Returns:
            dict with 'features', 'weights', and 'plot_path'
        """
        if self._lime_explainer is None:
            raise RuntimeError("Call fit_lime() first.")
        lime_cfg = self.cfg.lime
        exp = self._lime_explainer.explain_instance(
            x,
            self.score_fn,
            num_features=lime_cfg.num_features,
            num_samples=lime_cfg.num_samples,
        )
        feats_weights = exp.as_list()
        result = {
            "features": [fw[0] for fw in feats_weights],
            "weights": [fw[1] for fw in feats_weights],
        }

        # Save plot
        fig = exp.as_pyplot_figure()
        plt.tight_layout()
        plot_path = f"reports/lime_{sample_label}.png"
        Path(plot_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(plot_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        result["plot_path"] = plot_path
        logger.info(f"📊 LIME explanation saved → {plot_path}")
        return result

    def explain_top_anomalies(
        self,
        X: np.ndarray,
        scores: np.ndarray,
        top_n: int = 5
    ) -> List[Dict[str, Any]]:
        """Run LIME on the top-N most anomalous samples."""
        top_idx = np.argsort(scores)[::-1][:top_n]
        results = []
        for rank, idx in enumerate(top_idx):
            logger.info(f"  LIME → sample {idx} (rank {rank+1}, score={scores[idx]:.4f})")
            res = self.explain_lime(X[idx], sample_label=f"anomaly_rank{rank+1}")
            res["sample_index"] = int(idx)
            res["anomaly_score"] = float(scores[idx])
            results.append(res)
        return results

    # -----------------------------------------------------------------------
    # Combined pipeline
    # -----------------------------------------------------------------------
    def run_full_pipeline(
        self,
        X_train: np.ndarray,
        X_test: np.ndarray,
        scores: np.ndarray,
        report_dir: str = "reports"
    ) -> Dict[str, Any]:
        """Run SHAP + LIME pipeline and return paths to all generated assets."""
        Path(report_dir).mkdir(parents=True, exist_ok=True)
        results: Dict[str, Any] = {}

        # --- SHAP ---
        if self.cfg.shap.enabled:
            self.fit_shap(X_train)
            self.compute_shap(X_test)
            results["shap_summary_plot"] = self.plot_shap_summary(f"{report_dir}/shap_summary.png")
            results["shap_bar_plot"] = self.plot_shap_bar(f"{report_dir}/shap_bar.png")
            results["shap_waterfall"] = self.plot_shap_waterfall(0, f"{report_dir}/shap_waterfall.png")
            results["shap_json"] = self.export_shap_json(f"{report_dir}/shap_values.json")
            results["shap_feature_importance"] = self.shap_feature_importance().head(20).to_dict(orient="records")

        # --- LIME ---
        if self.cfg.lime.enabled:
            self.fit_lime(X_train)
            lime_results = self.explain_top_anomalies(X_test, scores, top_n=3)
            results["lime_explanations"] = lime_results

        return results
