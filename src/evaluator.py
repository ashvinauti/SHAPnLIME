"""
Evaluation engine: classification metrics, ROC curve, PR curve, confusion matrix.
Produces both structured data (dict) and saved plots.
"""
import json
from pathlib import Path
from typing import Dict, Any, Optional

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.metrics import (
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
)

from .logger import get_logger

logger = get_logger("xai_ids.evaluator")


class Evaluator:
    """Comprehensive evaluation of anomaly detection performance."""

    def __init__(self, config):
        self.cfg = config.evaluation
        self.report_dir = Path(self.cfg.report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)
        self._results: Dict[str, Any] = {}

    # -----------------------------------------------------------------------
    # Main evaluation
    # -----------------------------------------------------------------------
    def evaluate(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        scores: np.ndarray,
        run_id: str = "eval"
    ) -> Dict[str, Any]:
        """
        Compute all metrics and generate plots.

        Args:
            y_true  : Binary ground truth (0=benign, 1=attack)
            y_pred  : Binary predictions
            scores  : Continuous anomaly scores (for ROC/PR)
            run_id  : Used for naming output files
        """
        logger.info("📐 Computing evaluation metrics …")
        results: Dict[str, Any] = {}

        # --- Classification metrics ---
        results["precision"] = float(precision_score(y_true, y_pred, zero_division=0))
        results["recall"] = float(recall_score(y_true, y_pred, zero_division=0))
        results["f1"] = float(f1_score(y_true, y_pred, zero_division=0))
        results["accuracy"] = float((y_true == y_pred).mean())

        if len(np.unique(y_true)) > 1:
            results["roc_auc"] = float(roc_auc_score(y_true, scores))
            results["avg_precision"] = float(average_precision_score(y_true, scores))
        else:
            results["roc_auc"] = None
            results["avg_precision"] = None
            logger.warning("⚠️  Only one class in y_true — ROC-AUC not computable")

        cm = confusion_matrix(y_true, y_pred)
        results["confusion_matrix"] = cm.tolist()
        tn, fp, fn, tp = cm.ravel() if cm.size == 4 else (0, 0, 0, 0)
        results["true_positives"] = int(tp)
        results["true_negatives"] = int(tn)
        results["false_positives"] = int(fp)
        results["false_negatives"] = int(fn)
        results["false_positive_rate"] = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
        results["detection_rate"] = results["recall"]
        results["classification_report"] = classification_report(y_true, y_pred,
                                                                   target_names=["Benign", "Attack"],
                                                                   output_dict=True)

        # Threshold analysis
        results["anomalies_detected"] = int(y_pred.sum())
        results["total_samples"] = int(len(y_pred))

        self._results = results
        self._log_summary(results)

        # --- Plots ---
        plot_paths = {}
        if self.cfg.plot_confusion_matrix:
            plot_paths["confusion_matrix"] = self._plot_confusion_matrix(cm, run_id)
        if self.cfg.plot_roc and results["roc_auc"] is not None:
            plot_paths["roc_curve"] = self._plot_roc(y_true, scores, results["roc_auc"], run_id)
        if self.cfg.plot_precision_recall and results["avg_precision"] is not None:
            plot_paths["pr_curve"] = self._plot_pr(y_true, scores, results["avg_precision"], run_id)

        plot_paths["score_distribution"] = self._plot_score_dist(y_true, scores, run_id)
        results["plots"] = plot_paths

        # Save JSON report
        json_path = self.report_dir / f"metrics_{run_id}.json"
        _safe = {k: v for k, v in results.items() if k != "plots"}
        with open(json_path, "w") as f:
            json.dump(_safe, f, indent=2, default=str)
        results["metrics_json"] = str(json_path)
        logger.info(f"📄 Metrics saved → {json_path}")

        return results

    # -----------------------------------------------------------------------
    # Threshold sweep (find optimal operating point)
    # -----------------------------------------------------------------------
    def threshold_sweep(
        self,
        y_true: np.ndarray,
        scores: np.ndarray,
        n_steps: int = 200
    ) -> Dict[str, Any]:
        """
        Sweep thresholds and find the one maximising F1.
        Returns optimal threshold and metrics at that point.
        """
        thresholds = np.linspace(scores.min(), scores.max(), n_steps)
        best_f1, best_thr, best_p, best_r = 0, 0, 0, 0
        f1s = []
        for thr in thresholds:
            pred = (scores > thr).astype(int)
            f1 = f1_score(y_true, pred, zero_division=0)
            f1s.append(f1)
            if f1 > best_f1:
                best_f1 = f1
                best_thr = thr
                best_p = precision_score(y_true, pred, zero_division=0)
                best_r = recall_score(y_true, pred, zero_division=0)

        logger.info(f"📏 Optimal threshold: {best_thr:.6f} → F1={best_f1:.4f}")

        # Plot F1 vs threshold
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.plot(thresholds, f1s, color="#4CAF50", linewidth=2)
        ax.axvline(best_thr, color="#F44336", linestyle="--", label=f"Best F1={best_f1:.3f}")
        ax.set_xlabel("Threshold")
        ax.set_ylabel("F1 Score")
        ax.set_title("Threshold Sweep — F1 Score")
        ax.legend()
        ax.grid(True, alpha=0.3)
        path = str(self.report_dir / "threshold_sweep.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()

        return {
            "optimal_threshold": float(best_thr),
            "optimal_f1": float(best_f1),
            "optimal_precision": float(best_p),
            "optimal_recall": float(best_r),
            "plot": path,
        }

    # -----------------------------------------------------------------------
    # Plots
    # -----------------------------------------------------------------------
    def _plot_confusion_matrix(self, cm: np.ndarray, run_id: str) -> str:
        fig, ax = plt.subplots(figsize=(6, 5))
        labels = ["Benign", "Attack"]
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        plt.colorbar(im, ax=ax)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(labels); ax.set_yticklabels(labels)
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black", fontsize=14)
        plt.tight_layout()
        path = str(self.report_dir / f"confusion_matrix_{run_id}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def _plot_roc(self, y_true: np.ndarray, scores: np.ndarray, auc: float, run_id: str) -> str:
        fpr, tpr, _ = roc_curve(y_true, scores)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(fpr, tpr, color="#2196F3", linewidth=2, label=f"ROC AUC = {auc:.4f}")
        ax.plot([0, 1], [0, 1], "k--", linewidth=1)
        ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve"); ax.legend(loc="lower right"); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = str(self.report_dir / f"roc_curve_{run_id}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def _plot_pr(self, y_true: np.ndarray, scores: np.ndarray, ap: float, run_id: str) -> str:
        prec, rec, _ = precision_recall_curve(y_true, scores)
        fig, ax = plt.subplots(figsize=(7, 5))
        ax.plot(rec, prec, color="#FF9800", linewidth=2, label=f"Avg Precision = {ap:.4f}")
        ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
        ax.set_title("Precision-Recall Curve"); ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = str(self.report_dir / f"pr_curve_{run_id}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    def _plot_score_dist(self, y_true: np.ndarray, scores: np.ndarray, run_id: str) -> str:
        fig, ax = plt.subplots(figsize=(10, 5))
        benign_scores = scores[y_true == 0]
        attack_scores = scores[y_true == 1]
        bins = 80
        ax.hist(benign_scores, bins=bins, alpha=0.6, color="#4CAF50", label="Benign", density=True)
        ax.hist(attack_scores, bins=bins, alpha=0.6, color="#F44336", label="Attack", density=True)
        ax.set_xlabel("Anomaly Score"); ax.set_ylabel("Density")
        ax.set_title("Anomaly Score Distribution")
        ax.legend(); ax.grid(True, alpha=0.3)
        plt.tight_layout()
        path = str(self.report_dir / f"score_distribution_{run_id}.png")
        plt.savefig(path, dpi=150, bbox_inches="tight")
        plt.close()
        return path

    # -----------------------------------------------------------------------
    # Logging
    # -----------------------------------------------------------------------
    def _log_summary(self, r: Dict[str, Any]) -> None:
        logger.info("=" * 50)
        logger.info("📊 EVALUATION RESULTS")
        logger.info(f"   Precision  : {r['precision']:.4f}")
        logger.info(f"   Recall     : {r['recall']:.4f}")
        logger.info(f"   F1 Score   : {r['f1']:.4f}")
        logger.info(f"   Accuracy   : {r['accuracy']:.4f}")
        if r["roc_auc"]:
            logger.info(f"   ROC-AUC    : {r['roc_auc']:.4f}")
        logger.info(f"   FP Rate    : {r['false_positive_rate']:.4f}")
        logger.info(f"   Detected   : {r['anomalies_detected']} / {r['total_samples']}")
        logger.info("=" * 50)
