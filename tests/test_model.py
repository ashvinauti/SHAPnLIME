"""
Unit tests for the AnomalyDetector model.
Run with: pytest tests/ -v
"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import numpy as np
import pytest
from src.config import Config


@pytest.fixture
def cfg():
    c = Config()
    c.model.autoencoder.epochs = 3
    c.model.autoencoder.batch_size = 16
    c.model.autoencoder.encoder_dims = [16, 8]
    c.model.autoencoder.decoder_dims = [8, 16]
    c.model.autoencoder.latent_dim = 4
    c.model.ensemble.use_isolation_forest = False
    return c


@pytest.fixture
def synthetic_data():
    rng = np.random.default_rng(42)
    X_train = rng.normal(0, 1, (200, 20)).astype(np.float32)
    X_test_benign = rng.normal(0, 1, (80, 20)).astype(np.float32)
    X_test_attack = rng.normal(5, 2, (20, 20)).astype(np.float32)  # far from benign
    X_test = np.vstack([X_test_benign, X_test_attack])
    y = np.array([0] * 80 + [1] * 20)
    return X_train, X_test, y


class TestAnomalyDetector:
    def test_fit_and_predict(self, cfg, synthetic_data):
        from src.model import AnomalyDetector
        X_train, X_test, y = synthetic_data
        det = AnomalyDetector(cfg)
        det.fit(X_train)

        assert det.threshold > 0, "Threshold should be positive"
        assert det.autoencoder is not None

        preds, scores, details = det.predict(X_test)
        assert preds.shape == (len(X_test),), "Predictions shape mismatch"
        assert scores.shape == (len(X_test),), "Scores shape mismatch"
        assert set(preds).issubset({0, 1}), "Predictions must be binary"
        assert scores.min() >= 0.0 and scores.max() <= 1.0, "Scores must be in [0,1]"

    def test_threshold_strategies(self, cfg, synthetic_data):
        from src.model import AnomalyDetector
        X_train, X_test, _ = synthetic_data

        for strategy in ["percentile", "iqr", "zscore", "mad"]:
            cfg.model.threshold.strategy = strategy
            det = AnomalyDetector(cfg)
            det.fit(X_train)
            assert det.threshold > 0, f"Invalid threshold for strategy: {strategy}"

    def test_anomaly_score_fn(self, cfg, synthetic_data):
        from src.model import AnomalyDetector
        X_train, X_test, _ = synthetic_data
        det = AnomalyDetector(cfg)
        det.fit(X_train)

        errors = det.anomaly_score_fn(X_test)
        assert errors.shape == (len(X_test),)
        assert all(e >= 0 for e in errors), "Reconstruction errors must be non-negative"

    def test_training_summary(self, cfg, synthetic_data):
        from src.model import AnomalyDetector
        X_train, _, _ = synthetic_data
        det = AnomalyDetector(cfg)
        det.fit(X_train)
        summary = det.training_summary()

        assert "final_train_loss" in summary
        assert "epochs_run" in summary
        assert summary["epochs_run"] <= cfg.model.autoencoder.epochs


class TestDataLoader:
    def test_encode_labels(self, cfg):
        from src.data_loader import DataLoader
        import pandas as pd
        loader = DataLoader(cfg)
        labels = pd.Series(["benign", "dos attack", "benign", "portscan"])
        y = loader._encode_labels(labels)
        np.testing.assert_array_equal(y, [0, 1, 0, 1])

    def test_feature_selection_variance(self, cfg):
        from src.data_loader import DataLoader
        import pandas as pd
        loader = DataLoader(cfg)
        df = pd.DataFrame({
            "a": [1.0] * 100,              # zero variance — should be dropped
            "b": np.random.randn(100),
            "c": np.random.randn(100),
        })
        result = loader._feature_select(df)
        assert "a" not in result.columns, "Zero-variance column should be dropped"
        assert "b" in result.columns


class TestEvaluator:
    def test_basic_metrics(self, cfg):
        from src.evaluator import Evaluator
        import tempfile, os
        cfg.evaluation.report_dir = tempfile.mkdtemp()
        cfg.evaluation.plot_roc = False
        cfg.evaluation.plot_precision_recall = False
        cfg.evaluation.plot_confusion_matrix = False

        evaluator = Evaluator(cfg)
        y_true = np.array([0, 0, 1, 1, 0, 1])
        y_pred = np.array([0, 0, 1, 0, 0, 1])
        scores = np.array([0.1, 0.2, 0.9, 0.4, 0.1, 0.8])

        metrics = evaluator.evaluate(y_true, y_pred, scores, run_id="test")
        assert metrics["precision"] == pytest.approx(1.0, abs=0.01)
        assert metrics["recall"] == pytest.approx(2 / 3, abs=0.01)
        assert "f1" in metrics
        assert "confusion_matrix" in metrics
