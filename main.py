#!/usr/bin/env python3
"""
XAI-IDS Pro — Command Line Interface
=====================================
Commands:
  train     — Load data, train autoencoder + Isolation Forest, save model
  evaluate  — Evaluate model on full dataset, generate metrics & plots
  explain   — Run SHAP + LIME pipeline, save explanation artifacts
  report    — Generate HTML report from existing run artifacts
  detect    — Run inference on a new CSV/Parquet file
  serve     — Start FastAPI REST API server
  dashboard — Launch Streamlit dashboard
  info      — Show model info for a saved run
  sweep     — Threshold sweep to find optimal detection threshold

Usage:
  python main.py train --config config.yaml
  python main.py evaluate --run-id latest
  python main.py explain --run-id latest
  python main.py report --run-id latest
  python main.py detect --input data/test.csv --output results.csv
  python main.py serve --host 0.0.0.0 --port 8000
  python main.py dashboard
  python main.py sweep --run-id latest
"""
import sys
import argparse
import datetime
import json
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Sub-command implementations
# ---------------------------------------------------------------------------

def cmd_train(args, cfg):
    from src.data_loader import DataLoader
    from src.model import AnomalyDetector
    from src.logger import get_logger
    logger = get_logger("xai_ids.train")

    run_id = args.run_id or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"🚀 Training pipeline | run_id={run_id}")

    loader = DataLoader(cfg)
    X_train, X_test, y_test, df_raw = loader.load_and_prepare(args.data or cfg.data.path)
    loader.save(f"models/{run_id}/preprocessor.pkl")

    detector = AnomalyDetector(cfg)
    detector.fit(X_train)

    model_dir = detector.save(run_id)
    # Also save as "latest"
    detector.save("latest")
    loader.save("models/latest/preprocessor.pkl")

    summary = detector.training_summary()
    logger.info(f"✅ Training complete: {summary}")
    logger.info(f"💾 Model saved → {model_dir}")
    print(json.dumps({"status": "ok", "run_id": run_id, **summary}, indent=2))


def cmd_evaluate(args, cfg):
    from src.data_loader import DataLoader
    from src.model import AnomalyDetector
    from src.evaluator import Evaluator
    from src.logger import get_logger
    logger = get_logger("xai_ids.evaluate")

    run_id = args.run_id or "latest"
    logger.info(f"📐 Evaluating run_id={run_id}")

    loader = DataLoader.load_preprocessor(cfg, f"models/{run_id}/preprocessor.pkl")
    detector = AnomalyDetector.load(cfg, run_id)

    # Re-load test data
    _, X_test, y_test, _ = loader.load_and_prepare(args.data or cfg.data.path, fit_scaler=False)

    preds, scores, _ = detector.predict(X_test)
    evaluator = Evaluator(cfg)
    metrics = evaluator.evaluate(y_test, preds, scores, run_id=run_id)

    # Threshold sweep
    sweep = evaluator.threshold_sweep(y_test, scores)
    logger.info(f"🏆 Optimal threshold: {sweep['optimal_threshold']:.6f} → F1={sweep['optimal_f1']:.4f}")
    print(json.dumps({k: v for k, v in metrics.items() if k not in ("plots", "classification_report")},
                     indent=2, default=str))


def cmd_explain(args, cfg):
    from src.data_loader import DataLoader
    from src.model import AnomalyDetector
    from src.explainer import XAIExplainer
    from src.logger import get_logger
    logger = get_logger("xai_ids.explain")

    run_id = args.run_id or "latest"
    logger.info(f"🔍 Generating explanations for run_id={run_id}")

    loader = DataLoader.load_preprocessor(cfg, f"models/{run_id}/preprocessor.pkl")
    detector = AnomalyDetector.load(cfg, run_id)

    _, X_test, y_test, df_raw = loader.load_and_prepare(args.data or cfg.data.path, fit_scaler=False)
    benign_mask = (y_test == 0)
    X_train = X_test[benign_mask]

    preds, scores, _ = detector.predict(X_test)

    explainer = XAIExplainer(cfg, detector.anomaly_score_fn, loader.feature_names)
    results = explainer.run_full_pipeline(X_train, X_test, scores, cfg.evaluation.report_dir)
    logger.info("✅ Explanation pipeline complete")
    print(json.dumps({"status": "ok", **{k: v for k, v in results.items()
                                          if isinstance(v, (str, list, dict))}}, indent=2))


def cmd_report(args, cfg):
    from src.reporter import HTMLReporter
    from src.logger import get_logger
    import json as _json
    logger = get_logger("xai_ids.report")

    run_id = args.run_id or "latest"
    report_dir = Path(cfg.evaluation.report_dir)

    metrics_path = report_dir / f"metrics_{run_id}.json"
    shap_json_path = report_dir / "shap_values.json"

    if not metrics_path.exists():
        logger.error(f"❌ Metrics not found at {metrics_path}. Run `evaluate` first.")
        sys.exit(1)

    with open(metrics_path) as f:
        metrics = _json.load(f)

    # Reconstruct plots dict
    metrics["plots"] = {
        "confusion_matrix": str(report_dir / f"confusion_matrix_{run_id}.png"),
        "roc_curve": str(report_dir / f"roc_curve_{run_id}.png"),
        "pr_curve": str(report_dir / f"pr_curve_{run_id}.png"),
        "score_distribution": str(report_dir / f"score_distribution_{run_id}.png"),
    }

    xai_results = {}
    if shap_json_path.exists():
        with open(shap_json_path) as f:
            shap_data = _json.load(f)
        xai_results["shap_feature_importance"] = shap_data.get("feature_importance", [])
        xai_results["shap_bar_plot"] = str(report_dir / "shap_bar.png")
        xai_results["shap_summary_plot"] = str(report_dir / "shap_summary.png")
        xai_results["shap_waterfall"] = str(report_dir / "shap_waterfall.png")

    lime_plots = sorted(report_dir.glob("lime_*.png"))
    xai_results["lime_explanations"] = [
        {"plot_path": str(p), "sample_index": i, "anomaly_score": 0}
        for i, p in enumerate(lime_plots)
    ]

    dataset_info = {
        "num_samples": metrics.get("total_samples", "?"),
        "benign_count": metrics.get("true_negatives", "?"),
        "attack_count": metrics.get("true_positives", "?"),
        "num_features": "?",
    }

    reporter = HTMLReporter(cfg)
    path = reporter.generate(metrics, dataset_info, xai_results, run_id)
    logger.info(f"📝 Report generated → {path}")
    print(f"Report: {path}")


def cmd_detect(args, cfg):
    from src.data_loader import DataLoader
    from src.model import AnomalyDetector
    from src.logger import get_logger
    import pandas as pd
    logger = get_logger("xai_ids.detect")

    if not args.input:
        logger.error("❌ --input required for detect command")
        sys.exit(1)

    loader = DataLoader.load_preprocessor(cfg, "models/latest/preprocessor.pkl")
    detector = AnomalyDetector.load(cfg, "latest")

    p = Path(args.input)
    df = pd.read_parquet(p) if p.suffix == ".parquet" else pd.read_csv(p)

    # Drop label if present
    label_col = cfg.data.label_column
    if label_col in df.columns:
        df = df.drop(columns=[label_col])

    X = loader.transform(df)
    preds, scores, _ = detector.predict(X)

    df["ANOMALY_SCORE"] = scores
    df["PREDICTION"] = np.where(preds == 1, "ANOMALY", "NORMAL")

    out = args.output or "results.csv"
    df.to_csv(out, index=False)
    logger.info(f"✅ Detected {preds.sum()} anomalies in {len(X)} samples → {out}")
    print(json.dumps({
        "total": int(len(X)),
        "anomalies": int(preds.sum()),
        "normal": int((preds == 0).sum()),
        "output": out,
    }, indent=2))


def cmd_serve(args, cfg):
    """Start FastAPI server."""
    try:
        import uvicorn
    except ImportError:
        print("❌ uvicorn not installed. Run: pip install uvicorn")
        sys.exit(1)

    from src.model import AnomalyDetector
    from src.data_loader import DataLoader
    from src.api import create_app

    detector = None
    preprocessor = None
    model_path = Path(cfg.model.persistence.model_dir) / "latest" / "autoencoder.keras"
    if model_path.exists():
        detector = AnomalyDetector.load(cfg, "latest")
        preprocessor = DataLoader.load_preprocessor(cfg)

    app = create_app(cfg, detector, preprocessor)

    host = args.host or cfg.api.host
    port = args.port or cfg.api.port
    print(f"🚀 XAI-IDS Pro API starting → http://{host}:{port}")
    print(f"📚 Docs: http://{host}:{port}/docs")
    uvicorn.run(app, host=host, port=port, log_level="info")


def cmd_dashboard(args, cfg):
    """Launch Streamlit dashboard."""
    import subprocess
    config_path = args.config or "config.yaml"
    print(f"🎯 Launching Streamlit dashboard …")
    subprocess.run([sys.executable, "-m", "streamlit", "run",
                    "src/dashboard.py", "--", f"--config={config_path}"])


def cmd_sweep(args, cfg):
    from src.data_loader import DataLoader
    from src.model import AnomalyDetector
    from src.evaluator import Evaluator

    run_id = args.run_id or "latest"
    loader = DataLoader.load_preprocessor(cfg, f"models/{run_id}/preprocessor.pkl")
    detector = AnomalyDetector.load(cfg, run_id)
    _, X_test, y_test, _ = loader.load_and_prepare(args.data or cfg.data.path, fit_scaler=False)
    _, scores, _ = detector.predict(X_test)
    evaluator = Evaluator(cfg)
    result = evaluator.threshold_sweep(y_test, scores)
    print(json.dumps(result, indent=2))


def cmd_info(args, cfg):
    from src.model import AnomalyDetector
    run_id = args.run_id or "latest"
    detector = AnomalyDetector.load(cfg, run_id)
    print(json.dumps({
        "run_id": run_id,
        "input_dim": detector.input_dim,
        "threshold": detector.threshold,
        "has_isolation_forest": detector.isolation_forest is not None,
    }, indent=2))


# ---------------------------------------------------------------------------
# CLI setup
# ---------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="xai-ids",
        description="XAI-IDS Pro — Explainable AI Intrusion Detection System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--config", default="config.yaml", help="Path to config YAML")

    sub = parser.add_subparsers(dest="command", required=True)

    # train
    p_train = sub.add_parser("train", help="Train the model")
    p_train.add_argument("--data", help="Override data path")
    p_train.add_argument("--run-id", help="Run identifier (default: timestamp)")

    # evaluate
    p_eval = sub.add_parser("evaluate", help="Evaluate model performance")
    p_eval.add_argument("--run-id", default="latest")
    p_eval.add_argument("--data", help="Override data path")

    # explain
    p_exp = sub.add_parser("explain", help="Run SHAP + LIME explanations")
    p_exp.add_argument("--run-id", default="latest")
    p_exp.add_argument("--data", help="Override data path")

    # report
    p_rep = sub.add_parser("report", help="Generate HTML report")
    p_rep.add_argument("--run-id", default="latest")

    # detect
    p_det = sub.add_parser("detect", help="Run inference on new data")
    p_det.add_argument("--input", required=True, help="Input CSV/Parquet file")
    p_det.add_argument("--output", default="results.csv", help="Output CSV")

    # serve
    p_srv = sub.add_parser("serve", help="Start REST API server")
    p_srv.add_argument("--host", default=None)
    p_srv.add_argument("--port", type=int, default=None)

    # dashboard
    p_dash = sub.add_parser("dashboard", help="Launch Streamlit dashboard")

    # sweep
    p_sweep = sub.add_parser("sweep", help="Threshold sweep for optimal F1")
    p_sweep.add_argument("--run-id", default="latest")
    p_sweep.add_argument("--data", help="Override data path")

    # info
    p_info = sub.add_parser("info", help="Show saved model info")
    p_info.add_argument("--run-id", default="latest")

    return parser


COMMANDS = {
    "train": cmd_train,
    "evaluate": cmd_evaluate,
    "explain": cmd_explain,
    "report": cmd_report,
    "detect": cmd_detect,
    "serve": cmd_serve,
    "dashboard": cmd_dashboard,
    "sweep": cmd_sweep,
    "info": cmd_info,
}


def main():
    parser = build_parser()
    args = parser.parse_args()

    from src.config import Config
    cfg = Config.from_yaml(args.config)

    # Ensure required dirs exist
    for d in ["logs", "models", "reports", "data"]:
        Path(d).mkdir(parents=True, exist_ok=True)

    fn = COMMANDS.get(args.command)
    if fn:
        fn(args, cfg)
    else:
        parser.print_help()


if __name__ == "__main__":
    main()
