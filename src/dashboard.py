"""
Streamlit Dashboard for XAI-IDS Pro.
Run with:  streamlit run src/dashboard.py -- --config config.yaml

Tabs:
  1. 📊 Live Detection   — upload CSV for instant detection
  2. 📈 Model Metrics    — view evaluation plots
  3. 🔍 SHAP Explorer    — global explanations
  4. 🔬 LIME Inspector   — per-sample local explanations
  5. ⚙️  Config          — view / edit configuration
"""
import sys
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


def run_dashboard(config_path: str = "config.yaml"):
    try:
        import streamlit as st
    except ImportError:
        print("❌ Streamlit not installed. Run: pip install streamlit")
        sys.exit(1)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from .config import Config
    from .model import AnomalyDetector
    from .data_loader import DataLoader
    from .explainer import XAIExplainer

    cfg = Config.from_yaml(config_path)

    st.set_page_config(
        page_title="XAI-IDS Pro",
        page_icon="🛡",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    # Custom CSS for dark cybersecurity theme
    st.markdown("""
    <style>
    [data-testid="stAppViewContainer"] {background: #0d1117;}
    [data-testid="stSidebar"] {background: #161b22;}
    h1, h2, h3 {color: #58a6ff !important;}
    .metric-label {color: #8b949e !important;}
    div[data-testid="metric-container"] {
        background: #161b22; border: 1px solid #30363d;
        border-radius: 8px; padding: 12px;
    }
    </style>
    """, unsafe_allow_html=True)

    # --- Sidebar ---
    st.sidebar.image("https://via.placeholder.com/200x60/2196F3/ffffff?text=XAI-IDS+Pro", width=200)
    st.sidebar.title("🛡 XAI-IDS Pro")
    st.sidebar.markdown("**Explainable AI Intrusion Detection**")
    st.sidebar.divider()

    # Model state
    if "detector" not in st.session_state:
        st.session_state.detector = None
    if "preprocessor" not in st.session_state:
        st.session_state.preprocessor = None
    if "explainer" not in st.session_state:
        st.session_state.explainer = None

    # Load existing model if available
    model_path = Path(cfg.model.persistence.model_dir) / "latest" / "autoencoder.keras"
    if st.session_state.detector is None and model_path.exists():
        try:
            st.session_state.detector = AnomalyDetector.load(cfg, run_id="latest")
            st.session_state.preprocessor = DataLoader.load_preprocessor(cfg)
            st.sidebar.success("✅ Model loaded")
        except Exception as e:
            st.sidebar.warning(f"⚠️ Could not load model: {e}")

    model_status = "🟢 Loaded" if st.session_state.detector else "🔴 Not Loaded"
    st.sidebar.metric("Model Status", model_status)
    st.sidebar.divider()

    # Tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Live Detection",
        "📈 Model Metrics",
        "🔍 SHAP Global",
        "🔬 LIME Local",
        "⚙️  Config"
    ])

    # ===========================================================================
    # TAB 1: Live Detection
    # ===========================================================================
    with tab1:
        st.header("📊 Live Network Traffic Detection")

        if st.session_state.detector is None:
            st.warning("⚠️ No model loaded. Train one first using the CLI: `python main.py train`")
            st.stop()

        uploaded = st.file_uploader("Upload network traffic CSV or Parquet", type=["csv", "parquet"])

        col1, col2, col3 = st.columns(3)
        col1.metric("Model Threshold", f"{st.session_state.detector.threshold:.6f}")
        col2.metric("Input Features", st.session_state.detector.input_dim)
        col3.metric("Ensemble", "Yes" if st.session_state.detector.isolation_forest else "No")

        if uploaded:
            with st.spinner("Processing …"):
                try:
                    if uploaded.name.endswith(".csv"):
                        df_in = pd.read_csv(uploaded)
                    else:
                        df_in = pd.read_parquet(uploaded)

                    pre = st.session_state.preprocessor
                    det = st.session_state.detector

                    # Drop label if present
                    label_col = cfg.data.label_column
                    y_true = None
                    if label_col in df_in.columns:
                        labels = df_in[label_col].astype(str).str.strip().str.lower()
                        y_true = np.where(labels.str.contains(cfg.data.benign_label), 0, 1)
                        df_in = df_in.drop(columns=[label_col])

                    X = pre.transform(df_in)
                    preds, scores, details = det.predict(X)

                    df_out = df_in.copy()
                    df_out["ANOMALY_SCORE"] = scores
                    df_out["PREDICTION"] = np.where(preds == 1, "🔴 ANOMALY", "🟢 NORMAL")

                    anomaly_pct = 100 * preds.mean()
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("Total Samples", len(X))
                    c2.metric("Anomalies", int(preds.sum()), delta=f"{anomaly_pct:.1f}%")
                    c3.metric("Max Score", f"{scores.max():.4f}")
                    c4.metric("Mean Score", f"{scores.mean():.4f}")

                    st.subheader("Score Distribution")
                    fig, ax = plt.subplots(figsize=(10, 3))
                    ax.hist(scores[preds == 0], bins=60, alpha=0.6, color="#4CAF50", label="Normal")
                    ax.hist(scores[preds == 1], bins=60, alpha=0.6, color="#F44336", label="Anomaly")
                    ax.axvline(det.threshold, color="#FF9800", linestyle="--", label="Threshold")
                    ax.set_facecolor("#0d1117"); fig.patch.set_facecolor("#0d1117")
                    ax.tick_params(colors="#e6edf3"); ax.legend(facecolor="#161b22", labelcolor="#e6edf3")
                    for spine in ax.spines.values():
                        spine.set_edgecolor("#30363d")
                    st.pyplot(fig)
                    plt.close()

                    st.subheader("Detections")
                    st.dataframe(
                        df_out.sort_values("ANOMALY_SCORE", ascending=False).head(200),
                        use_container_width=True
                    )

                    csv = df_out.to_csv(index=False)
                    st.download_button("⬇️ Download Results CSV", data=csv,
                                       file_name="xai_ids_results.csv", mime="text/csv")

                except Exception as e:
                    st.error(f"❌ Error processing file: {e}")

    # ===========================================================================
    # TAB 2: Model Metrics
    # ===========================================================================
    with tab2:
        st.header("📈 Evaluation Metrics")
        report_dir = Path(cfg.evaluation.report_dir)
        metric_files = sorted(report_dir.glob("metrics_*.json"), reverse=True)

        if not metric_files:
            st.info("No evaluation results found. Run `python main.py evaluate` first.")
        else:
            selected = st.selectbox("Select run", [f.name for f in metric_files])
            with open(report_dir / selected) as f:
                m = json.load(f)

            c1, c2, c3, c4, c5 = st.columns(5)
            c1.metric("F1 Score", f"{m['f1']:.4f}")
            c2.metric("Precision", f"{m['precision']:.4f}")
            c3.metric("Recall", f"{m['recall']:.4f}")
            c4.metric("ROC-AUC", f"{m.get('roc_auc', 'N/A')}")
            c5.metric("FP Rate", f"{m['false_positive_rate']:.4f}")

            # Show saved plots
            run_id = selected.replace("metrics_", "").replace(".json", "")
            for key, label in [
                ("confusion_matrix", "Confusion Matrix"),
                ("roc_curve", "ROC Curve"),
                ("pr_curve", "PR Curve"),
                ("score_distribution", "Score Distribution"),
            ]:
                plot_path = report_dir / f"{key}_{run_id}.png"
                if plot_path.exists():
                    st.image(str(plot_path), caption=label)

    # ===========================================================================
    # TAB 3: SHAP Global
    # ===========================================================================
    with tab3:
        st.header("🔍 SHAP Global Feature Importances")
        report_dir = Path(cfg.evaluation.report_dir)

        shap_json = report_dir / "shap_values.json"
        if shap_json.exists():
            with open(shap_json) as f:
                shap_data = json.load(f)
            fi = pd.DataFrame(shap_data["feature_importance"])
            st.dataframe(fi, use_container_width=True)

        for fname, label in [
            ("shap_bar.png", "Feature Importance (Bar)"),
            ("shap_summary.png", "Summary Plot (Beeswarm)"),
            ("shap_waterfall.png", "Waterfall — Most Anomalous"),
        ]:
            p = report_dir / fname
            if p.exists():
                st.image(str(p), caption=label)
            else:
                st.info(f"{label} not available. Run full pipeline first.")

    # ===========================================================================
    # TAB 4: LIME Local
    # ===========================================================================
    with tab4:
        st.header("🔬 LIME Local Explanations")
        report_dir = Path(cfg.evaluation.report_dir)
        lime_plots = sorted(report_dir.glob("lime_*.png"))

        if not lime_plots:
            st.info("No LIME plots found. Run `python main.py explain` first.")
        else:
            for p in lime_plots:
                st.image(str(p), caption=p.stem)

    # ===========================================================================
    # TAB 5: Config
    # ===========================================================================
    with tab5:
        st.header("⚙️ Configuration")
        st.json(cfg.dict())

    st.sidebar.markdown("---")
    st.sidebar.caption("XAI-IDS Pro v1.0.0")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    run_dashboard(args.config)
