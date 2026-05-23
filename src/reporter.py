"""
HTML Report Generator — produces a standalone, self-contained HTML report
with embedded plots (base64), metrics table, and LIME/SHAP summaries.
"""
import base64
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

from .logger import get_logger

logger = get_logger("xai_ids.reporter")


def _img_to_b64(path: str) -> str:
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _img_tag(path: str, alt: str = "", width: str = "100%") -> str:
    if not Path(path).exists():
        return f"<p style='color:gray'>[{alt} — not generated]</p>"
    b64 = _img_to_b64(path)
    ext = Path(path).suffix.lstrip(".")
    return (f'<img src="data:image/{ext};base64,{b64}" '
            f'alt="{alt}" style="width:{width};border-radius:8px;margin:8px 0"/>')


TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<title>XAI-IDS Pro — Detection Report</title>
<style>
  :root{{--accent:#2196F3;--good:#4CAF50;--bad:#F44336;--warn:#FF9800;}}
  body{{font-family:'Segoe UI',sans-serif;background:#0d1117;color:#e6edf3;margin:0;padding:0}}
  header{{background:var(--accent);padding:24px 40px;}}
  header h1{{margin:0;font-size:2rem}}
  header p{{margin:4px 0 0;opacity:.8}}
  .container{{max-width:1200px;margin:auto;padding:32px 20px}}
  .card{{background:#161b22;border:1px solid #30363d;border-radius:12px;padding:24px;margin:24px 0}}
  .card h2{{margin:0 0 16px;color:var(--accent);font-size:1.2rem;border-bottom:1px solid #30363d;padding-bottom:8px}}
  .metric-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(160px,1fr));gap:16px}}
  .metric{{background:#0d1117;border-radius:8px;padding:16px;text-align:center}}
  .metric .val{{font-size:2rem;font-weight:700;color:var(--accent)}}
  .metric .lbl{{font-size:.8rem;color:#8b949e;margin-top:4px}}
  table{{width:100%;border-collapse:collapse;font-size:.9rem}}
  th{{background:#21262d;padding:10px;text-align:left;color:#8b949e}}
  td{{padding:10px;border-bottom:1px solid #21262d}}
  .badge{{display:inline-block;padding:2px 10px;border-radius:20px;font-size:.75rem;font-weight:600}}
  .badge.good{{background:#1a3a1a;color:var(--good)}}
  .badge.bad{{background:#3a1a1a;color:var(--bad)}}
  .badge.warn{{background:#3a2a10;color:var(--warn)}}
  .img-grid{{display:grid;grid-template-columns:repeat(auto-fill,minmax(480px,1fr));gap:16px}}
  pre{{background:#0d1117;border:1px solid #30363d;border-radius:6px;padding:12px;overflow-x:auto;font-size:.8rem}}
  footer{{text-align:center;padding:24px;color:#8b949e;font-size:.8rem}}
</style>
</head>
<body>
<header>
  <h1>🛡 XAI-IDS Pro — Intrusion Detection Report</h1>
  <p>Generated: {timestamp} &nbsp;|&nbsp; Run ID: {run_id}</p>
</header>
<div class="container">

<!-- DATASET SUMMARY -->
<div class="card">
  <h2>📂 Dataset Summary</h2>
  <div class="metric-grid">
    <div class="metric"><div class="val">{total_samples}</div><div class="lbl">Total Samples</div></div>
    <div class="metric"><div class="val">{benign_count}</div><div class="lbl">Benign</div></div>
    <div class="metric"><div class="val">{attack_count}</div><div class="lbl">Attacks</div></div>
    <div class="metric"><div class="val">{num_features}</div><div class="lbl">Features</div></div>
  </div>
</div>

<!-- DETECTION METRICS -->
<div class="card">
  <h2>📊 Detection Performance</h2>
  <div class="metric-grid">
    <div class="metric"><div class="val" style="color:{f1_color}">{f1}</div><div class="lbl">F1 Score</div></div>
    <div class="metric"><div class="val">{precision}</div><div class="lbl">Precision</div></div>
    <div class="metric"><div class="val">{recall}</div><div class="lbl">Recall (Detection Rate)</div></div>
    <div class="metric"><div class="val">{roc_auc}</div><div class="lbl">ROC-AUC</div></div>
    <div class="metric"><div class="val" style="color:var(--bad)">{fpr}</div><div class="lbl">False Positive Rate</div></div>
    <div class="metric"><div class="val">{tp}</div><div class="lbl">True Positives</div></div>
    <div class="metric"><div class="val">{fp}</div><div class="lbl">False Positives</div></div>
    <div class="metric"><div class="val">{fn}</div><div class="lbl">False Negatives</div></div>
  </div>
</div>

<!-- PLOTS -->
<div class="card">
  <h2>📈 Visualizations</h2>
  <div class="img-grid">
    {confusion_matrix_img}
    {roc_img}
    {pr_img}
    {score_dist_img}
  </div>
</div>

<!-- SHAP -->
<div class="card">
  <h2>🔍 SHAP — Global Feature Importances</h2>
  <div class="img-grid">
    {shap_bar_img}
    {shap_summary_img}
    {shap_waterfall_img}
  </div>
  <h3 style="margin-top:20px;color:#8b949e">Top Features</h3>
  {shap_table}
</div>

<!-- LIME -->
<div class="card">
  <h2>🔬 LIME — Local Explanations (Top Anomalies)</h2>
  {lime_section}
</div>

<!-- MODEL INFO -->
<div class="card">
  <h2>⚙️ Model Configuration</h2>
  <pre>{model_config_json}</pre>
</div>

</div>
<footer>XAI-IDS Pro v1.0.0 &mdash; Explainable AI Intrusion Detection System</footer>
</body>
</html>"""


class HTMLReporter:
    def __init__(self, config):
        self.cfg = config
        self.report_dir = Path(config.evaluation.report_dir)
        self.report_dir.mkdir(parents=True, exist_ok=True)

    def generate(
        self,
        metrics: Dict[str, Any],
        dataset_info: Dict[str, Any],
        xai_results: Dict[str, Any],
        run_id: str = "run",
    ) -> str:
        plots = metrics.get("plots", {})
        lime_explanations = xai_results.get("lime_explanations", [])
        shap_fi = xai_results.get("shap_feature_importance", [])

        # --- SHAP feature importance table ---
        if shap_fi:
            rows = "".join(
                f"<tr><td>{i+1}</td><td>{r['feature']}</td>"
                f"<td>{r['importance']:.6f}</td></tr>"
                for i, r in enumerate(shap_fi[:15])
            )
            shap_table = (f"<table><tr><th>#</th><th>Feature</th><th>Mean |SHAP|</th></tr>"
                          f"{rows}</table>")
        else:
            shap_table = "<p style='color:gray'>SHAP not computed</p>"

        # --- LIME section ---
        lime_html_parts = []
        for exp in lime_explanations:
            idx = exp.get("sample_index", "?")
            score = exp.get("anomaly_score", 0)
            plot = exp.get("plot_path", "")
            lime_html_parts.append(
                f"<div style='margin-bottom:20px'>"
                f"<p><strong>Sample #{idx}</strong> — Anomaly Score: "
                f"<span class='badge bad'>{score:.4f}</span></p>"
                f"{_img_tag(plot, alt='LIME plot', width='100%')}"
                f"</div>"
            )
        lime_section = "".join(lime_html_parts) or "<p style='color:gray'>LIME not computed</p>"

        f1_val = metrics.get("f1", 0)
        f1_color = "var(--good)" if f1_val > 0.8 else ("var(--warn)" if f1_val > 0.5 else "var(--bad)")

        roc_str = f"{metrics.get('roc_auc', 0):.4f}" if metrics.get("roc_auc") else "N/A"

        html = TEMPLATE.format(
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            run_id=run_id,
            total_samples=dataset_info.get("num_samples", "?"),
            benign_count=dataset_info.get("benign_count", "?"),
            attack_count=dataset_info.get("attack_count", "?"),
            num_features=dataset_info.get("num_features", "?"),
            f1=f"{f1_val:.4f}",
            f1_color=f1_color,
            precision=f"{metrics.get('precision', 0):.4f}",
            recall=f"{metrics.get('recall', 0):.4f}",
            roc_auc=roc_str,
            fpr=f"{metrics.get('false_positive_rate', 0):.4f}",
            tp=metrics.get("true_positives", 0),
            fp=metrics.get("false_positives", 0),
            fn=metrics.get("false_negatives", 0),
            confusion_matrix_img=_img_tag(plots.get("confusion_matrix", ""), "Confusion Matrix"),
            roc_img=_img_tag(plots.get("roc_curve", ""), "ROC Curve"),
            pr_img=_img_tag(plots.get("pr_curve", ""), "PR Curve"),
            score_dist_img=_img_tag(plots.get("score_distribution", ""), "Score Distribution"),
            shap_bar_img=_img_tag(xai_results.get("shap_bar_plot", ""), "SHAP Bar"),
            shap_summary_img=_img_tag(xai_results.get("shap_summary_plot", ""), "SHAP Summary"),
            shap_waterfall_img=_img_tag(xai_results.get("shap_waterfall", ""), "SHAP Waterfall"),
            shap_table=shap_table,
            lime_section=lime_section,
            model_config_json=json.dumps(self.cfg.model.dict(), indent=2),
        )

        out = self.report_dir / f"report_{run_id}.html"
        with open(out, "w", encoding="utf-8") as f:
            f.write(html)
        logger.info(f"📝 HTML report saved → {out}")
        return str(out)
