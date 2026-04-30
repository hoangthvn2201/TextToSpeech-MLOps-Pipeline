from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlflow

from tts_pipeline.utils.logging import get_logger

logger = get_logger(__name__)

_GRADE_ORDER = ["A", "B", "C", "D", "F"]


def check_gates(
    metrics: dict[str, float],
    absolute_floor: dict[str, float],
    relative_delta: dict[str, float],
    baseline_metrics: dict[str, float] | None,
) -> tuple[bool, list[str]]:
    failures: list[str] = []

    # Absolute floor checks
    floor_checks = [
        ("utmos_mean", metrics.get("utmos_mean", 0.0), absolute_floor.get("utmos_mean", 3.0), ">="),
        ("cer", metrics.get("cer", 1.0), absolute_floor.get("cer", 0.20), "<="),
        ("stoi_mean", metrics.get("stoi_mean", 0.0), absolute_floor.get("stoi_mean", 0.60), ">="),
        ("rtf_p95", metrics.get("rtf_p95", 1.0), absolute_floor.get("rtf_p95", 0.20), "<="),
    ]
    for name, value, threshold, op in floor_checks:
        if op == ">=" and value < threshold:
            failures.append(f"ABSOLUTE FLOOR: {name}={value:.4f} < {threshold}")
        elif op == "<=" and value > threshold:
            failures.append(f"ABSOLUTE FLOOR: {name}={value:.4f} > {threshold}")

    # Relative delta checks (only if baseline exists)
    if baseline_metrics:
        for key, max_delta in relative_delta.items():
            candidate = metrics.get(key)
            base = baseline_metrics.get(key)
            if candidate is None or base is None or base == 0:
                continue
            delta = (candidate - base) / abs(base)
            if delta < max_delta:  # improvement or within tolerance
                continue
            failures.append(f"REGRESSION: {key} delta={delta:.4f} > allowed {max_delta:.4f} vs baseline {base:.4f}")

    return len(failures) == 0, failures


def generate_report(
    metrics: dict[str, float],
    baseline_metrics: dict[str, float] | None,
    synthesis_results: dict[str, dict[str, Any]],
    absolute_floor: dict[str, float],
    relative_delta: dict[str, float],
    report_dir: Path,
    model_info: dict[str, str] | None = None,
) -> dict[str, Any]:
    report_dir.mkdir(parents=True, exist_ok=True)

    passed, failures = check_gates(metrics, absolute_floor, relative_delta, baseline_metrics)

    report: dict[str, Any] = {
        "pass": passed,
        "failures": failures,
        "metrics": metrics,
        "baseline_metrics": baseline_metrics,
        "model_info": model_info or {},
        "n_synthesized": len(synthesis_results),
    }

    # JSON report
    json_path = report_dir / "evaluation_report.json"
    with open(json_path, "w") as f:
        json.dump(report, f, indent=2)

    # HTML report
    html_path = report_dir / "evaluation_report.html"
    _write_html_report(html_path, report, synthesis_results)

    # Log to MLflow
    if mlflow.active_run():
        mlflow.log_metrics(metrics)
        mlflow.log_artifact(str(json_path))
        mlflow.log_artifact(str(html_path))

    status = "PASSED" if passed else "FAILED"
    logger.info("Evaluation gate: %s | UTMOS=%.3f CER=%.3f STOI=%.3f RTF_p95=%.3f",
                status,
                metrics.get("utmos_mean", 0),
                metrics.get("cer", 1),
                metrics.get("stoi_mean", 0),
                metrics.get("rtf_p95", 1))
    if failures:
        for f in failures:
            logger.warning("  Gate failure: %s", f)

    return report


def _write_html_report(path: Path, report: dict[str, Any], synthesis_results: dict[str, dict[str, Any]]) -> None:
    passed = report["pass"]
    status_color = "#2ecc71" if passed else "#e74c3c"
    status_text = "PASSED" if passed else "FAILED"

    metrics_rows = "".join(
        f"<tr><td>{k}</td><td>{v:.4f}</td></tr>"
        for k, v in report.get("metrics", {}).items()
    )
    failures_html = ""
    if report.get("failures"):
        failures_html = "<h3>Gate Failures</h3><ul>" + "".join(
            f"<li style='color:red'>{f}</li>" for f in report["failures"]
        ) + "</ul>"

    audio_rows = ""
    for uid, info in list(synthesis_results.items())[:20]:
        wav_rel = Path(str(info["wav_path"])).name
        audio_rows += f"<tr><td>{uid}</td><td><audio controls src='synthesized/{wav_rel}'></audio></td><td>{info.get('rtf', 0):.3f}</td></tr>"

    html = f"""<!DOCTYPE html>
<html><head><meta charset='utf-8'><title>TTS Evaluation Report</title>
<style>body{{font-family:sans-serif;margin:2rem}}table{{border-collapse:collapse;width:100%}}
td,th{{border:1px solid #ddd;padding:8px}}tr:nth-child(even){{background:#f9f9f9}}</style></head>
<body>
<h1>TTS Evaluation Report</h1>
<h2 style='color:{status_color}'>Status: {status_text}</h2>
{failures_html}
<h3>Metrics</h3>
<table><tr><th>Metric</th><th>Value</th></tr>{metrics_rows}</table>
<h3>Audio Samples (first 20)</h3>
<table><tr><th>ID</th><th>Audio</th><th>RTF</th></tr>{audio_rows}</table>
</body></html>"""
    path.write_text(html)
