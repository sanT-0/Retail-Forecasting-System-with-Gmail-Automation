"""
╔══════════════════════════════════════════════════════════════════════════════╗
║  Retail Sales Forecasting — Flask API                                        ║
║  Created by: sanT                                                            ║
║                                                                              ║
║  This API wraps the main.py run_pipeline() function and exposes it over      ║
║  HTTP so that n8n can call it with a CSV attachment and receive back:        ║
║    • A structured JSON forecast summary                                      ║
║    • Paths to output chart images (accessible via shared Docker volume)      ║
║                                                                              ║
║  Endpoints:                                                                  ║
║    GET  /health              → Health check                                  ║
║    POST /forecast            → Run ML pipeline on uploaded CSV               ║
║    GET  /outputs/<filename>  → Serve output image files                      ║
╚══════════════════════════════════════════════════════════════════════════════╝
"""

import os
import sys
import json
import uuid
import shutil
import traceback
import logging
from datetime import datetime
from pathlib import Path

from flask import Flask, request, jsonify, send_from_directory, abort

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S"
)
log = logging.getLogger("forecast_api")

# ── App Init ──────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024   # 100 MB max upload

# ── Paths ─────────────────────────────────────────────────────────────────────
# The project is bind-mounted into /app/project inside the container
PROJECT_DIR  = os.environ.get("PROJECT_DIR", "/app/project")
SHARED_DIR   = os.environ.get("SHARED_DIR",  "/shared")
UPLOAD_DIR   = os.path.join(SHARED_DIR, "uploads")
OUTPUT_DIR   = os.path.join(SHARED_DIR, "outputs")

os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Inject project into Python path so we can import main.py / utils ──────────
if PROJECT_DIR not in sys.path:
    sys.path.insert(0, PROJECT_DIR)

# ── Import pipeline (lazy — catches import errors gracefully) ──────────────────
_pipeline_ready = False
_pipeline_error  = None
run_pipeline = None
get_strategic_recommendations = None

def _try_import_pipeline():
    global _pipeline_ready, _pipeline_error
    try:
        # Suppress Streamlit-specific imports that don't exist in API context
        import pandas as pd
        try:
            pd.options.future.infer_string = False
        except Exception:
            pass

        global run_pipeline, get_strategic_recommendations
        from main import run_pipeline   # noqa: F401
        from utils.insights import get_strategic_recommendations
        _pipeline_ready = True
        log.info("✅ Pipeline imported successfully from %s", PROJECT_DIR)
    except Exception as exc:
        _pipeline_error = str(exc)
        log.error("❌ Pipeline import failed: %s", exc)

_try_import_pipeline()


# ─────────────────────────────────────────────────────────────────────────────
# UTILITY FUNCTIONS
# ─────────────────────────────────────────────────────────────────────────────

def _build_text_summary(dataset_name: str, results: dict, forecast_periods: int, mode: str) -> str:
    """Build a human-readable email-friendly forecast summary."""
    import numpy as np

    best_model   = results.get("best_model")
    best_name    = getattr(best_model, "name", str(best_model))
    metrics_list = results.get("metrics", [])
    future_preds = results.get("future_preds", [])
    future_dates = results.get("future_dates", [])
    series       = results.get("series")

    # Best metrics row
    best_metrics = {}
    if metrics_list:
        best_metrics = min(metrics_list, key=lambda x: float(x.get("RMSE", 9e12)))

    # Forecast totals
    total_forecast = float(np.sum(future_preds)) if len(future_preds) > 0 else 0
    avg_forecast   = float(np.mean(future_preds)) if len(future_preds) > 0 else 0

    # Historical comparison
    pct_change_str = ""
    if series is not None and len(series) > 0:
        avg_hist   = float(np.mean(series))
        pct_change = ((avg_forecast - avg_hist) / avg_hist) * 100 if avg_hist != 0 else 0
        direction  = "higher" if pct_change >= 0 else "lower"
        pct_change_str = f"\n📊 Market Trend     : {abs(pct_change):.1f}% {direction} than historical average"

    # Date range
    date_from = str(future_dates[0].date())  if len(future_dates) > 0 else "N/A"
    date_to   = str(future_dates[-1].date()) if len(future_dates) > 0 else "N/A"

    # Format INR values
    def fmt(v):
        if v >= 1e7:  return f"₹{v/1e7:.2f} Cr"
        if v >= 1e5:  return f"₹{v/1e5:.2f} L"
        return f"₹{v:,.2f}"

    # Model leaderboard (top 5)
    leaderboard = ""
    if metrics_list:
        sorted_m = sorted(metrics_list, key=lambda x: float(x.get("RMSE", 9e12)))[:5]
        leaderboard = "\n\n📋 TOP 5 MODEL LEADERBOARD\n" + "─" * 50
        for i, m in enumerate(sorted_m, 1):
            medal = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣"][i - 1]
            leaderboard += f"\n{medal} {m.get('Model', 'N/A'):<28} RMSE: {float(m.get('RMSE', 0)):>12,.2f}  R²: {float(m.get('R2', 0)):.4f}"

    summary = f"""
🛒 RETAIL SALES FORECAST REPORT
{"=" * 55}
📁 Dataset         : {dataset_name}
🤖 Best Model      : {best_name}
⚡ Training Mode   : {mode.upper()}
📅 Generated At    : {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}

📈 FORECAST SUMMARY ({forecast_periods} periods ahead)
{"─" * 55}
🗓️  Forecast From   : {date_from}
🗓️  Forecast To     : {date_to}
💰 Total Forecast  : {fmt(total_forecast)}
📊 Avg / Period    : {fmt(avg_forecast)}{pct_change_str}

🏆 BEST MODEL PERFORMANCE
{"─" * 55}
Model              : {best_name}
RMSE               : {float(best_metrics.get('RMSE', 0)):,.2f}
MAE                : {float(best_metrics.get('MAE', 0)):,.2f}
R² Score           : {float(best_metrics.get('R2', 0)):.4f}
{leaderboard}

{"=" * 55}
📌 Please find the forecast charts attached.
   Powered by Future Retail Sales Forecasting v2.1
   Created by sanT
{"=" * 55}
""".strip()

    return summary


def _copy_charts_to_shared(out_dir: str, run_id: str) -> list[str]:
    """Copy pipeline output PNGs to the shared volume and return their filenames."""
    shared_run_dir = os.path.join(OUTPUT_DIR, run_id)
    os.makedirs(shared_run_dir, exist_ok=True)

    chart_files = []
    if not os.path.isdir(out_dir):
        return chart_files

    # Priority order for key charts (attach these first in email)
    priority_keywords = ["forecast", "comparison", "error_dist", "period_revenue", "time_series"]
    all_pngs = sorted(Path(out_dir).glob("*.png"))

    def _priority(p):
        name = p.stem.lower()
        for i, kw in enumerate(priority_keywords):
            if kw in name:
                return i
        return len(priority_keywords)

    sorted_pngs = sorted(all_pngs, key=_priority)

    for src in sorted_pngs:
        dst_name = f"{run_id}_{src.name}"
        dst_path = os.path.join(shared_run_dir, dst_name)
        shutil.copy2(str(src), dst_path)
        chart_files.append({"filename": dst_name, "run_id": run_id, "path": dst_path})
        log.info("  Copied chart: %s → %s", src.name, dst_path)

    return chart_files


# ─────────────────────────────────────────────────────────────────────────────
# ROUTES
# ─────────────────────────────────────────────────────────────────────────────

@app.route("/health", methods=["GET"])
def health():
    """Health check endpoint — n8n polls this before calling /forecast."""
    return jsonify({
        "status"          : "ok",
        "pipeline_ready"  : _pipeline_ready,
        "pipeline_error"  : _pipeline_error,
        "timestamp"       : datetime.utcnow().isoformat() + "Z",
        "project_dir"     : PROJECT_DIR,
        "shared_dir"      : SHARED_DIR,
    }), 200


@app.route("/forecast", methods=["POST"])
def forecast():
    """
    Main endpoint — accepts a CSV file via multipart form and runs the ML pipeline.

    Form Fields:
        file           (required) — CSV dataset file
        mode           (optional) — "quick" or "full"    [default: quick]
        periods        (optional) — Forecast periods      [default: 30]
        dataset_name   (optional) — Label for output dir  [default: from filename]

    Returns JSON:
        {
            "success"        : true,
            "run_id"         : "abc123",
            "dataset"        : "my_sales",
            "best_model"     : "Gradient Boosting",
            "rmse"           : 1234.56,
            "r2"             : 0.9234,
            "total_forecast" : 45000.00,
            "avg_forecast"   : 1500.00,
            "forecast_periods": 30,
            "date_from"      : "2024-02-01",
            "date_to"        : "2024-03-01",
            "summary_text"   : "... human-readable report ...",
            "charts"         : [{"filename": "abc_forecast.png", "run_id": "abc"}],
            "elapsed_s"      : 45.2
        }
    """
    import time
    import numpy as np

    if not _pipeline_ready:
        return jsonify({
            "success" : False,
            "error"   : f"Pipeline not ready: {_pipeline_error}"
        }), 503

    # ── Validate input ────────────────────────────────────────────────────────
    if "file" not in request.files:
        return jsonify({"success": False, "error": "No file attached. Send as multipart/form-data with field 'file'."}), 400

    uploaded = request.files["file"]
    if uploaded.filename == "":
        return jsonify({"success": False, "error": "Empty filename."}), 400
    if not uploaded.filename.lower().endswith(".csv"):
        return jsonify({"success": False, "error": "Only CSV files are accepted."}), 400

    # ── Parse parameters ──────────────────────────────────────────────────────
    mode            = request.form.get("mode",         "quick").strip().lower()
    forecast_periods = int(request.form.get("periods",  "30"))
    dataset_name    = request.form.get("dataset_name", "").strip()
    
    date_col_req    = request.form.get("date_col",     "").strip()
    target_col_req  = request.form.get("target_col",   "").strip()
    freq_req        = request.form.get("freq",         "").strip()

    if mode not in ("quick", "full"):
        mode = "quick"
    forecast_periods = max(1, min(forecast_periods, 365))

    # ── Generate unique run ID ────────────────────────────────────────────────
    run_id = uuid.uuid4().hex[:8]
    if not dataset_name:
        dataset_name = os.path.splitext(uploaded.filename)[0].replace(" ", "_")

    log.info("▶  Run [%s] | dataset=%s | mode=%s | periods=%d", run_id, dataset_name, mode, forecast_periods)

    # ── Save uploaded CSV ─────────────────────────────────────────────────────
    upload_path = os.path.join(UPLOAD_DIR, f"{run_id}_{uploaded.filename}")
    uploaded.save(upload_path)
    log.info("   Saved upload: %s", upload_path)

    # ── Build Custom Schema (if mapped) ───────────────────────────────────────
    custom_schema = None
    if target_col_req:
        custom_schema = {
            "required_cols": [date_col_req] if date_col_req else [],
            "date_col": date_col_req if date_col_req else None,
            "target": target_col_req,
            "freq": freq_req if freq_req else "D",
            "description": f"Custom Mapped ({dataset_name})"
        }
        # Required cols
        if date_col_req:
            custom_schema["required_cols"].append(target_col_req)
        else:
            custom_schema["required_cols"] = [target_col_req]

    # ── Run pipeline ──────────────────────────────────────────────────────────
    t0 = time.time()
    try:
        results = run_pipeline(
            dataset_name     = dataset_name,
            forecast_periods = forecast_periods,
            mode             = mode,
            custom_csv       = upload_path,
            custom_schema    = custom_schema,  # Uses user mapping or auto-detect if None
            save_models      = False,  # skip model serialisation in API mode
            skip_heavy_viz   = True,   # prevent timeouts in API calls
        )
    except Exception as exc:
        tb = traceback.format_exc()
        log.error("Pipeline error [%s]: %s\n%s", run_id, exc, tb)
        return jsonify({
            "success"  : False,
            "run_id"   : run_id,
            "error"    : str(exc),
            "traceback": tb,
        }), 500
    finally:
        pass  # keep the upload file for debugging

    elapsed = round(time.time() - t0, 2)
    log.info("   Pipeline done in %.1fs", elapsed)

    # ── Extract key metrics ───────────────────────────────────────────────────
    best_model   = results.get("best_model")
    best_name    = getattr(best_model, "name", str(best_model))
    metrics_list = results.get("metrics", [])
    future_preds = results.get("future_preds", [])
    future_dates = results.get("future_dates", [])

    best_metrics = {}
    if metrics_list:
        best_metrics = min(metrics_list, key=lambda x: float(x.get("RMSE", 9e12)))

    total_forecast = float(np.sum(future_preds)) if len(future_preds) > 0 else 0
    avg_forecast   = float(np.mean(future_preds)) if len(future_preds) > 0 else 0
    date_from      = str(future_dates[0].date())  if len(future_dates) > 0 else ""
    date_to        = str(future_dates[-1].date()) if len(future_dates) > 0 else ""

    # ── Calculate Market Trend ───────────────────────────────────────────────
    market_trend_pct = 0.0
    market_trend_direction = "higher"
    series = results.get("series")
    if series is not None and len(series) > 0:
        avg_hist = float(np.mean(series))
        if avg_hist != 0:
            market_trend_pct = ((avg_forecast - avg_hist) / avg_hist) * 100
            market_trend_direction = "higher" if market_trend_pct >= 0 else "lower"

    # ── Copy charts to shared volume ──────────────────────────────────────────
    out_dir    = results.get("out_dir", "")
    chart_info = _copy_charts_to_shared(out_dir, run_id)
    log.info("   Copied %d chart(s) to shared volume", len(chart_info))

    # ── 🚀 Dynamic Strategic Growth Recommendations ─────────────────────────────
    recommendations = get_strategic_recommendations(results)
    log.info("   Generated %d growth recommendation(s)", len(recommendations))

    # ── Build human-readable summary ──────────────────────────────────────────
    summary_text = _build_text_summary(dataset_name, results, forecast_periods, mode)

    # ── Build leaderboard for JSON ────────────────────────────────────────────
    leaderboard = []
    for m in sorted(metrics_list, key=lambda x: float(x.get("RMSE", 9e12)))[:10]:
        leaderboard.append({
            "model": m.get("Model", ""),
            "rmse" : round(float(m.get("RMSE", 0)), 4),
            "mae"  : round(float(m.get("MAE",  0)), 4),
            "r2"   : round(float(m.get("R2",   0)), 4),
        })

    # ── Response ──────────────────────────────────────────────────────────────
    response = {
        "success"         : True,
        "run_id"          : run_id,
        "dataset"         : dataset_name,
        "best_model"      : best_name,
        "rmse"            : round(float(best_metrics.get("RMSE", 0)), 4),
        "mae"             : round(float(best_metrics.get("MAE",  0)), 4),
        "r2"              : round(float(best_metrics.get("R2",   0)), 4),
        "total_forecast"  : round(total_forecast, 2),
        "avg_forecast"    : round(avg_forecast,   2),
        "forecast_periods": forecast_periods,
        "mode"            : mode,
        "date_from"       : date_from,
        "date_to"         : date_to,
        "market_trend_pct": round(abs(market_trend_pct), 1),
        "market_trend_direction": market_trend_direction,
        "elapsed_s"       : elapsed,
        "summary_text"    : summary_text,
        "leaderboard"     : leaderboard,
        "charts"          : chart_info,
        "chart_count"     : len(chart_info),
        "recommendations" : recommendations,
    }

    log.info("✅ Run [%s] complete | best=%s | RMSE=%.2f | charts=%d",
             run_id, best_name, float(best_metrics.get("RMSE", 0)), len(chart_info))

    return jsonify(response), 200


@app.route("/outputs/<run_id>/<filename>", methods=["GET"])
def serve_output(run_id: str, filename: str):
    """Serve a chart PNG from the shared volume (n8n uses this to fetch images)."""
    run_dir = os.path.join(OUTPUT_DIR, run_id)
    if not os.path.isdir(run_dir):
        abort(404, description=f"Run '{run_id}' not found.")
    file_path = os.path.join(run_dir, filename)
    if not os.path.isfile(file_path):
        abort(404, description=f"File '{filename}' not found in run '{run_id}'.")
    return send_from_directory(run_dir, filename, mimetype="image/png")


@app.route("/outputs/<run_id>", methods=["GET"])
def list_outputs(run_id: str):
    """List all chart files available for a given run_id."""
    run_dir = os.path.join(OUTPUT_DIR, run_id)
    if not os.path.isdir(run_dir):
        return jsonify({"run_id": run_id, "files": [], "error": "Run not found"}), 404
    files = [f for f in os.listdir(run_dir) if f.endswith(".png")]
    return jsonify({"run_id": run_id, "files": files, "count": len(files)}), 200


@app.route("/", methods=["GET"])
def index():
    """API info page."""
    return jsonify({
        "name"           : "Retail Sales Forecasting API",
        "version"        : "2.1.0",
        "created_by"     : "sanT",
        "pipeline_ready" : _pipeline_ready,
        "endpoints": {
            "GET  /health"                        : "Health check",
            "POST /forecast"                      : "Run ML forecast on CSV",
            "GET  /outputs/<run_id>/<filename>"   : "Serve chart image",
            "GET  /outputs/<run_id>"              : "List charts for a run",
        }
    }), 200


# ─────────────────────────────────────────────────────────────────────────────
# ERROR HANDLERS
# ─────────────────────────────────────────────────────────────────────────────

@app.errorhandler(400)
def handle_bad_request(e):
    return jsonify({
        "success": False,
        "error": "Bad Request",
        "message": str(e.description) if hasattr(e, 'description') else str(e)
    }), 400

@app.errorhandler(413)
def handle_payload_too_large(e):
    return jsonify({
        "success": False,
        "error": "File too large",
        "message": "The uploaded CSV exceeds the 100MB limit."
    }), 413

@app.errorhandler(500)
def handle_internal_server_error(e):
    return jsonify({
        "success": False,
        "error": "Internal Server Error",
        "message": "The pipeline encountered a critical error.",
        "traceback": traceback.format_exc()
    }), 500

@app.errorhandler(Exception)
def handle_exception(e):
    # Pass through HTTP errors
    if hasattr(e, 'code') and isinstance(e.code, int):
        return jsonify({"success": False, "error": str(e)}), e.code
    # Handle non-HTTP exceptions only
    return jsonify({
        "success": False,
        "error": "Unexpected Error",
        "message": str(e),
        "traceback": traceback.format_exc()
    }), 500


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log.info("Starting Flask API in dev mode on port 8000...")
    app.run(host="0.0.0.0", port=8000, debug=True)
