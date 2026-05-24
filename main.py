"""
╔══════════════════════════════════════════════════════════════════╗
║                                                                  ║
║ 🛒  Future Retail Sales Forecasting Using Machine Learning  🛒  ║
║                                                                  ║
║   Created by : sanT                                              ║
║   Version    : 2.0                                               ║
║   Language   : Python 3.10+                                      ║
║                                                                  ║
║   Models Implemented:                                            ║
║     • Naive Baseline          • Linear Regression                ║
║     • Ridge Regression        • Lasso Regression                 ║
║     • ElasticNet              • Bayesian Ridge                   ║
║     • Huber Regressor         • Decision Tree                    ║
║     • KNN Regressor           • SVR (RBF)                        ║
║     • Extra Trees             • Random Forest                    ║
║     • Gradient Boosting       • MLP Neural Network               ║
║     • AR Model (manual)       • Holt-Winters ES                  ║
║     • Weighted Ensemble                                          ║
║                                                                  ║
║   Datasets Supported:                                            ║
║     • pharmacy_daily    (Health & medicine daily sales)          ║
║     • electronics_gadgets (High-value tech gadgets)              ║
║     • grocery_delivery   (App-based delivery data)               ║
║     • furniture_decor    (Seasonal high-ticket furniture)        ║
║     • luxury_brand       (Elite premium brand sales)             ║
║     • custom             (Any user-supplied CSV)                 ║
║                                                                  ║
╚══════════════════════════════════════════════════════════════════╝

Usage
─────
  python main.py                          # Interactive menu
  python main.py --dataset daily_sales    # Quick run specific dataset
  python main.py --dataset all            # Run all datasets
  python main.py --dataset daily_sales --forecast-periods 90 --mode full
"""

import sys
import os
import argparse
import time
import traceback
import warnings
warnings.filterwarnings("ignore")

# ── Path fix so submodules resolve ────────────────────────────────────────────
sys.path.insert(0, os.path.dirname(__file__))

import pandas as pd
import numpy as np

from datasets.generate_datasets import generate_all_datasets
from utils.helpers import (banner, cprint, C, compute_metrics,
                            aggregate_time_series, load_and_validate,
                            build_forecast_index, print_metrics_table,
                            DATASET_SCHEMAS, format_inr, safe_print)
from models.forecasters import (get_all_models, get_quick_models,
                                   EnsembleForecaster, StackingForecaster, get_best_model,
                                   make_gradient_boosting)
from utils.visualise import (plot_time_series, plot_forecast, plot_model_comparison,
                               plot_error_distribution, plot_feature_importance,
                               plot_seasonal_decomposition, plot_category_heatmap,
                               plot_forecast_table, plot_period_revenue,
                               plot_correlation_matrix, plot_rolling_accuracy,
                               plot_acf_pacf)
from reports.report_gen import (save_metrics_csv, save_forecast_csv,
                                  generate_html_report)

# ─── Constants ────────────────────────────────────────────────────────────────
BASE_DIR     = os.path.dirname(__file__)
DATASET_DIR  = os.path.join(BASE_DIR, "datasets")
OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs")
REPORT_DIR   = os.path.join(BASE_DIR, "reports")
MODEL_DIR    = os.path.join(BASE_DIR, "models", "saved")


# ─────────────────────────────────────────────────────────────────────────────
# CORE PIPELINE
# ─────────────────────────────────────────────────────────────────────────────
def run_pipeline(dataset_name: str,
                 forecast_periods: int = 30,
                 mode: str = "quick",
                 custom_csv: str = None,
                 custom_schema: dict = None,
                 save_models: bool = True,
                 auto_optimize: bool = True,
                 tune: bool = False,
                 skip_heavy_viz: bool = False) -> dict:
    """
    End-to-end forecasting pipeline:
      1. Load & validate dataset
      2. Aggregate to time series
      3. Exploratory visualisations
      4. Feature engineering + temporal split
      5. Train & evaluate all models
      6. Build weighted ensemble
      7. Future forecast with best model
      8. Save results / report

    Returns dict with keys: series, metrics, best_model, future_preds
    """
    t0 = time.time()
    out_dir = os.path.join(OUTPUT_DIR, dataset_name)
    os.makedirs(out_dir, exist_ok=True)
    os.makedirs(MODEL_DIR, exist_ok=True)

    cprint(f"\n{'─'*60}", C.BOLD)
    cprint(f"  📂  Dataset : {dataset_name.upper()}", C.BOLD)
    cprint(f"  🎯  Mode    : {mode}  |  Horizon: {forecast_periods} periods", C.BOLD)
    cprint(f"{'─'*60}\n", C.BOLD)

    # ── 1. Load ───────────────────────────────────────────────────────────────
    if custom_csv:
        filepath = custom_csv
        schema_key = "custom"
    else:
        filepath = os.path.join(DATASET_DIR, f"{dataset_name}.csv")
        schema_key = dataset_name

    df, schema = load_and_validate(filepath, schema_key, injected_schema=custom_schema)
    cprint(f"  Schema    : {schema['description']}", C.CYAN)

    # ── 2. Aggregate to time series ───────────────────────────────────────────
    cprint("\n  🔄 Aggregating time series...", C.YELLOW)
    series = aggregate_time_series(df, schema)
    if len(series) < 20:
        raise ValueError(f"Too few data points ({len(series)}) after aggregation.")

    cprint(f"     Periods   : {len(series):,}  ({series.index.min().date()} → {series.index.max().date()})", C.CYAN)
    cprint(f"     Total Rev : {format_inr(series.sum())}", C.CYAN)
    cprint(f"     Avg / pd  : {format_inr(series.mean())}", C.CYAN)
    cprint(f"     Peak      : {format_inr(series.max())}", C.CYAN)

    chart_paths = []

    # ── 3. Exploratory visualisations ─────────────────────────────────────────
    cprint("\n  📊 Generating exploratory charts...", C.YELLOW)
    try:
        p = plot_time_series(series, f"{schema['description']} — Time Series", out_dir)
        chart_paths.append(p)
    except Exception as e:
        cprint(f"    [WARN] Time series plot: {e}", C.YELLOW)

    try:
        freq_map = {"D": "ME", "W": "ME", "MS": "YS", "M": "YS"}
        agg_freq = freq_map.get(schema["freq"], "ME")
        p = plot_period_revenue(series, agg_freq, f"Periodic Revenue — {dataset_name}", out_dir)
        chart_paths.append(p)
    except Exception as e:
        cprint(f"    [WARN] Period revenue plot: {e}", C.YELLOW)

    try:
        season_period = {"D": 7, "W": 4, "MS": 12, "M": 12}.get(schema["freq"], 7)
        if len(series) >= season_period * 2:
            p = plot_seasonal_decomposition(series, season_period,
                                             schema["description"], out_dir)
            chart_paths.append(p)
    except Exception as e:
        cprint(f"    [WARN] Decomposition: {e}", C.YELLOW)

    # Category heatmap (if applicable)
    cat_col = next((c for c in ["Category","Store_Type","City","Season","Subcategory"]
                    if c in df.columns), None)
    if cat_col and len(df[cat_col].unique()) <= 20:
        try:
            row_col = cat_col
            col_col = next((c for c in ["City","Month","Season","Store_Type"]
                            if c in df.columns and c != cat_col), None)
            if col_col and df[col_col].nunique() <= 15:
                p = plot_category_heatmap(df, row_col, col_col,
                                           schema["target"],
                                           f"{row_col} × {col_col} Revenue Share",
                                           out_dir)
                chart_paths.append(p)
        except Exception as e:
            cprint(f"    [WARN] Heatmap: {e}", C.YELLOW)

    try:
        p = plot_correlation_matrix(df, schema["description"], out_dir)
        if p:
            chart_paths.append(p)
    except Exception as e:
        cprint(f"    [WARN] Correlation: {e}", C.YELLOW)

    try:
        p = plot_acf_pacf(series, schema["description"], out_dir)
        chart_paths.append(p)
    except Exception as e:
        cprint(f"    [WARN] ACF/PACF Profiling: {e}", C.YELLOW)

    # ── 4. Train / Test split ─────────────────────────────────────────────────
    n      = len(series)
    split  = int(n * 0.80)
    train_s = series.iloc[:split]
    test_s  = series.iloc[split:]
    cprint(f"\n  ✂  Train: {len(train_s):,} | Test: {len(test_s):,}", C.CYAN)

    # ── 5. Train & evaluate all models ────────────────────────────────────────
    cprint(f"\n  🤖 Training models (mode={mode})...\n", C.YELLOW)
    models = get_all_models(series=series) if mode == "full" else get_quick_models(series=series)
    metrics_list = []
    scored_runs = []

    def _train_eval(model, train_data, test_data, tune_model=False):
        mname = model.name
        try:
            t1 = time.time()
            if tune_model and hasattr(model, "tune"):
                model.tune(train_data)
            model.fit(train_data)
            preds = model.predict(len(test_data))[:len(test_data)]
            m = compute_metrics(test_data.values, preds, mname)
            m["Time_s"] = round(time.time() - t1, 2)
            return (model, preds, m, None)
        except Exception as e:
            traceback.print_exc()
            return (model, None, None, str(e))

    # Sequential execution to support tuning and avoid memory overhead
    results_p = [_train_eval(model, train_s, test_s, tune_model=tune) for model in models]

    for model, preds, m, err in results_p:
        mname = model.name
        if err:
            cprint(f"    ✗ {mname:<28} ERROR: {err}", C.RED)
        else:
            metrics_list.append(m)
            scored_runs.append({"model": model, "preds": preds, "metrics": m})
            cprint(f"    ✔ {mname:<28} RMSE: {m['RMSE']:>12,.2f}  R2: {m['R2']:.4f} ({m['Time_s']}s)", C.GREEN)

    if not metrics_list:
        raise RuntimeError("No models were successfully trained. Check the logs for errors.")

    # ── 6. Weighted Ensemble ──────────────────────────────────────────────────
    cprint(f"\n  🔀 Training Weighted Ensemble...", C.YELLOW)
    try:
        core_models = [
            run["model"] for run in scored_runs
            if hasattr(run["model"], "fit") and hasattr(run["model"], "predict")
        ][:6]
        if len(core_models) >= 2:
            ens = EnsembleForecaster(core_models)
            ens.fit(train_s)
            ens_preds = ens.predict(len(test_s))[:len(test_s)]
            em = compute_metrics(test_s.values, ens_preds, "Weighted Ensemble")
            em["Time_s"] = 0
            metrics_list.append(em)
            scored_runs.append({"model": ens, "preds": ens_preds, "metrics": em})
            cprint(f"    ✔ {'Weighted Ensemble':<28} RMSE: {em['RMSE']:>12,.2f}  R2: {em['R2']:.4f}", C.GREEN)
        # ── God Level: Stacking Regressor ──────────────────────────────────────
        if len(core_models) >= 3:
            cprint(f"\n  🧠 Training Stacking Meta-Learner...", C.YELLOW)
            stack = StackingForecaster(core_models[:5])
            stack.fit(train_s)
            s_preds = stack.predict(len(test_s))[:len(test_s)]
            sm = compute_metrics(test_s.values, s_preds, stack.name)
            sm["Time_s"] = 0
            metrics_list.append(sm)
            scored_runs.append({"model": stack, "preds": s_preds, "metrics": sm})
            cprint(f"    ✔ {stack.name:<28} RMSE: {sm['RMSE']:>12,.2f}  R2: {sm['R2']:.4f}", C.GREEN)

    except Exception as e:
        cprint(f"    [WARN] Ensemble/Stacking failed: {e}", C.YELLOW)

    # ── Print comparison table ────────────────────────────────────────────────
    cprint("\n  📋 Model Leaderboard:\n", C.BOLD)
    sorted_metrics = sorted(metrics_list, key=lambda x: float(x.get("RMSE",9e9)))
    print_metrics_table(sorted_metrics)

    # Save comparison chart
    try:
        p = plot_model_comparison(metrics_list, out_dir)
        chart_paths.append(p)
    except Exception as e:
        cprint(f"    [WARN] Comparison chart: {e}", C.YELLOW)

    # ── 7. Best model + detailed analysis ────────────────────────────────────
    best_run = min(scored_runs, key=lambda run: float(run["metrics"].get("RMSE", 9e9)))
    best_metrics = best_run["metrics"]
    best_model = best_run["model"]
    best_preds = best_run["preds"]

    cprint(f"\n  🏆 Best Model: {best_model.name}", C.GREEN)

    # Re-fit best model on FULL series for forecasting
    best_model.fit(series)

    # Error distribution
    try:
        p = plot_error_distribution(test_s.values, best_preds[:len(test_s)],
                                     best_model.name, out_dir)
        chart_paths.append(p)
    except Exception as e:
        cprint(f"    [WARN] Error dist: {e}", C.YELLOW)

    # Feature importance
    if hasattr(best_model, "feature_importance"):
        try:
            imp = best_model.feature_importance()
            if imp is not None:
                p = plot_feature_importance(imp, best_model.name, out_dir)
                chart_paths.append(p)
        except Exception as e:
            cprint(f"    [WARN] Feature importance: {e}", C.YELLOW)

    # Rolling accuracy (Heavy Calculation - skipped in API/Auto mode)
    if len(series) > 60 and not skip_heavy_viz:
        try:
            test_fc = make_gradient_boosting()
            p = plot_rolling_accuracy(series, test_fc, window=min(30, len(series)//4), out_dir=out_dir)
            chart_paths.append(p)
        except Exception as e:
            cprint(f"    [WARN] Rolling accuracy: {e}", C.YELLOW)

    # ── 8. Future Forecast ────────────────────────────────────────────────────
    cprint(f"\n  🔮 Generating {forecast_periods}-period future forecast...", C.YELLOW)
    future_dates = build_forecast_index(series.index[-1], forecast_periods, schema["freq"])
    future_preds = best_model.predict(forecast_periods)
    future_preds = np.maximum(future_preds, 0)

    cprint(f"\n  📈 Forecast Summary ({best_model.name}):", C.CYAN)
    cprint(f"     From: {future_dates[0].date()}  To: {future_dates[-1].date()}", C.CYAN)
    cprint(f"     Total Forecast : {format_inr(future_preds.sum())}", C.CYAN)
    cprint(f"     Avg / Period   : {format_inr(future_preds.mean())}", C.CYAN)
    cprint(f"     Peak Period    : {format_inr(future_preds.max())}", C.CYAN)

    # Forecast chart
    try:
        p = plot_forecast(train_s, test_s, best_preds, future_dates, future_preds,
                          best_model.name, out_dir)
        chart_paths.insert(1, p)  # 2nd chart in report
    except Exception as e:
        cprint(f"    [WARN] Forecast plot: {e}", C.YELLOW)

    # Forecast table image
    try:
        p = plot_forecast_table(future_dates, future_preds, best_model.name, out_dir)
        chart_paths.append(p)
    except Exception as e:
        cprint(f"    [WARN] Forecast table: {e}", C.YELLOW)

    # ── 9. Save artefacts ────────────────────────────────────────────────────
    cprint(f"\n  💾 Saving artefacts...", C.YELLOW)
    save_metrics_csv(metrics_list, out_dir)
    save_forecast_csv(future_dates, future_preds, best_model.name, out_dir)

    if save_models and hasattr(best_model, "save"):
        safe = best_model.name.replace(" ","_").replace("(","").replace(")","").replace("=","")
        best_model.save(os.path.join(MODEL_DIR, f"{dataset_name}_{safe}.pkl"))

    report_path = generate_html_report(
        dataset_name, schema, series, metrics_list,
        best_model.name, future_dates, future_preds,
        [p for p in chart_paths if p],
        out_dir
    )

    # ── 10. Adaptive Frequency Recommendation ─────────────────────────────────
    recommendation = None
    if best_metrics.get("R2", 0) < 0.1:
        cur_freq = schema.get("freq", "D")
        if cur_freq == "D":
            if auto_optimize:
                cprint("\n  💡 Optimizer Suggestion: Automatically switching to 'Weekly' (W) aggregation...", C.YELLOW)
                schema["freq"] = "W"
                # Recursively rerun the pipeline with Weekly aggregation
                result = run_pipeline(dataset_name, forecast_periods, mode, custom_csv, schema, save_models, False, tune, skip_heavy_viz)
                base_msg = "Auto-applied Optimizer Suggestion: Switched to 'Weekly' (W) aggregation to reduce Daily noise and improve R²."
                if result.get("recommendation"):
                    result["recommendation"] = f"{base_msg} | Further Suggestion: {result['recommendation']}"
                else:
                    result["recommendation"] = base_msg
                return result
            else:
                recommendation = "Switch to 'Weekly' (W) aggregation to reduce Daily noise and improve R²."
        elif cur_freq == "W":
            recommendation = "Switch to 'Monthly' (MS) aggregation to find stronger patterns and improve R²."
        elif cur_freq in ["MS", "M"]:
             recommendation = "Consider adding more historical features or checking for data consistency/outliers."
    
    if best_metrics.get("R2", 0) < 0.7 and not tune:
        tune_msg = "Enable Hyperparameter Tuning (--tune) to squeeze out more performance and reach >0.8 R²."
        recommendation = f"{recommendation} | {tune_msg}" if recommendation else tune_msg

    if recommendation:
        cprint(f"\n  💡 SUGGESTION: {recommendation}", C.YELLOW)

    elapsed = time.time() - t0
    cprint(f"\n  ✅ Pipeline complete in {elapsed:.1f}s", C.GREEN)
    cprint(f"     Outputs  → {out_dir}", C.GREEN)
    cprint(f"     Report   → {report_path}", C.GREEN)

    return {
        "dataset":       dataset_name,
        "series":        series,
        "metrics":       metrics_list,
        "best_model":    best_model,
        "future_dates":  future_dates,
        "future_preds":  future_preds,
        "report":        report_path,
        "out_dir":       out_dir,
        "recommendation": recommendation,
    }


# ─────────────────────────────────────────────────────────────────────────────
# INTERACTIVE MENU
# ─────────────────────────────────────────────────────────────────────────────
def interactive_menu():
    banner()
    safe_print(
        f"""
{C.BOLD}Available Datasets:{C.END}
  [1] Pharmacy Daily Sales (Healthcare)
  [2] Electronics Gadgets (High-Value)
  [3] Grocery App Delivery (Hyper-local)
  [4] Furniture & Home Decor (Seasonal)
  [5] Luxury Brand (Elite High-Margin)
  [6] Run ALL datasets
  [7] Run CUSTOM CSV
  [0] Exit
"""
    )

    ds_map = {
        "1": "pharmacy_daily", "2": "electronics_gadgets", "3": "grocery_delivery",
        "4": "furniture_decor", "5": "luxury_brand"
    }

    choice = input(f"{C.BOLD}Select dataset [1-7 or 0]: {C.END}").strip()

    if choice == "0":
        cprint("Goodbye! 👋", C.CYAN)
        return

    elif choice == "6":
        mode    = input("Mode - [q]uick or [f]ull (default: q): ").strip().lower() or "q"
        tune    = input("Enable Hyperparameter Tuning? (y/n, default: n): ").strip().lower() == "y"
        periods = input("Forecast periods (default: 30): ").strip() or "30"
        run_all_datasets(mode="full" if mode == "f" else "quick",
                         forecast_periods=int(periods), tune=tune)
        return

    elif choice == "7":
        path = input("Enter CSV file path: ").strip()
        if not os.path.exists(path):
            cprint(f"File not found: {path}", C.RED)
            return
        name    = input("Dataset name (for output folder): ").strip() or "custom"
        periods = input("Forecast periods (default: 30): ").strip() or "30"
        mode    = input("Mode - [q]uick or [f]ull (default: q): ").strip().lower() or "q"
        tune    = input("Enable Hyperparameter Tuning? (y/n, default: n): ").strip().lower() == "y"
        run_pipeline(name, int(periods), mode="full" if mode=="f" else "quick",
                     custom_csv=path, tune=tune)
        return

    elif choice not in ds_map:
        cprint("Invalid choice.", C.RED)
        return

    dataset_name = ds_map[choice]
    mode    = input("Mode - [q]uick (fast) or [f]ull (all 16+ models) (default: q): ").strip().lower() or "q"
    tune    = input("Enable Hyperparameter Tuning? (y/n, default: n): ").strip().lower() == "y"
    periods = input("Forecast periods ahead (default: 30): ").strip() or "30"

    try:
        run_pipeline(dataset_name, int(periods),
                     mode="full" if mode == "f" else "quick", tune=tune)
    except Exception as e:
        cprint(f"\n❌ Pipeline error: {e}", C.RED)
        traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# RUN ALL DATASETS
# ─────────────────────────────────────────────────────────────────────────────
def run_all_datasets(mode: str = "quick", forecast_periods: int = 30, tune: bool = False):
    banner()
    all_datasets = list(DATASET_SCHEMAS.keys())
    results = {}

    cprint(f"\n🚀 Running ALL datasets ({len(all_datasets)} total)...\n", C.BOLD)

    for ds in all_datasets:
        cprint(f"\n{'═'*60}", C.BOLD)
        cprint(f"  Dataset: {ds}", C.BOLD)
        cprint(f"{'═'*60}", C.BOLD)
        try:
            r = run_pipeline(ds, forecast_periods, mode=mode, tune=tune)
            results[ds] = r
        except Exception as e:
            cprint(f"\n❌ '{ds}' failed: {e}", C.RED)
            traceback.print_exc()

    # Cross-dataset summary
    cprint(f"\n\n{'═'*60}", C.BOLD)
    cprint("  📊 CROSS-DATASET SUMMARY", C.BOLD)
    cprint(f"{'═'*60}", C.BOLD)
    for ds, r in results.items():
        best = min(r["metrics"], key=lambda x: float(x.get("RMSE",9e9)))
        cprint(f"  {ds:<22} Best: {best['Model']:<25} RMSE: {best['RMSE']:>12,.0f}", C.GREEN)

    return results


# ─────────────────────────────────────────────────────────────────────────────
# STARTUP: ENSURE DATASETS EXIST
# ─────────────────────────────────────────────────────────────────────────────
def ensure_datasets():
    """Generate datasets if any are missing."""
    names = list(DATASET_SCHEMAS.keys())
    missing = [n for n in names
               if not os.path.exists(os.path.join(DATASET_DIR, f"{n}.csv"))]
    if missing:
        cprint(f"\n  ⚡ Generating missing datasets: {missing}", C.YELLOW)
        generate_all_datasets(DATASET_DIR)
    else:
        cprint("  ✔ All datasets present.", C.GREEN)


# ─────────────────────────────────────────────────────────────────────────────
# CLI ENTRY POINT
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    parser = argparse.ArgumentParser(
        description="Future Retail Sales Forecasting - sanT",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Run `python main.py` for interactive mode or pass --dataset for batch execution."
    )
    parser.add_argument("--dataset",    default=None,
                        help="Dataset name or 'all' (default: interactive)")
    parser.add_argument("--forecast-periods", type=int, default=30,
                        help="Number of future periods to forecast (default: 30)")
    parser.add_argument("--mode",       default="quick",
                        choices=["quick","full"],
                        help="Model set: quick (8 models) or full (16+ models)")
    parser.add_argument("--custom-csv", default=None,
                        help="Path to custom CSV file")
    parser.add_argument("--tune",       action="store_true",
                        help="Enable automated hyperparameter tuning")
    parser.add_argument("--generate-only", action="store_true",
                        help="Only generate datasets and exit")
    return parser.parse_args()


def main():
    banner()
    args = parse_args()

    # Ensure output dirs exist
    for d in [OUTPUT_DIR, REPORT_DIR, MODEL_DIR]:
        os.makedirs(d, exist_ok=True)

    # Generate datasets
    ensure_datasets()

    if args.generate_only:
        cprint("\n✅ Datasets generated. Exiting.", C.GREEN)
        return

    if args.custom_csv:
        name = os.path.splitext(os.path.basename(args.custom_csv))[0]
        run_pipeline(name, args.forecast_periods, args.mode, args.custom_csv)

    elif args.dataset == "all":
        run_all_datasets(args.mode, args.forecast_periods)

    elif args.dataset:
        if args.dataset not in DATASET_SCHEMAS:
            cprint(f"Unknown dataset '{args.dataset}'. Choices: {list(DATASET_SCHEMAS.keys())}", C.RED)
            sys.exit(1)
        run_pipeline(args.dataset, args.forecast_periods, args.mode, tune=args.tune)

    else:
        # Interactive mode
        interactive_menu()


if __name__ == "__main__":
    main()
