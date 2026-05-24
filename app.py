import os
import tempfile
import time
import traceback
import pandas as pd
import numpy as np
import streamlit as st
import streamlit.components.v1 as components

# Import the core pipeline from main.py
from main import run_pipeline, run_all_datasets, DATASET_SCHEMAS
from utils.insights import get_strategic_recommendations
from utils.helpers import check_stationarity

# ── Pandas Compatibility Fix (LargeUtf8 / PyArrow Error) ──────────────────────
import pandas as pd
try:
    pd.options.future.infer_string = False
except Exception:  # More specific error handling
    pass

# ── Streamlit Page Config ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="Future Retail Sales Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded",
)

def display_pipeline_results(results):
    import numpy as np
    
    # Show high-level best model
    best_model = results.get("best_model")
    best_model_name = getattr(best_model, "name", str(best_model))
    st.subheader(f"🏆 Best Auto-Selected Model: **{best_model_name}**")
    
    # 💡 Smart Frequency Recommendation
    recommendation = results.get("recommendation")
    if recommendation:
        st.warning(f"**💡 Optimizer Suggestion:** {recommendation}")
    
    # Show Metrics Leaderboard
    metrics_list = results.get("metrics", [])
    if metrics_list:
        st.markdown("### 📋 Model Leaderboard")
        df_metrics = pd.DataFrame(metrics_list)
        
        # Rename R2 for display
        if "R2" in df_metrics.columns:
            df_metrics = df_metrics.rename(columns={"R2": "R²"})
            
        # Sort by R² (Primary metric for accuracy, higher is better)
        if "R²" in df_metrics.columns:
            df_metrics = df_metrics.sort_values("R²", ascending=False)
        else:
            df_metrics = df_metrics.sort_values("RMSE", ascending=True)
            
        # Reset index to provide a clean "series of order" starting from 1
        df_metrics = df_metrics.reset_index(drop=True)
        df_metrics.index = df_metrics.index + 1
        
        # Insert Rank column as first column
        df_metrics.insert(0, "Rank", range(1, len(df_metrics) + 1))
        
        # Display the leaderboard
        st.dataframe(df_metrics.astype(object), use_container_width=True, hide_index=True)
        
    # Show Native Predictions
    future_dates = results.get("future_dates")
    future_preds = results.get("future_preds")
    series = results.get("series")
    
    if future_dates is not None and future_preds is not None:
        st.markdown(f"### 📈 Predicted Information (Next {len(future_preds)} Periods)")
        
        # Calculate Future vs Historical Stats
        total_future = np.sum(future_preds)
        avg_future = np.mean(future_preds)
        
        if series is not None and len(series) > 0:
            avg_hist = np.mean(series)
            pct_change = ((avg_future - avg_hist) / avg_hist) * 100
            
            # Formatting values for display
            def _format_curr(v):
                if v >= 1e7: return f"₹{v/1e7:.2f} Cr"
                if v >= 1e5: return f"₹{v/1e5:.2f} L"
                return f"₹{v:,.0f}"

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Total Future Forecast", _format_curr(total_future))
            with col2:
                st.metric("Avg. Predicted / Period", _format_curr(avg_future))
            with col3:
                st.metric("Market Trend", f"{pct_change:+.1f}%", 
                          help="Comparison between the Predicted Average vs Historical Average.")

            st.info(f"**🔍 Analysis:** Based on the **{best_model_name}** model, the future sales are expected to be **{abs(pct_change):.1f}% {'higher' if pct_change >= 0 else 'lower'}** compared to your historical average. The total forecasted revenue for the next {len(future_preds)} periods is **{_format_curr(total_future)}**.")
        
        df_future = pd.DataFrame({
            "Date": future_dates,
            "Predicted Sales": np.round(future_preds, 2)
        }).set_index("Date")
        
        if series is not None:
            df_hist = pd.DataFrame({"Historical Sales": series})
            df_hist.index.name = "Date"
            df_fut_plot = pd.DataFrame({f"Forecast ({best_model_name})": future_preds}, index=future_dates)
            df_fut_plot.index.name = "Date"
            df_combined = pd.concat([df_hist, df_fut_plot], axis=1)
            st.line_chart(df_combined, use_container_width=True)
        else:
            st.line_chart(df_future, use_container_width=True)

        with st.expander("👀 View Predicted Raw Data", expanded=False):
            st.dataframe(df_future.astype(object), use_container_width=True, hide_index=False) # Keep index for dates

    # ── 🔍 Forensic Diagnostics ──────────────────────────────────────────────
    st.subheader("🎯 Forensic Data Profiling")
    # Forensic analysis data
    from utils.helpers import detect_optimal_lags
    
    if len(results["series"]) < 7:
        st.warning("⚠️ **Insufficient Historical Data**: At least 7 data points are required for statistical profiling (ADF/ACF).")
        diag = {"is_stationary": True, "p_value": 0.0, "lags_used": 0}
        opt_lags = 0
    else:
        diag = check_stationarity(results["series"])
        opt_lags = detect_optimal_lags(results["series"])
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        status = "✅ Stationary" if diag["is_stationary"] else "⚠️ Trend Detected"
        st.metric("Mean Reversion", status)
    with col2:
        st.metric("p-Value (ADF)", f"{diag['p_value']:.4f}")
    with col3:
        st.metric("Statistical Memory", f"{opt_lags} Periods")
    with col4:
        st.metric("Seasonality Lags", diag["lags_used"])
    
    if not diag["is_stationary"]:
        st.info("💡 **Insight**: Data shows a non-stationary trend. The 'Stacking Regressor' is recommended to handle this drift.")
    else:
        st.success("💡 **Insight**: Data is stationary. Standard linear and ensemble models will be highly effective.")
    
    # ── 🚀 Strategic Growth Recommendations ─────────────────────────────────────
    recommendations = get_strategic_recommendations(results)
    if recommendations:
        st.markdown("---")
        st.subheader("🚀 Strategic Growth Recommendations")
        st.info("Directly actionable steps to maximize your future revenue based on this forecast.")
        
        cols_per_row = 3
        for i in range(0, len(recommendations), cols_per_row):
            cols = st.columns(cols_per_row)
            for j in range(cols_per_row):
                idx = i + j
                if idx < len(recommendations):
                    rec = recommendations[idx]
                    color = rec.get("color", "#ff4b4b")
                    icon = rec.get("icon", "")
                    category = rec.get("category", "")
                    title = rec.get("title", "")
                    description = rec.get("description", "")
                    tooltip_raw = rec.get("tooltip", "")
                    tooltip = tooltip_raw.replace("\n", " · ") if isinstance(tooltip_raw, str) else ""
                    
                    with cols[j]:
                        st.markdown(f"""
                        <div style="background:linear-gradient(135deg, #ffffff 0%, #f8f9fa 100%);padding:20px;border-radius:12px;margin-bottom:15px;box-shadow:0 2px 8px rgba(0,0,0,0.08);border-left:5px solid {color};transition:all 0.3s ease;cursor:pointer;" onmouseover="this.style.transform='translateY(-3px)';this.style.boxShadow='0 4px 15px rgba(0,0,0,0.15)'" onmouseout="this.style.transform='translateY(0)';this.style.boxShadow='0 2px 8px rgba(0,0,0,0.08)'">
                            <div style="display:flex;align-items:center;margin-bottom:12px;">
                                <span style="font-size:24px;margin-right:10px;">{icon}</span>
                                <span style="background-color:{color}20;color:{color};padding:4px 12px;border-radius:20px;font-size:12px;font-weight:600;">{category}</span>
                            </div>
                            <h4 style="margin:0 0 8px 0;color:#1a1a2e;font-size:15px;font-weight:700;line-height:1.3;">{title}</h4>
                            <p style="margin:0;color:#6b7280;font-size:13px;line-height:1.5;">{description}</p>
                            <div style="margin-top:12px;padding-top:12px;border-top:1px solid #e5e7eb;">
                                <span style="color:{color};font-size:12px;font-weight:600;" title="{tooltip}">→ {tooltip}</span>
                            </div>
                        </div>
                        """, unsafe_allow_html=True)
            
    # Embed HTML Report directly into UI
    report_path = results.get("report")
    if report_path and os.path.exists(report_path):
        st.markdown("---")
        st.subheader("📊 Full Interactive Dashboard Report")
        with open(report_path, "r", encoding="utf-8") as f:
            components.html(f.read(), height=1200, scrolling=True)
    else:
        st.warning("No HTML report file found to display.")

def display_all_results_summary(all_results):
    st.markdown("---")
    st.header("📊 Cross-Dataset Benchmark Summary")
    st.info("Comprehensive benchmark across all available retail datasets.")
    
    summary_data = []
    for ds_name, res in all_results.items():
        best_model = res.get("best_model")
        best_name = getattr(best_model, "name", str(best_model))
        metrics = res.get("metrics", [])
        # Find best metrics correctly
        best_metrics = min(metrics, key=lambda x: float(x.get("RMSE", 9e12))) if metrics else {}
        
        summary_data.append({
            "Dataset": ds_name.upper(),
            "🏆 Best Model": best_name,
            "RMSE": f"{best_metrics.get('RMSE', 0):,.2f}",
            "R² Score": f"{best_metrics.get('R2', 0):.4f}",
            "Avg Pred": np.mean(res.get("future_preds", [0]))
        })
    
    df_summary = pd.DataFrame(summary_data)
    
    # Sort by R² Score (convert back to numeric for sorting)
    if "R² Score" in df_summary.columns:
        df_summary["R2_val"] = pd.to_numeric(df_summary["R² Score"], errors="coerce")
        df_summary = df_summary.sort_values("R2_val", ascending=False).drop(columns=["R2_val"])
    
    # Reset index for sequential order
    df_summary = df_summary.reset_index(drop=True)
    df_summary.index = df_summary.index + 1
    
    st.dataframe(df_summary.astype(object), use_container_width=True, hide_index=True)
    
    # Detailed Tabs
    st.markdown("---")
    st.subheader("📂 Detailed Reports by Dataset")
    ds_names = list(all_results.keys())
    tabs = st.tabs([d.upper() for d in ds_names])
    
    for i, tab in enumerate(tabs):
        ds_name = ds_names[i]
        with tab:
            st.markdown(f"### Results for **{ds_name.upper()}**")
            display_pipeline_results(all_results[ds_name])


# ── Main UI ───────────────────────────────────────────────────────────────────
st.title("🛒 Future Retail Sales Forecasting Web App")
st.markdown("### Created by: **sanT**")
st.markdown("Upload your custom CSV dataset to run the automated ML forecasting pipeline with 22 different machine learning models.")

with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Mode selection
    st.subheader("Model Selection")
    mode = st.radio(
        "Run Mode",
        options=["quick", "full"],
        index=0,
        help="'quick' runs 8 baseline models. 'full' runs all 22+ models (slower)."
    )
    
    # Forecast horizon
    st.subheader("Forecast Horizon")
    forecast_periods = st.number_input(
        "Periods to forecast",
        min_value=1,
        max_value=365,
        value=30,
        help="Number of future time steps to predict."
    )
    st.sidebar.markdown("---")
    st.sidebar.markdown("Created by: **sanT**")
    
st.header("📤 Upload Custom Dataset")
st.markdown("""
Please ensure your CSV file has a date column and a target (sales/revenue) column. 
For optimal results, ensure the dataset is properly aggregated or that it can be aggregated by the system.
""")

# Initialize frequency variables (needed across all code paths)
freq_map = {"D": "Daily", "W": "Weekly", "MS": "Monthly", "QS": "Quarterly", "YS": "Yearly"}
freq_tooltip = {
    "D": "Daily data (1 day intervals)",
    "W": "Weekly data (7 day intervals)", 
    "MS": "Monthly data (month start intervals)",
    "QS": "Quarterly data (3 month intervals)",
    "YS": "Yearly data (12 month intervals)"
}
selected_freq = "D"
selected_freq_display = "Daily"
custom_schema = None
freq_for_schema = "D"

uploaded_file = st.file_uploader("Choose a CSV dataset", type=["csv"])

if uploaded_file is not None:
    st.success(f"File '{uploaded_file.name}' uploaded successfully!")
    
    # Show a quick preview of the dataset
    try:
        # Handle encoding issues
        encodings = ["utf-8", "utf-8-sig", "latin1", "ISO-8859-1", "cp1252"]
        df_preview = None
        for enc in encodings:
            try:
                uploaded_file.seek(0)
                df_preview = pd.read_csv(uploaded_file, encoding=enc)
                break
            except (UnicodeDecodeError, UnicodeError):
                continue
        
        if df_preview is None:
            raise ValueError("Unable to read CSV with any standard encoding.")
        
        with st.expander("🔍 Preview Uploaded Data"):
            st.dataframe(df_preview.head(10).astype(object), hide_index=True)
            st.write(f"**Shape:** {df_preview.shape}")
        
        # ── 🧩 Robust Auto-Detect Column Mapping ─────────────────────────────────
        cols = df_preview.columns.tolist()
        num_cols = df_preview.select_dtypes(include=[np.number]).columns.tolist()
        
        # ── 1. DATE COLUMN DETECTION ───────────────────────────────────────────
        DATE_KEYWORDS = [
            "date", "time", "timestamp", "period", "datetime", "day", "month", "year",
            "created", "updated", "order_date", "sale_date", "transaction_date",
            "delivery_date", "invoice_date", "billing_date", "ship_date",
            "week", "week_start", "week_end", "reporting_date", "post_date",
            "start_date", "end_date", "publish_date", "event_date", "visit_date",
            "closing_date", "opening_date", "modified", "added", "closed"
        ]
        
        def score_date_column(col_name):
            """Score a column for being a date column - comprehensive scoring."""
            cn = col_name.lower()
            
            # Name-based scoring with priorities
            name_score = 0
            primary_keywords = ["date", "datetime", "timestamp", "time"]
            secondary_keywords = ["period", "day", "month", "year", "week"]
            tertiary_keywords = ["created", "updated", "modified", "added", "closed"]
            
            for kw in primary_keywords:
                if kw in cn:
                    name_score += 50
            for kw in secondary_keywords:
                if kw in cn:
                    name_score += 25
            for kw in tertiary_keywords:
                if kw in cn:
                    name_score += 15
            
            # Bonus for suffix patterns
            if cn.endswith("_date") or cn.endswith("_datetime") or cn.endswith("_timestamp"):
                name_score += 30
            if cn.endswith("_time") or cn.endswith("_at"):
                name_score += 25
            if cn in ["dt", "ts", "d", "t"]:
                name_score += 10
            
            # Data validation
            data_score = 0
            try:
                sample = df_preview[col_name].dropna().head(200)
                if len(sample) < 5:
                    return name_score if name_score > 30 else 0
                
                parsed = pd.to_datetime(sample, errors="coerce")
                valid_count = parsed.notna().sum()
                valid_ratio = valid_count / len(sample)
                
                # Must have at least 80% valid dates
                if valid_ratio < 0.8:
                    return name_score
                
                # Date range validation
                if valid_count > 0:
                    valid_dates = parsed.dropna()
                    min_date = valid_dates.min()
                    max_date = valid_dates.max()
                    if pd.notna(min_date) and pd.notna(max_date):
                        if min_date.year < 1950 or max_date.year > 2100:
                            return name_score
                        data_score += 15  # Valid date range
                
                # Prefer columns with unique dates (not repeated values)
                unique_ratio = valid_dates.nunique() / len(valid_dates)
                if unique_ratio > 0.5:
                    data_score += 10
                
            except Exception:
                return name_score if name_score > 25 else 0
            
            return name_score + data_score
        
        def detect_date_column(columns):
            """Detect the best date column by comparing ALL columns."""
            candidates = []
            for col in columns:
                score = score_date_column(col)
                if score > 0:
                    candidates.append((col, score))
            
            if not candidates:
                # Fallback: try any column that parses as dates
                for col in columns:
                    try:
                        sample = df_preview[col].dropna().head(50)
                        if len(sample) > 10:
                            parsed = pd.to_datetime(sample, errors="coerce")
                            if parsed.notna().sum() / len(sample) > 0.9:
                                return col
                    except:
                        pass
                return columns[0] if columns else None
            
            # Sort by score
            candidates.sort(key=lambda x: x[1], reverse=True)
            
            # If multiple candidates, compare data quality
            if len(candidates) > 1:
                best_col = candidates[0][0]
                best_score = candidates[0][1]
                second_col = candidates[1][0]
                second_score = candidates[1][1]
                
                # If scores are close, compare actual date parsing quality
                if best_score - second_score < 20:
                    best_parsed = pd.to_datetime(df_preview[best_col].dropna(), errors="coerce").dropna()
                    second_parsed = pd.to_datetime(df_preview[second_col].dropna(), errors="coerce").dropna()
                    
                    # Prefer more valid dates
                    if len(second_parsed) > len(best_parsed) * 1.2:
                        best_col = second_col
                    # Prefer better date range
                    elif len(best_parsed) > 0 and len(second_parsed) > 0:
                        if second_parsed.max().year - second_parsed.min().year > best_parsed.max().year - best_parsed.min().year:
                            best_col = second_col
                return best_col
            
            return candidates[0][0]
        
        # ── 2. TARGET COLUMN DETECTION ─────────────────────────────────────────
        TARGET_KEYWORDS = [
            "revenue", "sales", "sale", "amount", "total", "price", "value", "income",
            "profit", "quantity", "qty", "units", "count", "order", "orders",
            "gross", "net", "turnover", "earning", "rev", "amt", "cost", "expense",
            "billing", "transaction", "volume", "gain", "loss", "sum", "paid",
            "received", "earned", "generated"
        ]
        
        def score_target_column(col_name):
            """Score a column for being a suitable target - comprehensive scoring."""
            cn = col_name.lower()
            
            # Name-based scoring with priorities
            name_score = 0
            primary_keywords = ["revenue", "sales", "sale", "amount", "total","profit", "income"]
            secondary_keywords = ["price", "value", "turnover", "earning"]
            tertiary_keywords = ["quantity", "qty", "units", "count", "order", "billing"]
            
            for kw in primary_keywords:
                if kw in cn:
                    name_score += 50
            for kw in secondary_keywords:
                if kw in cn:
                    name_score += 30
            for kw in tertiary_keywords:
                if kw in cn:
                    name_score += 15
            for kw in TARGET_KEYWORDS:
                if kw not in primary_keywords + secondary_keywords + tertiary_keywords:
                    name_score += 10
            
            # Data validation
            data_score = 0
            try:
                col_data = pd.to_numeric(df_preview[col_name], errors="coerce").dropna()
                if len(col_data) < 5:
                    return name_score if name_score > 30 else 0
                
                # Must be numeric
                if col_data.dtype not in ['int64', 'float64', 'int32', 'float32']:
                    return name_score if name_score > 0 else 0
                
                # Variance check
                if col_data.nunique() < 2:
                    return name_score if name_score > 25 else 0
                
                # Value characteristics
                if col_data.min() >= 0:
                    data_score += 15
                if col_data.mean() > 0:
                    data_score += 10
                if col_data.std() > 0:
                    data_score += 10
                
                # Range check
                max_val = col_data.max()
                if max_val > 1e12:
                    data_score -= 20
                elif max_val > 1e9:
                    data_score -= 10
                
                # Unique ratio for time-series
                unique_ratio = col_data.nunique() / len(col_data)
                if 0.1 < unique_ratio < 1.0:
                    data_score += 10
                
            except Exception:
                return name_score if name_score > 20 else 0
            
            return name_score + data_score
        
        def detect_target_column(numeric_cols):
            """Detect the best target - comprehensive scoring without skipping relevant columns."""
            if not numeric_cols:
                return None
            
            FIRST_PRIORITY_KEYWORDS = ["revenue", "final_amount", "sales"]
            
            cn_lower = {col: col.lower().replace(" ", "_").replace("-", "_") for col in numeric_cols}
            
            for col in numeric_cols:
                cn = cn_lower[col]
                for kw in FIRST_PRIORITY_KEYWORDS:
                    if kw in cn:
                        try:
                            col_data = pd.to_numeric(df_preview[col], errors="coerce")
                            valid_data = col_data.dropna()
                            if len(valid_data) >= 3 and valid_data.nunique() >= 2:
                                return col
                        except Exception:
                            continue
            
            PRIORITY_KEYWORDS = {
                "exact": {
                    1000: ["total_revenue", "gross_revenue", "net_revenue", "sales_revenue",
                           "total_sales", "gross_sales", "net_sales", "sale", "sales_amount",
                           "total_income", "gross_income", "net_income", "income", "business_income",
                           "total_profit", "gross_profit", "net_profit", "profit", "operating_profit",
                           "final_amount", "gross_margin", "turnover", "revenue_total", "sales_total",
                           "order_value", "order_amount", "transaction_value", "transaction_amount",
                           "sales_value", "sales_volume", "sales_income", "monthly_sales", "daily_sales",
                           "annual_revenue", "monthly_revenue", "annual_sales", "weekly_revenue", "weekly_sales"],
                    500: ["price", "value", "amount", "invoice_amount", "billing_amount", "payment_amount",
                          "earnings", "operating_income", "total_earnings", "purchase_price", "selling_price",
                          "unit_price", "product_price", "avg_price", "average_price", "contract_value",
                          "due_amount", "amount_paid", "amount_received", "total_amount", "gross_amount",
                          "net_amount", "profit_margin"],
                    200: ["transaction", "order", "billing", "margin", "growth", "volume",
                          "retail_sales", "online_sales", "wholesale_sales", "product_sales",
                          "revenue_growth", "sales_growth", "profit_growth", "gains", "proceeds",
                          "quantity", "qty", "units", "count", "paid", "received", "earned"]
                },
                "partial": {
                    300: ["revenue", "sales", "profit", "income"],
                    150: ["amount", "value", "price", "earning", "turnover", "margin"],
                    75: ["transaction", "order", "billing", "volume", "gains"]
                }
            }
            
            scored_cols = []
            
            for col in numeric_cols:
                try:
                    col_data = pd.to_numeric(df_preview[col], errors="coerce")
                    valid_data = col_data.dropna()
                    
                    if len(valid_data) < 3:
                        continue
                    if valid_data.nunique() < 2:
                        continue
                    
                    cn = col.lower().replace(" ", "_").replace("-", "_")
                    score = 0
                    match_type = None
                    
                    for priority, keywords in PRIORITY_KEYWORDS["exact"].items():
                        if cn in keywords:
                            score += priority
                            match_type = "exact"
                            break
                    
                    if match_type != "exact":
                        best_partial = 0
                        for priority, keywords in PRIORITY_KEYWORDS["partial"].items():
                            for kw in keywords:
                                if kw in cn:
                                    best_partial = max(best_partial, priority)
                                    break
                        if best_partial > 0:
                            score += best_partial
                    
                    if score > 0:
                        positive_ratio = (valid_data > 0).sum() / len(valid_data)
                        if positive_ratio >= 0.90:
                            score += 40
                        elif positive_ratio >= 0.70:
                            score += 20
                        
                        if valid_data.mean() > 0:
                            score += 15
                        
                        if valid_data.std() > 0:
                            cv = valid_data.std() / abs(valid_data.mean()) if valid_data.mean() != 0 else 1
                            if 0.01 < cv < 5.0:
                                score += 10
                        
                        null_ratio = valid_data.isna().sum() / len(valid_data)
                        if null_ratio < 0.1:
                            score += 10
                        
                        scored_cols.append((col, score))
                        
                except Exception:
                    continue
            
            if scored_cols:
                scored_cols.sort(key=lambda x: x[1], reverse=True)
                return scored_cols[0][0]
            
            for col in numeric_cols:
                try:
                    col_data = pd.to_numeric(df_preview[col], errors="coerce")
                    valid_data = col_data.dropna()
                    if len(valid_data) >= 3 and valid_data.nunique() >= 2:
                        return col
                except Exception:
                    continue
            
            return numeric_cols[0] if numeric_cols else None
        
        # ── 3. FREQUENCY DETECTION ─────────────────────────────────────────────
        freq_map = {"D": "Daily", "W": "Weekly", "MS": "Monthly", "QS": "Quarterly", "YS": "Yearly"}
        freq_tooltip = {
            "D": "Daily data (1 day intervals)",
            "W": "Weekly data (7 day intervals)", 
            "MS": "Monthly data (month start intervals)",
            "QS": "Quarterly data (3 month intervals)",
            "YS": "Yearly data (12 month intervals)"
        }

        def analyze_frequency(series):
            """Auto-detect frequency with confidence analysis."""
            if series is None or len(series) < 2:
                return "D", "low", {}
            
            try:
                sorted_dates = series.sort_values().dropna().drop_duplicates()
                if len(sorted_dates) < 2:
                    return "D", "low", {}
                
                diffs = sorted_dates.diff().dropna()
                if diffs.empty:
                    return "D", "low", {}
                
                try:
                    diff_days = diffs.dt.days
                    median_diff = diff_days.median()
                    std_diff = diff_days.std()
                    mean_diff = diff_days.mean()
                    min_diff = diff_days.min()
                    max_diff = diff_days.max()
                except:
                    return "D", "low", {}
                
                if pd.isna(median_diff):
                    return "D", "low", {}
                
                cv = std_diff / median_diff if median_diff > 0 else float('inf')
                
                if cv < 0.2:
                    confidence = "high"
                elif cv < 0.5:
                    confidence = "medium"
                else:
                    confidence = "low"
                
                freq = None
                if median_diff <= 1:
                    freq = "D"
                elif median_diff <= 7:
                    freq = "W"
                elif median_diff <= 31:
                    freq = "MS"
                elif median_diff <= 93:
                    freq = "QS"
                elif median_diff <= 366:
                    freq = "YS"
                else:
                    freq = "MS"
                
                details = {
                    "median_days": round(median_diff, 1),
                    "std_days": round(std_diff, 1),
                    "cv": round(cv, 2),
                    "date_range_days": (sorted_dates.max() - sorted_dates.min()).days,
                    "unique_dates": len(sorted_dates)
                }
                
                return freq, confidence, details
                
            except Exception:
                return "D", "low", {}
        
        def validate_frequency_choice(detected_freq_code, selected_freq_code, details):
            """Validate if user-selected frequency is reasonable for the data."""
            if not details:
                return "Could not analyze data structure", "warning"
            
            detected_days = details.get("median_days", 0)
            warnings = []
            
            if selected_freq_code != detected_freq_code:
                if selected_freq_code == "D" and detected_days > 7:
                    warnings.append(f"⚠️ Data appears to be {details.get('median_days', '?')} days apart on average, not daily")
                elif selected_freq_code == "W" and detected_days > 35:
                    warnings.append(f"⚠️ Data appears less frequent than weekly")
                elif selected_freq_code == "MS" and detected_days > 180:
                    warnings.append(f"⚠️ Data appears less frequent than monthly")
                elif selected_freq_code == "QS" and detected_days > 400:
                    warnings.append(f"⚠️ Data appears less frequent than quarterly")
            
            if details.get("cv", 0) > 0.5:
                warnings.append("⚠️ Date intervals are highly irregular - consider data quality")
            
            if warnings:
                return "\n".join(warnings), "warning"
            return None, "ok"
        
        # ── DETECT COLUMNS WITH COMPARISON ─────────────────────────────────────
        date_col = detect_date_column(cols)
        target_col = detect_target_column(num_cols) if num_cols else None
        
        # Frequency detection with confidence
        detected_freq_code = "D"
        selected_freq = "D"
        freq_confidence = "low"
        freq_details = {}
        if date_col:
            try:
                parsed_dates = pd.to_datetime(df_preview[date_col], errors="coerce")
                valid_dates = parsed_dates.dropna()
                if len(valid_dates) > 1:
                    detected_freq_code, freq_confidence, freq_details = analyze_frequency(valid_dates)
                    selected_freq = detected_freq_code
            except Exception:
                selected_freq = "D"
        
        freq_options = list(freq_map.keys())
        freq_idx = freq_options.index(selected_freq) if selected_freq in freq_options else 0
        
        # Show auto-detected mapping
        with st.expander("🔍 View/Edit Auto-Detected Mapping", expanded=False):
            col1, col2, col3 = st.columns(3)
            with col1:
                date_idx = cols.index(date_col) if date_col and date_col in cols else 0
                date_col = st.selectbox("📅 Date Column", cols, index=date_idx)
            with col2:
                if num_cols:
                    if target_col and target_col in num_cols:
                        target_idx = num_cols.index(target_col)
                    else:
                        target_idx = 0
                    target_col = st.selectbox("🎯 Target Column", num_cols, index=target_idx)
                else:
                    target_col = st.selectbox("🎯 Target Column", cols, index=0)
            with col3:
                freq_label = st.selectbox("📡 Data Frequency", freq_options, index=freq_idx, help=freq_tooltip.get(freq_options[freq_idx], ""))
                selected_freq_code = freq_label  # Keep the code for schema
                selected_freq_display = freq_map[freq_label]  # Get display name for UI
                
                # Validate frequency selection
                if freq_details:
                    validation_msg, validation_type = validate_frequency_choice(detected_freq_code, freq_label, freq_details)
                    if validation_msg:
                        if validation_type == "warning":
                            st.warning(validation_msg)
                        else:
                            st.error(validation_msg)
        
        # Display detected info
        if 'selected_freq_display' in dir():
            st.success(f"✅ Auto-detected: Date='{date_col}' | Target='{target_col}' | Frequency={selected_freq_display}")
        else:
            st.success(f"✅ Auto-detected: Date='{date_col}' | Target='{target_col}' | Frequency={freq_map.get(selected_freq, selected_freq)}")

        # Use the frequency code for schema (D, W, MS, etc.)
        freq_for_schema = selected_freq_code if 'selected_freq_code' in dir() else selected_freq
        
        custom_schema = {
            "required_cols": [date_col, target_col],
            "date_col": date_col,
            "target": target_col,
            "freq": freq_for_schema,
            "description": f"User Uploaded: {uploaded_file.name}"
        }

    except Exception as e:
        st.error(f"Error reading CSV for mapping: {e}")
        custom_schema = None
        # Set defaults in case of exception
        freq_for_schema = "D"

    start_run = st.button("🚀 Run ML Forecasting", type="primary")

    if start_run:
        custom_csv_path = None
        # Save the uploaded file to a temporary location so main.py can process it
        with st.spinner("Initializing pipeline..."):
            with tempfile.NamedTemporaryFile(delete=False, suffix=".csv") as tmp:
                tmp.write(uploaded_file.getvalue())
                custom_csv_path = tmp.name

            dataset_name_clean = os.path.splitext(uploaded_file.name)[0]
            
        # Use display name for log message
        display_freq = selected_freq_display if 'selected_freq_display' in dir() else freq_map.get(selected_freq, selected_freq)
        st.info(f"Training models for '{dataset_name_clean}'... Running {display_freq} aggregation.")
        
        start_time = time.time()
        try:
            with st.spinner(f"Running '{mode}' mode pipeline for {forecast_periods} periods. Please wait..."):
                results = run_pipeline(
                    dataset_name=dataset_name_clean,
                    forecast_periods=forecast_periods,
                    mode=mode,
                    custom_csv=custom_csv_path,
                    custom_schema=custom_schema
                )
            # Cleanup temp file
            os.remove(custom_csv_path)
            
            elapsed = time.time() - start_time
            st.success(f"✅ Pipeline completed in {elapsed:.1f} seconds!")
            
            # Render unified results
            display_pipeline_results(results)

        except Exception as e:
            st.error(f"❌ An error occurred during the ML Pipeline: {e}")
            st.code(traceback.format_exc())
        finally:
            # Ensure temp file cleanup even if exception occurs
            if custom_csv_path and os.path.exists(custom_csv_path):
                try:
                    os.remove(custom_csv_path)
                except Exception:
                    pass

else:
    # If no file is uploaded yet, optionally allow running default datasets 
    with st.expander("Or, run one of the built-in datasets instead", expanded=True):
        built_in_ds = ["All Datasets"] + list(DATASET_SCHEMAS.keys())
        selected_ds = st.selectbox("Select Built-in Dataset", built_in_ds)
        if st.button("Run Built-in Dataset"):
            if selected_ds == "All Datasets":
                with st.spinner(f"🚀 Running ALL Datasets in '{mode}' mode. This may take a few minutes..."):
                    start_time = time.time()
                    try:
                        all_results = run_all_datasets(mode=mode, forecast_periods=forecast_periods)
                        elapsed = time.time() - start_time
                        st.success(f"✅ Full benchmark completed in {elapsed:.1f} seconds!")
                        display_all_results_summary(all_results)
                    except Exception as e:
                        st.error(f"Batch processing error: {e}")
            else:
                with st.spinner(f"Running '{mode}' mode pipeline on '{selected_ds}'..."):
                    start_time = time.time()
                    try:
                        results = run_pipeline(
                            dataset_name=selected_ds,
                            forecast_periods=forecast_periods,
                            mode=mode
                        )
                        elapsed = time.time() - start_time
                        st.success(f"✅ Pipeline completed in {elapsed:.1f} seconds!")
                        
                        # Render unified results natively 
                        display_pipeline_results(results)
                    except Exception as e:
                        st.error(f"Pipeline error: {e}")
