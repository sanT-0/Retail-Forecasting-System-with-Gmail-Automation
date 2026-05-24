"""
Future Retail Sales Forecasting utilities.
"""

import os
import unicodedata
from functools import lru_cache

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from statsmodels.tsa.stattools import acf, adfuller


class C:
    HEADER = "\033[95m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    BOLD = "\033[1m"
    END = "\033[0m"


CONSOLE_REPLACEMENTS = {
    "₹": "Rs.",
    "→": "->",
    "—": "-",
    "–": "-",
    "−": "-",
    "•": "*",
    "═": "=",
    "─": "-",
    "║": "|",
    "│": "|",
    "╔": "+",
    "╗": "+",
    "╚": "+",
    "╝": "+",
    "✂": "[Split]",
    "✔": "[OK]",
    "✗": "[FAIL]",
    "✅": "[OK]",
    "❌": "[ERROR]",
    "⚡": "[Init]",
    "📂": "[Dataset]",
    "🎯": "[Mode]",
    "🔄": "[Aggregate]",
    "📊": "[Charts]",
    "🤖": "[Train]",
    "🔀": "[Ensemble]",
    "🧠": "[Stacking]",
    "📋": "[Results]",
    "🏆": "[Best]",
    "🔮": "[Forecast]",
    "📈": "[Summary]",
    "💾": "[Save]",
    "🚀": "[Run]",
    "📄": "[File]",
    "👋": "",
}


def safe_console_text(text) -> str:
    """Convert console output to plain ASCII for Windows-safe stdout."""
    safe_text = str(text)
    for source, target in CONSOLE_REPLACEMENTS.items():
        safe_text = safe_text.replace(source, target)
    safe_text = unicodedata.normalize("NFKD", safe_text)
    return safe_text.encode("ascii", errors="ignore").decode("ascii")


def safe_print(*values, sep=" ", end="\n"):
    print(safe_console_text(sep.join(str(value) for value in values)), end=end, flush=True)


def cprint(text, color=C.CYAN):
    safe_print(f"{color}{text}{C.END}")


def banner():
    safe_print(
        f"""{C.BOLD}{C.BLUE}
============================================================
  Future Retail Sales Forecasting Using Machine Learning
  Created by: sanT
  Indian Retail Market - Multi-Model Forecasting
============================================================
{C.END}"""
    )


def compute_metrics(y_true, y_pred, model_name: str = "") -> dict:
    """Compute evaluation metrics with stable defaults."""
    y_true = np.array(y_true, dtype=float)
    y_pred = np.array(y_pred, dtype=float)

    # Sanitize predictions — clip to a physically meaningful range for retail
    # (prevents sklearn check_array from crashing on any model's divergent output)
    y_pred = np.clip(np.nan_to_num(y_pred, nan=0.0, posinf=1e12, neginf=0.0), 0, 1e12)

    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)

    try:
        if np.std(y_true) < 1e-9:
            # If y_true is constant, r2_score is technically undefined. 
            # We return 1.0 if y_pred matches it, else 0.0 or negative based on error.
            r2 = 1.0 if np.allclose(y_true, y_pred, atol=1e-3) else 0.0
        else:
            r2 = r2_score(y_true, y_pred)
    except Exception:
        r2 = 0.0

    denom = np.abs(y_true)
    mask = denom > 1e-6
    if mask.any():
        mape = (np.abs(y_true[mask] - y_pred[mask]) / denom[mask]).mean() * 100
    else:
        mape = np.nan

    smape_denom = np.abs(y_true) + np.abs(y_pred)
    smape_mask = smape_denom > 1e-6
    if smape_mask.any():
        smape = 100 * np.mean(
            2 * np.abs(y_pred[smape_mask] - y_true[smape_mask]) / smape_denom[smape_mask]
        )
    else:
        smape = 0.0

    mdape_vals = (
        np.abs((y_true[mask] - y_pred[mask]) / y_true[mask]) * 100 if mask.any() else [0.0]
    )
    mdape = np.median(mdape_vals)

    return {
        "Model": model_name,
        "MAE": round(mae, 2),
        "MSE": round(mse, 2),
        "RMSE": round(rmse, 2),
        "R2": round(float(r2), 4),
        "MAPE%": round(mape, 2) if not np.isnan(mape) else "N/A",
        "sMAPE": round(smape, 2),
        "MDAPE": round(mdape, 2),
    }


DATASET_SCHEMAS = {
    "pharmacy_daily": {
        "required_cols": ["Date", "Revenue_INR"],
        "date_col": "Date",
        "target": "Revenue_INR",
        "freq": "D",
        "description": "Pharmacy Daily Sales (Healthcare)",
    },
    "electronics_gadgets": {
        "required_cols": ["Order_Date", "Revenue_INR"],
        "date_col": "Order_Date",
        "target": "Revenue_INR",
        "freq": "D",
        "description": "Electronics Gadgets (High-Value)",
    },
    "grocery_delivery": {
        "required_cols": ["Delivery_Date", "Revenue_INR"],
        "date_col": "Delivery_Date",
        "target": "Revenue_INR",
        "freq": "D",
        "description": "Grocery App Delivery (Hyper-local)",
    },
    "furniture_decor": {
        "required_cols": ["Week_End", "Revenue_INR"],
        "date_col": "Week_End",
        "target": "Revenue_INR",
        "freq": "W",
        "description": "Furniture & Home Decor (Seasonal)",
    },
    "luxury_brand": {
        "required_cols": ["Sale_Date", "Revenue_INR"],
        "date_col": "Sale_Date",
        "target": "Revenue_INR",
        "freq": "D",
        "description": "Luxury Brand (Elite High-Margin)",
    },
}


def load_and_validate(
    filepath: str, dataset_name: str, injected_schema: dict = None
) -> tuple[pd.DataFrame, dict]:
    """Load CSV with adaptive encoding and robust cleaning."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset not found: {filepath}")

    encodings = ["utf-8", "ISO-8859-1", "latin1", "cp1252"]
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(filepath, encoding=enc)
            break
        except UnicodeDecodeError:
            continue

    if df is None:
        raise ValueError(f"Unable to read CSV '{filepath}' with standard encodings.")

    if injected_schema:
        schema = dict(injected_schema)
    else:
        schema = DATASET_SCHEMAS.get(dataset_name)
        if schema is None:
            # ── 1. Date Column Detection ──
            date_cols = []
            for col in df.columns:
                if any(tok in col.lower() for tok in ["date", "time", "period", "day", "month", "week"]):
                    date_cols.append(col)
            
            if not date_cols:
                for col in df.columns:
                    if df[col].dtype == 'object' or pd.api.types.is_datetime64_any_dtype(df[col]):
                        try:
                            if pd.to_datetime(df[col].dropna().head(50), errors='coerce').notna().sum() > 10:
                                date_cols.append(col)
                        except Exception:
                            pass

            date_col = date_cols[0] if date_cols else None
            
            # ── 2. Target Column Detection ──
            num_cols = df.select_dtypes(include=np.number).columns.tolist()
            if date_col in num_cols:
                num_cols.remove(date_col)
                
            target_col = None
            if num_cols:
                FIRST_PRIORITY_KEYWORDS = ["revenue", "final_amount", "sales"]
                cn_lower = {col: col.lower().replace(" ", "_").replace("-", "_") for col in num_cols}
                
                for col in num_cols:
                    cn = cn_lower[col]
                    for kw in FIRST_PRIORITY_KEYWORDS:
                        if kw in cn:
                            try:
                                col_data = pd.to_numeric(df[col], errors="coerce")
                                valid_data = col_data.dropna()
                                if len(valid_data) >= 3 and valid_data.nunique() >= 2:
                                    target_col = col
                                    break
                            except Exception:
                                continue
                    if target_col:
                        break
                        
                if not target_col:
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
                    for col in num_cols:
                        try:
                            col_data = pd.to_numeric(df[col], errors="coerce")
                            valid_data = col_data.dropna()
                            
                            if len(valid_data) < 3 or valid_data.nunique() < 2:
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
                                
                                # Penalize IDs
                                if any(x in cn for x in ["id", "index", "code", "zip", "phone", "year", "month"]):
                                    score -= 500
    
                                scored_cols.append((col, score))
                        except Exception:
                            continue
                    
                    if scored_cols:
                        scored_cols.sort(key=lambda x: x[1], reverse=True)
                        target_col = scored_cols[0][0]
                    else:
                        target_col = num_cols[-1]

            if not date_col or not target_col:
                raise ValueError("Auto-detection failed. Please use custom mapping.")

                
            # ── 3. Frequency Detection ──
            freq = "D"
            try:
                dates = pd.to_datetime(df[date_col], errors='coerce').dropna().sort_values().drop_duplicates()
                if len(dates) >= 2:
                    diffs = dates.diff().dropna().dt.days
                    median_diff = diffs.median()
                    if median_diff >= 360: freq = "YS"
                    elif median_diff >= 85: freq = "QS"
                    elif median_diff >= 25: freq = "MS"
                    elif median_diff >= 6: freq = "W"
            except Exception:
                pass

            schema = {
                "required_cols": [date_col, target_col],
                "date_col": date_col,
                "target": target_col,
                "freq": freq,
                "description": f"Auto-detected ({dataset_name})",
            }
            schema = dict(schema)

    required_schema_keys = ["date_col", "target", "freq", "description"]
    missing_schema_keys = [key for key in required_schema_keys if key not in schema]
    if missing_schema_keys:
        raise ValueError(f"Schema is missing required keys: {missing_schema_keys}")

    required_cols = list(
        dict.fromkeys(schema.get("required_cols", []) + [schema["date_col"], schema["target"]])
    )
    schema["required_cols"] = required_cols

    missing = [column for column in required_cols if column not in df.columns]
    if missing:
        raise ValueError(f"Required columns missing: {missing}")

    date_col = schema["date_col"]
    target_col = schema["target"]

    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df.dropna(subset=[date_col, target_col], inplace=True)
    df = df.sort_values(date_col)

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[target_col] = pd.to_numeric(df[target_col], errors="coerce")
    df.dropna(subset=[date_col, target_col], inplace=True)
    df[target_col] = df[target_col].astype(float)
    df = df.sort_values(date_col)

    if df.empty:
        raise ValueError(
            f"No valid rows remain after parsing '{dataset_name}'. "
            f"Check date column '{date_col}' and target column '{target_col}'."
        )

    cprint(f"[OK] Validated '{dataset_name}': {len(df):,} rows | Target: {target_col}", C.GREEN)
    return df, schema


@lru_cache(maxsize=32)
def engineer_features_cached(
    vals: tuple, dates: tuple, lags: int = 14, windows: tuple = (3, 7, 14, 30), last_only: bool = False
) -> pd.DataFrame:
    """Core feature engineering with extreme numerical stability."""
    # Cap lags to prevent excessive feature explosion in neural networks
    lags = min(lags, 30)
    
    df = pd.DataFrame({"y": vals}, index=pd.to_datetime(dates))

    if last_only:
        # Ensure we have enough tail to calculate all lags up to lag_365
        max_req = max(lags, max(windows) if windows else 0, 366) + 1
        df = df.tail(max_req)

    df["year"]        = df.index.year
    df["month"]       = df.index.month
    df["day"]         = df.index.day
    df["dayofweek"]   = df.index.dayofweek
    df["dayofyear"]   = df.index.dayofyear
    df["quarter"]     = df.index.quarter
    df["weekofyear"]  = df.index.isocalendar().week.astype(int)
    df["is_weekend"]  = (df.index.dayofweek >= 5).astype(int)
    df["is_payday"]   = df.index.day.isin([1, 2, 15, 16, 30, 31]).astype(int)
    # Distance to payday (approximate)
    df["dist_payday"] = df.index.day.map(lambda d: min(abs(d-1), abs(d-15), abs(d-30)))
    df["is_month_start"] = df.index.is_month_start.astype(int)
    df["is_month_end"]   = df.index.is_month_end.astype(int)
    df["is_quarter_start"] = df.index.is_quarter_start.astype(int)
    df["is_weekend_month"] = df["is_weekend"] * df["month"]

    # Smooth monthly seasonality (better than raw month integer)
    df["month_sin"] = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"] = np.cos(2 * np.pi * df["month"] / 12)
    df["dow_sin"]   = np.sin(2 * np.pi * df["dayofweek"] / 7)
    df["dow_cos"]   = np.cos(2 * np.pi * df["dayofweek"] / 7)

    period = 365.25
    for k in [1, 2, 3]:
        df[f"sin_{k}"] = np.sin(2 * np.pi * k * df["dayofyear"] / period)
        df[f"cos_{k}"] = np.cos(2 * np.pi * k * df["dayofyear"] / period)

    # Monthly Fourier Harmonics (for recurring monthly patterns)
    period_m = 30.4375
    for k in [1, 2]:
        df[f"month_sin_{k}"] = np.sin(2 * np.pi * k * df.index.day / period_m)
        df[f"month_cos_{k}"] = np.cos(2 * np.pi * k * df.index.day / period_m)

    # Weekly Fourier Harmonics (for non-sinusoidal weekend spikes)
    for k in [1, 2, 3]:
        df[f"dow_sin_{k}"] = np.sin(2 * np.pi * k * df["dayofweek"] / 7)
        df[f"dow_cos_{k}"] = np.cos(2 * np.pi * k * df["dayofweek"] / 7)

    # Indian Festivals with added Navratri
    festivals = [(10, 24), (3, 25), (4, 10), (10, 12), (12, 25), (1, 1), (8, 15), (1, 26), (8, 19), (11, 1), (10, 3)]

    def get_festival_score(date):
        score = 0
        for month, day in festivals:
            try:
                fest = pd.Timestamp(date.year, month, day)
                diff = abs((date - fest).days)
                if diff <= 14:
                    # Linear decay score from 1.0 to 0.0
                    score = max(score, (15 - diff) / 15)
            except (ValueError, OverflowError):
                continue
        return score

    df["festival_score"] = [get_festival_score(date) for date in df.index]
    df["near_festival"]  = (df["festival_score"] > 0).astype(int)

    # Lags
    for lag in range(1, lags + 1):
        df[f"lag_{lag}"] = df["y"].shift(lag)
    
    # Differenced Lags
    if lags >= 2:
        df["lag_diff_1"] = df["lag_1"] - df["lag_2"]
    if lags >= 8:
        df["lag_diff_7"] = df["lag_1"] - df["lag_7"]

    # Seasonal Lags (Critical for Annual/Semi-Annual patterns)
    if len(df) > 182:
        df["lag_182"] = df["y"].shift(182)
    if len(df) > 365:
        df["lag_365"] = df["y"].shift(365)

    # Rolling Windows
    for window in [3, 7, 14, 30, 60, 90]:
        if len(df) > window:
            shifted = df["y"].shift(1)
            df[f"roll_mean_{window}"] = shifted.rolling(window).mean()
            df[f"roll_std_{window}"] = shifted.rolling(window).std().fillna(0)
            df[f"roll_max_{window}"] = shifted.rolling(window).max().ffill().fillna(0)
            df[f"roll_min_{window}"] = shifted.rolling(window).min().ffill().fillna(0)
            # Add rolling coefficient of variation for volatility tracking
            df[f"roll_cv_{window}"] = (df[f"roll_std_{window}"] / (df[f"roll_mean_{window}"] + 1e-9)).fillna(0)

    # EWMA (Exponential Smoothing)
    for span in [7, 30]:
        df[f"ewm_mean_{span}"] = df["y"].shift(1).ewm(span=span).mean().fillna(0)

    # Absolute trend to prevent resetting during last_only=True prediction loops
    epoch = pd.Timestamp("2020-01-01").toordinal()
    df["trend"] = df.index.map(pd.Timestamp.toordinal) - epoch
    
    # Clip trend to prevent exploding squares in very long datasets
    df["trend"] = df["trend"].clip(lower=-5000, upper=5000) 
    df["trend_sq"] = (df["trend"] / 100) ** 2  # Scaled down to prevent huge numbers
    
    # Interaction Features (Critical for Multiplicative components in Linear models)
    df["fest_trend"] = df["festival_score"] * (df["trend"] / 100)
    df["fest_weekend"] = df["festival_score"] * df["is_weekend"]
    df["payday_weekend"] = df["is_payday"] * df["is_weekend"]
    df["trend_month"] = (df["trend"] / 100) * df["month_sin"]
    df["trend_dow"] = (df["trend"] / 100) * df["dow_sin_1"]
    
    # Final data cleaning: Catch any NaNs or Infs (even hidden ones)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.fillna(0)
    
    # Final layer: clip to float64 safe range (ignore if already done, but be explicit)
    mask = df.columns != "Date"
    df[df.columns[mask]] = df[df.columns[mask]].astype(float).clip(-1e15, 1e15)

    return df


def engineer_features(
    series: pd.Series, lags: int = 14, windows: tuple = (3, 7, 14, 30, 60, 90), last_only: bool = False
) -> pd.DataFrame:
    """Public API for cached feature engineering."""
    vals = tuple(series.values)
    dates = tuple(series.index.values)
    if isinstance(windows, list):
        windows = tuple(windows)

    df = engineer_features_cached(vals, dates, lags, windows, last_only)
    if last_only:
        return df.tail(1)
    return df


def winsorize_series(series: pd.Series, lower: float = 0.005, upper: float = 0.995) -> pd.Series:
    """
    Clip extreme outliers at the given quantiles.
    Relaxed quantiles to preserve legitimate signal spikes (like Diwali).
    """
    lo = series.quantile(lower)
    hi = series.quantile(upper)
    return series.clip(lower=lo, upper=hi)


def aggregate_time_series(df: pd.DataFrame, schema: dict, group_col: str = None) -> pd.Series:
    """Aggregate a dataset into a uniform time series, then winsorize outliers."""
    del group_col
    date_col = schema["date_col"]
    target   = schema["target"]
    freq     = schema["freq"]

    ts = df.groupby(pd.Grouper(key=date_col, freq=freq))[target].sum()
    ts = ts.asfreq(freq).fillna(0)
    ts = ts.sort_index()

    # Remove zero-only prefix (startup ramp-up with no data)
    first_nonzero = (ts > 0).idxmax()
    ts = ts.loc[first_nonzero:]

    # Winsorize to reduce spike influence on R²
    if ts.std() > 0:
        ts = winsorize_series(ts)

    return ts


def temporal_split(df: pd.DataFrame, test_ratio: float = 0.2):
    """Chronological train/test split."""
    n = len(df)
    split = int(n * (1 - test_ratio))
    return df.iloc[:split], df.iloc[split:]


def build_forecast_index(last_date: pd.Timestamp, periods: int, freq: str) -> pd.DatetimeIndex:
    return pd.date_range(
        start=last_date + pd.tseries.frequencies.to_offset(freq), periods=periods, freq=freq
    )


def check_stationarity(series: pd.Series) -> dict:
    """Perform the Augmented Dickey-Fuller test."""
    try:
        res = adfuller(series.dropna())
        p_val = res[1]
        return {
            "is_stationary": p_val < 0.05,
            "p_value": round(p_val, 4),
            "test_stat": round(res[0], 4),
            "lags_used": res[2],
        }
    except Exception:
        return {"is_stationary": False, "p_value": 1.0, "test_stat": 0, "lags_used": 0}


def detect_optimal_lags(series: pd.Series) -> int:
    """Use ACF significance to determine an optimal lag window."""
    try:
        y = series.dropna().values
        if len(y) < 10:
            return 7
        acf_vals, confint = acf(y, nlags=min(40, len(y) // 2 - 1), alpha=0.05)
        significant = np.where(np.abs(acf_vals[1:]) > confint[1:, 1])[0]
        if len(significant) == 0:
            return 7
        opt_lag = int(significant[-1]) + 1
        return min(max(opt_lag, 7), 30)
    except Exception:
        return 14


def print_metrics_table(metrics_list: list):
    """Pretty-print metrics as an ASCII table."""
    if not metrics_list:
        return

    keys = list(metrics_list[0].keys())
    col_w = {key: max(len(key), max(len(str(metric[key])) for metric in metrics_list)) + 2 for key in keys}
    sep = "+" + "+".join("-" * col_w[key] for key in keys) + "+"
    hdr = "|" + "|".join(f" {key:<{col_w[key] - 1}}" for key in keys) + "|"
    safe_print(f"\n{C.BOLD}{sep}\n{hdr}\n{sep}{C.END}")

    best_r2 = max((metric.get("R2", float("-inf")) for metric in metrics_list), default=float("-inf"))
    for metric in metrics_list:
        row = "|" + "|".join(f" {str(metric[key]):<{col_w[key] - 1}}" for key in keys) + "|"
        if metric.get("R2", float("-inf")) == best_r2:
            safe_print(f"{C.GREEN}{row}{C.END}")
        else:
            safe_print(row)
    safe_print(sep)


def format_inr(value: float) -> str:
    """Format a number as Indian Rupees with lakh/crore notation."""
    if value >= 1e7:
        return f"₹{value / 1e7:.2f} Cr"
    if value >= 1e5:
        return f"₹{value / 1e5:.2f} L"
    return f"₹{value:,.0f}"
