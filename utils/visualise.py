"""

  Future Retail Sales Forecasting  Visualisations               
  Created by: sanT                                                

All charts use Matplotlib + Seaborn.
Saved as high-resolution PNG files inside outputs/.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless  no display needed
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import seaborn as sns
import os
import warnings
from statsmodels.tsa.stattools import acf, pacf
warnings.filterwarnings("ignore")

#  Theme 
PALETTE = ["#2563EB","#16A34A","#DC2626","#D97706","#7C3AED",
           "#0891B2","#BE185D","#65A30D","#EA580C","#0F766E",
           "#9333EA","#B45309","#15803D","#1D4ED8","#B91C1C","#6D28D9"]
sns.set_theme(style="whitegrid", palette=PALETTE)
plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.labelsize": 11,
    "figure.dpi": 120,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "legend.framealpha": 0.9,
})

def _save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print(f"   Saved  {path}")
    return path

# 
# 1. TIME SERIES OVERVIEW
# 
def plot_time_series(series: pd.Series, title: str, out_dir: str,
                     resample: str = None) -> str:
    """Full time-series with trend overlay and moving average."""
    if resample:
        series = series.resample(resample).sum()
    series = series.dropna()

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True,
                              gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f" {title}", fontweight="bold", fontsize=16, y=1.01)

    ax = axes[0]
    ax.fill_between(series.index, series.values, alpha=0.15, color=PALETTE[0])
    ax.plot(series.index, series.values, lw=1.2, color=PALETTE[0], label="Actual Sales")

    # Moving average
    for w, c in [(7, PALETTE[1]), (30, PALETTE[2])]:
        if len(series) >= w:
            ma = series.rolling(w).mean()
            ax.plot(ma.index, ma.values, lw=2, color=c, label=f"{w}-period MA")

    # Trend line
    x_num = np.arange(len(series))
    z = np.polyfit(x_num, series.values, 1)
    p = np.poly1d(z)
    ax.plot(series.index, p(x_num), "--", lw=1.5, color=PALETTE[3], label="Trend")

    ax.set_ylabel("Revenue ()")
    ax.legend(loc="upper left", fontsize=9)
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e5:.1f}L" if v >= 1e5 else f"{v:,.0f}")
    )

    # Lower panel: MoM % change
    ax2 = axes[1]
    pct = series.pct_change() * 100
    colors = [PALETTE[1] if v >= 0 else PALETTE[2] for v in pct]
    ax2.bar(pct.index, pct.values, color=colors, width=0.8, alpha=0.75)
    ax2.axhline(0, color="black", lw=0.8)
    ax2.set_ylabel("% Change")
    ax2.set_xlabel("Date")

    fig.tight_layout()
    return _save(fig, os.path.join(out_dir, "01_time_series_overview.png"))

# 
# 2. FORECAST vs ACTUAL
# 
def plot_forecast(train: pd.Series, test: pd.Series,
                  preds: np.ndarray, future_dates: pd.DatetimeIndex,
                  future_preds: np.ndarray, model_name: str,
                  out_dir: str) -> str:
    """Shows train history, test vs. prediction, and future forecast."""
    fig, ax = plt.subplots(figsize=(16, 6))
    fig.suptitle(f" Forecast  {model_name}", fontweight="bold", fontsize=15)

    # Train
    ax.plot(train.index, train.values, color=PALETTE[0], lw=1.2,
            label="Training Data", alpha=0.8)
    # Test actual
    ax.plot(test.index, test.values, color=PALETTE[1], lw=2,
            label="Actual (Test)", zorder=5)
    # Test predictions
    test_idx = test.index[:len(preds)]
    ax.plot(test_idx, preds[:len(test)], "--", color=PALETTE[2], lw=2,
            label="Predicted (Test)", zorder=4)
    # Future forecast
    ax.plot(future_dates, future_preds, "-o", color=PALETTE[3], lw=2.5,
            markersize=4, label=f"Future Forecast ({len(future_dates)} periods)")
    # Confidence interval 10%
    lo = future_preds * 0.90
    hi = future_preds * 1.10
    ax.fill_between(future_dates, lo, hi, alpha=0.2, color=PALETTE[3],
                    label="10% Confidence Band")
    # Divider
    ax.axvline(train.index[-1], color="grey", ls="--", lw=1, label="Train/Test Split")
    ax.axvline(test.index[-1], color="black", ls=":", lw=1, label="Forecast Start")

    ax.set_ylabel("Revenue ()")
    ax.set_xlabel("Date")
    ax.legend(fontsize=8, loc="upper left")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e5:.1f}L" if v >= 1e5 else f"{v:,.0f}")
    )
    fig.tight_layout()
    safe_name = model_name.replace(" ", "_").replace("(", "").replace(")", "").replace("=","")
    return _save(fig, os.path.join(out_dir, f"forecast_{safe_name}.png"))

# 
# 3. MODEL COMPARISON BAR CHART
# 
def plot_model_comparison(metrics_list: list, out_dir: str) -> str:
    """Horizontal bar chart comparing all models by RMSE, MAE, and R2."""
    df = pd.DataFrame(metrics_list).sort_values("RMSE")
    n  = len(df)

    fig, axes = plt.subplots(1, 3, figsize=(18, max(6, n * 0.5)))
    fig.suptitle(" Model Performance Comparison", fontweight="bold", fontsize=16)

    for ax, metric, color_col, fmt in [
        (axes[0], "RMSE",   "RMSE",  "{:.0f}"),
        (axes[1], "MAE",    "MAE",   "{:.0f}"),
        (axes[2], "R2",     "R2",    "{:.4f}"),
    ]:
        vals   = df[metric].values.astype(float)
        models = df["Model"].values
        colors = [PALETTE[i % len(PALETTE)] for i in range(n)]

        if metric == "R2":
            # higher is better  reverse sort for display
            order  = np.argsort(vals)[::-1]
            vals   = vals[order]
            models = models[order]
            colors = [colors[i] for i in order]

        bars = ax.barh(models, vals, color=colors, edgecolor="white", height=0.6)
        ax.set_title(f"{metric}", fontweight="bold")
        ax.set_xlabel(metric)

        for bar, v in zip(bars, vals):
            ax.text(bar.get_width() + bar.get_width() * 0.01,
                    bar.get_y() + bar.get_height() / 2,
                    fmt.format(v), va="center", fontsize=7.5)
        ax.invert_yaxis()

    plt.close('all') # Clear memory
    fig.tight_layout()
    return _save(fig, os.path.join(out_dir, "02_model_comparison.png"))

# 
# 4. ERROR DISTRIBUTION
# 
def plot_error_distribution(y_true: np.ndarray, y_pred: np.ndarray,
                             model_name: str, out_dir: str) -> str:
    residuals = y_true - y_pred
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    fig.suptitle(f" Residual Analysis  {model_name}", fontweight="bold")

    # Histogram
    axes[0].hist(residuals, bins=30, color=PALETTE[0], edgecolor="white", alpha=0.85)
    axes[0].axvline(0, color="red", ls="--")
    axes[0].set_title("Residual Distribution")
    axes[0].set_xlabel("Residual ()")

    # Scatter: actual vs predicted
    axes[1].scatter(y_true, y_pred, alpha=0.4, color=PALETTE[1], s=15)
    mn, mx = min(y_true.min(), y_pred.min()), max(y_true.max(), y_pred.max())
    axes[1].plot([mn, mx], [mn, mx], "r--", lw=1.5, label="Perfect fit")
    axes[1].set_title("Actual vs Predicted")
    axes[1].set_xlabel("Actual ()")
    axes[1].set_ylabel("Predicted ()")
    axes[1].legend(fontsize=8)

    # Residuals over time
    axes[2].plot(residuals, color=PALETTE[2], lw=0.8, alpha=0.7)
    axes[2].axhline(0, color="black", lw=1)
    axes[2].fill_between(range(len(residuals)), residuals, 0,
                          where=residuals > 0, color=PALETTE[1], alpha=0.3)
    axes[2].fill_between(range(len(residuals)), residuals, 0,
                          where=residuals < 0, color=PALETTE[2], alpha=0.3)
    axes[2].set_title("Residuals Over Time")
    axes[2].set_xlabel("Step")

    fig.tight_layout()
    safe = model_name.replace(" ", "_").replace("(","").replace(")","").replace("=","")
    return _save(fig, os.path.join(out_dir, f"errors_{safe}.png"))

# 
# 5. FEATURE IMPORTANCE
# 
def plot_feature_importance(importance: pd.Series, model_name: str,
                             out_dir: str, top_n: int = 20) -> str:
    top = importance.head(top_n)
    fig, ax = plt.subplots(figsize=(10, max(5, len(top) * 0.4)))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(top))]
    ax.barh(top.index, top.values, color=colors, edgecolor="white", height=0.6)
    ax.set_title(f" Top {top_n} Feature Importances  {model_name}",
                 fontweight="bold", fontsize=13)
    ax.set_xlabel("Importance Score")
    ax.invert_yaxis()
    fig.tight_layout()
    safe = model_name.replace(" ", "_").replace("(","").replace(")","").replace("=","")
    return _save(fig, os.path.join(out_dir, f"feature_imp_{safe}.png"))

# 
# 6. SEASONAL DECOMPOSITION (manual)
# 
def plot_seasonal_decomposition(series: pd.Series, period: int,
                                 title: str, out_dir: str) -> str:
    """Manual trend/seasonal/residual decomposition."""
    if len(series) < 2 * period:
        return ""
    y   = series.values.astype(float)
    n   = len(y)

    # Trend via centred MA
    trend = np.full(n, np.nan)
    half  = period // 2
    for i in range(half, n - half):
        trend[i] = y[i - half:i + half + 1].mean()

    seasonal_vals = y - trend
    # Average seasonal pattern
    seasonal = np.full(n, np.nan)
    for i in range(n):
        idx = i % period
        same = [seasonal_vals[j] for j in range(n) if (j % period == idx) and not np.isnan(seasonal_vals[j])]
        if same:
            seasonal[i] = np.mean(same)

    residual = y - trend - seasonal
    
    fig, axes = plt.subplots(4, 1, figsize=(14, 10), sharex=True)

    # Check if we have enough non-NaN data to plot
    if np.isnan(trend).all() or np.isnan(seasonal).all():
        plt.close(fig)
        return ""
    fig.suptitle(f" Seasonal Decomposition  {title}", fontweight="bold", fontsize=14)

    for ax, data, label, color in zip(
        axes,
        [y, trend, seasonal, residual],
        ["Observed", "Trend", "Seasonal", "Residual"],
        PALETTE[:4]
    ):
        ax.plot(series.index, data, color=color, lw=1.2)
        ax.set_ylabel(label, fontsize=10)
        ax.fill_between(series.index, data, alpha=0.15, color=color)

    axes[-1].set_xlabel("Date")
    fig.tight_layout()
    return _save(fig, os.path.join(out_dir, "03_seasonal_decomposition.png"))

# 
# 7. CATEGORY / CITY HEATMAP
# 
def plot_category_heatmap(df: pd.DataFrame, row_col: str, col_col: str,
                           val_col: str, title: str, out_dir: str) -> str:
    pivot = df.pivot_table(values=val_col, index=row_col,
                            columns=col_col, aggfunc="sum", fill_value=0)
    # Normalise per row for visibility
    pivot_norm = pivot.div(pivot.sum(axis=1), axis=0)

    fig, ax = plt.subplots(figsize=(max(10, len(pivot.columns)), max(6, len(pivot) * 0.5)))
    sns.heatmap(pivot_norm, cmap="YlOrRd", annot=False, fmt=".0%",
                linewidths=0.3, ax=ax, cbar_kws={"label": "Share of Sales"})
    ax.set_title(f" {title}", fontweight="bold", fontsize=13)
    ax.set_xlabel(col_col)
    ax.set_ylabel(row_col)
    fig.tight_layout()
    return _save(fig, os.path.join(out_dir, "04_category_heatmap.png"))

# 
# 8. FUTURE FORECAST SUMMARY TABLE (rendered as image)
# 
def plot_forecast_table(future_dates: pd.DatetimeIndex, future_preds: np.ndarray,
                         model_name: str, out_dir: str) -> str:
    """Render forecast values as a styled table image."""
    rows = []
    for dt, val in zip(future_dates, future_preds):
        rows.append([dt.strftime("%Y-%m-%d"), f"{val:>12,.0f}",
                     f"{val*0.90:>12,.0f}", f"{val*1.10:>12,.0f}"])

    fig, ax = plt.subplots(figsize=(10, max(4, len(rows) * 0.35 + 1)))
    ax.axis("off")
    tbl = ax.table(
        cellText  = rows,
        colLabels = ["Date", "Forecast ()", "Lower Band ()", "Upper Band ()"],
        cellLoc   = "center",
        loc       = "center",
    )
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9)
    tbl.scale(1, 1.4)

    for (r, c), cell in tbl.get_celld().items():
        if r == 0:
            cell.set_facecolor(PALETTE[0])
            cell.set_text_props(color="white", fontweight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#F0F4FF")
        else:
            cell.set_facecolor("#FFFFFF")

    ax.set_title(f" Future Forecast  {model_name}", fontweight="bold",
                 fontsize=13, pad=12)
    fig.tight_layout()
    safe = model_name.replace(" ", "_").replace("(","").replace(")","").replace("=","")
    return _save(fig, os.path.join(out_dir, f"forecast_table_{safe}.png"))

# 
# 9. ROLLING FORECAST ACCURACY
# 
def plot_rolling_accuracy(series: pd.Series, model, window: int = 30,
                           out_dir: str = ".", step: int = None) -> str:
    """Walk-forward MAPE across the test window. Optimized with step sampling."""
    results = []
    n = len(series)
    
    # Optimization: If no step defined, target ~20-25 fits instead of n fits
    if step is None:
        step = max(1, (n - window) // 20)
        
    for i in range(window, n, step):
        train_s = series.iloc[:i]
        actual  = series.iloc[i]
        try:
            model.fit(train_s)
            pred = model.predict(1)[0]
            mape = abs(actual - pred) / (actual + 1e-9) * 100
            results.append({"date": series.index[i], "MAPE": mape})
        except Exception:
            pass

    if not results:
        return ""

    res_df = pd.DataFrame(results).set_index("date")
    fig, ax = plt.subplots(figsize=(12, 4))
    ax.plot(res_df.index, res_df["MAPE"], marker='o', markersize=4, 
            color=PALETTE[0], lw=1.2, alpha=0.7)
    ax.axhline(res_df["MAPE"].mean(), color="red", ls="--",
               label=f"Mean MAPE: {res_df['MAPE'].mean():.1f}%")
    ax.set_title(f" Rolling Walk-Forward MAPE (sampled every {step} pds)", fontweight="bold")
    ax.set_xlabel("Date")
    ax.set_ylabel("MAPE (%)")
    ax.set_ylim(0, min(100, res_df["MAPE"].max() * 1.1) if not res_df.empty else 100)
    ax.legend()
    ax.fill_between(res_df.index, res_df["MAPE"], alpha=0.15, color=PALETTE[0])
    fig.tight_layout()
    return _save(fig, os.path.join(out_dir, "05_rolling_accuracy.png"))

# 
# 10. CORRELATION MATRIX (for datasets with multiple features)
# 
def plot_correlation_matrix(df: pd.DataFrame, title: str, out_dir: str) -> str:
    num_df = df.select_dtypes(include=np.number)
    if num_df.shape[1] < 2 or num_df.std().sum() == 0:
        return ""
    corr = num_df.corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    fig, ax = plt.subplots(figsize=(max(8, len(corr)), max(6, len(corr)*0.7)))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
                center=0, ax=ax, linewidths=0.3, annot_kws={"size": 7})
    ax.set_title(f" Feature Correlation  {title}", fontweight="bold", fontsize=13)
    fig.tight_layout()
    return _save(fig, os.path.join(out_dir, "06_correlation_matrix.png"))

# 
# 11. MONTHLY / WEEKLY REVENUE BARS
# 
def plot_period_revenue(series: pd.Series, freq: str, title: str, out_dir: str,
                         fname: str = "07_period_revenue.png") -> str:
    agg = series.resample(freq).sum()
    fig, ax = plt.subplots(figsize=(max(10, len(agg)//3), 5))
    colors = [PALETTE[i % len(PALETTE)] for i in range(len(agg))]
    ax.bar(agg.index, agg.values, width=15 if freq=="ME" else 5,
           color=colors, edgecolor="white", alpha=0.85)
    ax.set_title(f" {title}", fontweight="bold", fontsize=13)
    ax.set_ylabel("Revenue ()")
    ax.yaxis.set_major_formatter(
        matplotlib.ticker.FuncFormatter(lambda v, _: f"{v/1e5:.1f}L" if v>=1e5 else f"{v:,.0f}")
    )
    fig.autofmt_xdate()
    fig.tight_layout()
    return _save(fig, os.path.join(out_dir, fname))


#  GOD LEVEL DIAGNOSTICS 
def plot_acf_pacf(series: pd.Series, title: str, out_dir: str) -> str:
    """God Level: Diagnostic Autocorrelation and Partial Autocorrelation plots."""
    from statsmodels.tsa.stattools import acf, pacf
    import numpy as np
    
    y = series.dropna().values
    if len(y) < 10: return "" # Safeguard
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 5))
    
    lags = min(40, len(y) // 2 - 1)
    
    # ACF
    acf_vals = acf(y, nlags=lags)
    ax1.bar(range(len(acf_vals)), acf_vals, width=0.5, color="#3498db")
    ax1.axhline(y=0, color='black', linestyle='-', alpha=0.5)
    ax1.axhline(y=-1.96/np.sqrt(len(y)), linestyle='--', color='gray')
    ax1.axhline(y=1.96/np.sqrt(len(y)), linestyle='--', color='gray')
    ax1.set_title(f"Autocorrelation (ACF)  {title}")
    
    # PACF
    try:
        pacf_vals = pacf(y, nlags=lags)
        ax2.bar(range(len(pacf_vals)), pacf_vals, width=0.5, color="#e67e22")
        ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
        ax2.axhline(y=-1.96/np.sqrt(len(y)), linestyle='--', color='gray')
        ax2.axhline(y=1.96/np.sqrt(len(y)), linestyle='--', color='gray')
        ax2.set_title(f"Partial Autocorrelation (PACF)  {title}")
    except Exception:
        ax2.set_title("PACF Calculation Failed (Too few points)")
    
    plt.tight_layout()
    return _save(fig, os.path.join(out_dir, "diagnostic_acf_pacf.png"))
