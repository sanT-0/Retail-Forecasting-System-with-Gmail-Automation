"""

  Future Retail Sales Forecasting  All ML Models                 
  Created by: sanT                                                

Models:
  1. Naive Baseline (Moving Average)
  2. Seasonal Naive Baseline
  3. AR Model (Autoregressive OLS)
  4. Holt-Winters (Triple Exponential Smoothing)
  5. Linear Regression
  6. Differenced + Linear Regression
  7. Ridge Regression
  8. Lasso Regression
  9. ElasticNet
 10. Bayesian Ridge
 11. Huber Regressor
 12. Decision Tree
 13. K-Nearest Neighbors
 14. Support Vector Regression (SVR)
 15. Extra Trees
 16. Random Forest
 17. Gradient Boosting (sklearn)
 18. HistGradientBoosting (native LightGBM)
 19. XGBoost
 20. LightGBM
 21. CatBoost
 22. MLP Neural Network (LSTM-style dense architecture)
 23. Weighted Ensemble (Meta-learner)
 24. Stacking Regressor (God-Mode Meta-Regressor)
"""

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from sklearn.linear_model import (LinearRegression, Ridge, Lasso, ElasticNet,
                                   BayesianRidge, HuberRegressor)
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor,
                               ExtraTreesRegressor, VotingRegressor, 
                               HistGradientBoostingRegressor)
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import TimeSeriesSplit, cross_val_score, RandomizedSearchCV
import joblib
import os

try:
    import xgboost as xgb
except ImportError:
    xgb = None

try:
    import lightgbm as lgb
except ImportError:
    lgb = None

try:
    import catboost as cb
except ImportError:
    cb = None

from utils.helpers import (compute_metrics, engineer_features, 
                            temporal_split, detect_optimal_lags, C, cprint)


# 
# BASE FORECASTER
# 
class BaseForecaster:
    """Abstract base for all forecasters."""
    name: str = "Base"

    def fit(self, series: pd.Series): ...
    def predict(self, steps: int) -> np.ndarray: ...
    def backtest(self, series: pd.Series, test_size: float = 0.2) -> dict:
        n = len(series)
        split = int(n * (1 - test_size))
        train, test = series.iloc[:split], series.iloc[split:]
        self.fit(train)
        preds = self.predict(len(test))
        return compute_metrics(test.values, preds[:len(test)], self.name)


# 
# 1. NAIVE BASELINE
# 
class NaiveForecaster(BaseForecaster):
    """Predicts the last observed value (or moving average)."""
    name = "Naive Baseline"
    def __init__(self, window: int = 7):
        self.window = window
        self._last_vals = None

    def fit(self, series: pd.Series):
        self._last_vals = series.values[-self.window:]
        return self

    def predict(self, steps: int) -> np.ndarray:
        base = self._last_vals.mean()
        return np.full(steps, base)


#
# 1-B. SEASONAL NAIVE BASELINE
#
class SeasonalNaiveForecaster(BaseForecaster):
    """Predicts values by repeating the previous seasonal cycle (y_t = y_t-k)."""
    name = "Seasonal Naive"
    def __init__(self, season_len: int = 7):
        self.k = season_len
        self._history = None

    def fit(self, series: pd.Series):
        self._history = series.values.tolist()
        return self

    def predict(self, steps: int) -> np.ndarray:
        history = list(self._history)
        preds = []
        for i in range(steps):
            # Pick value from 1 season ago
            idx = len(history) - self.k
            val = history[idx] if idx >= 0 else history[-1]
            val = max(float(val), 0)
            preds.append(val)
            history.append(val)
        return np.array(preds)


#
# 1-C. DIFFERENCED WRAPPER (Auto-Stationarity)
#
class DifferencedForecaster(BaseForecaster):
    """
    Wrapper that transforms a series using First-Differencing
    before fitting/predicting with a base forecaster.
    Useful for non-stationary data with strong trends.
    """
    def __init__(self, base_forecaster):
        self.base = base_forecaster
        self.name = f"Diff + {base_forecaster.name}"
        self._last_val = None

    def fit(self, series: pd.Series):
        self._last_val = series.iloc[-1]
        diff_series = series.diff().dropna()
        self.base.fit(diff_series)
        return self

    def predict(self, steps: int) -> np.ndarray:
        diff_preds = self.base.predict(steps)
        # Re-integrate: y_t = y_t-1 + diff_t
        final_preds = []
        curr = self._last_val
        for dp in diff_preds:
            curr += dp
            final_preds.append(max(float(curr), 0))
        return np.array(final_preds)


# 
# 2-13: SKLEARN-BASED FEATURE MODELS (shared base)
# 
class FeatureForecaster(BaseForecaster):
    """
    Feature-engineering forecaster.
    Builds lag/calendar features, trains an sklearn regressor,
    then iteratively predicts future steps.
    """
    def __init__(self, estimator, name: str, lags: int = 14, scale: bool = True, 
                 log_transform: bool = True, use_robust_scaler: bool = False):
        self.estimator     = estimator
        self.name          = name
        self.lags          = min(lags, 30) # Hard ceiling for feature stability
        self.scale         = scale
        self.log_transform = log_transform
        self.use_robust_scaler = use_robust_scaler
        self._model        = None
        self._scaler       = None
        self._train_df     = None
        self._feature_cols = None
        self._do_log       = False  # set at fit-time

    def _build_features(self, series: pd.Series) -> pd.DataFrame:
        return engineer_features(series, lags=self.lags)

    def fit(self, series: pd.Series):
        self._series_tail = series.copy()

        # Auto log1p: apply when all values are positive (typical for sales data)
        # This stabilises variance and dramatically improves R² on skewed distributions.
        self._do_log = (
            self.log_transform
            and (series >= 0).all()
            and len(series) >= 10
        )
        work = np.log1p(series) if self._do_log else series
        self._series_tail_work = work.copy()

        feat_df = self._build_features(work)
        self._feature_cols = [c for c in feat_df.columns if c != "y"]
        
        # Pull raw arrays
        X = feat_df[self._feature_cols].values
        y = feat_df["y"].values

        # Final Numerical Disinfectant: Guarantee float64 and finite values
        X = np.nan_to_num(X.astype(np.float64), nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float64)
        y = np.nan_to_num(y.astype(np.float64), nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float64)

        if self.scale:
            self._scaler = RobustScaler() if self.use_robust_scaler else StandardScaler()
            X = self._scaler.fit_transform(X)
            
            # Post-scaling disinfectant (for zero-variance columns)
            X = np.nan_to_num(X, nan=0.0, posinf=1e10, neginf=-1e10).astype(np.float64)

        self._model = self.estimator
        try:
            # Memory layout optimization and final clipping
            X_clean = np.ascontiguousarray(np.clip(X, -1e15, 1e15))
            y_clean = np.ascontiguousarray(np.clip(y, -1e15, 1e15))
            self._model.fit(X_clean, y_clean)
        except Exception as e:
            cprint(f"       [CRITICAL ERROR] {self.name} fit failed: {e}", C.YELLOW)
            raise e
            
        self._train_df = feat_df
        return self

    def tune(self, series: pd.Series, n_iter: int = 20):
        """Perform hyperparameter optimization using TimeSeriesSplit."""
        # Setup data
        work = np.log1p(series) if self.log_transform and (series >= 0).all() else series
        feat_df = self._build_features(work)
        X = feat_df[[c for c in feat_df.columns if c != "y"]].values
        y = feat_df["y"].values
        X = np.nan_to_num(X.astype(float), nan=0.0)
        y = np.nan_to_num(y.astype(float), nan=0.0)

        if self.scale:
            scaler = RobustScaler() if self.use_robust_scaler else StandardScaler()
            X = scaler.fit_transform(X)

        param_grids = {
            "XGBoost": {
                "n_estimators": [300, 500, 800],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [3, 4, 5, 6],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9]
            },
            "LightGBM": {
                "n_estimators": [300, 500, 800],
                "learning_rate": [0.01, 0.03, 0.05, 0.1],
                "max_depth": [3, 4, 5, 6],
                "subsample": [0.7, 0.8, 0.9],
                "colsample_bytree": [0.7, 0.8, 0.9]
            },
            "CatBoost": {
                "iterations": [300, 500, 800],
                "learning_rate": [0.01, 0.03, 0.05],
                "depth": [4, 5, 6]
            },
            "Random Forest": {
                "n_estimators": [100, 200, 300],
                "max_depth": [None, 10, 20],
                "min_samples_leaf": [1, 5, 10]
            },
            "Ridge": {
                "alpha": [0.01, 0.1, 1.0, 10.0, 100.0]
            },
            "Lasso": {
                "alpha": [1e-4, 1e-3, 0.01, 0.1, 1.0]
            },
            "ElasticNet": {
                "alpha": [0.001, 0.01, 0.1, 1.0],
                "l1_ratio": [0.2, 0.5, 0.8]
            }
        }

        # Check if we have a grid for this model
        grid = None
        for key in param_grids:
            if key in self.name:
                grid = param_grids[key]
                break

        if not grid:
            cprint(f"       [INFO] No tuning grid for {self.name}. Skipping.", C.CYAN)
            return self

        cprint(f"       [TUNE] Tuning {self.name} ({n_iter} iterations)...", C.YELLOW)
        tscv = TimeSeriesSplit(n_splits=2)  # Use 2 splits for faster, stable validation
        
        # Use a copy of the estimator to avoid side effects during search
        from sklearn.base import clone
        base_est = clone(self.estimator)
        
        # Handle CatBoost/XGBoost/LightGBM verbose flags in grid search
        if "CatBoost" in self.name: base_est.set_params(verbose=0)
        if "XGBoost" in self.name: base_est.set_params(verbosity=0)
        if "LightGBM" in self.name: base_est.set_params(verbose=-1)

        search = RandomizedSearchCV(base_est, grid, n_iter=n_iter, cv=tscv, 
                                    scoring='neg_mean_absolute_error', n_jobs=1, random_state=42)
        
        try:
            search.fit(X, y)
            self.estimator = search.best_estimator_
            cprint(f"       [TUNE] Best Params: {search.best_params_}", C.GREEN)
        except Exception as e:
            cprint(f"       [WARN] Tuning failed for {self.name}: {e}. Using defaults.", C.YELLOW)
            
        return self

    def predict(self, steps: int) -> np.ndarray:
        """Iterative multi-step prediction using recursive strategy. Optimized for speed."""
        # Work in log-space if the model was trained that way
        extended = (
            self._series_tail_work.copy()
            if self._do_log
            else self._series_tail.copy()
        ).astype(float)
        preds = []

        # Pre-calculate frequency to avoid re-calculating inside loop
        freq = pd.infer_freq(extended.index) or "D"

        for _ in range(steps):
            # Only calculate the last row of features (incremental)
            feat_df = engineer_features(extended, lags=self.lags, last_only=True)

            if feat_df.empty:
                p = float(extended.iloc[-1])
            else:
                row = feat_df[self._feature_cols].values
                # Disinfect features before transformation/prediction
                row = np.nan_to_num(row.astype(float), nan=0.0, posinf=1e9, neginf=-1e9)
                
                if self.scale:
                    row = self._scaler.transform(row)
                
                p = float(self._model.predict(row)[0])

            # Atomic Clean: Clip prediction p before it's reused as a feature
            # In log-space, anything > 20 is already ~485M INR per day, which is physically impossible.
            if self._do_log:
                p = np.clip(np.nan_to_num(p, nan=0.0), -12, 18)
            else:
                p = np.clip(np.nan_to_num(p, nan=0.0), -1e6, 1e10)

            preds.append(p)

            # Append new prediction to extended series for next lag calculation
            nxt_date = extended.index[-1] + pd.tseries.frequencies.to_offset(freq)
            extended.loc[nxt_date] = p

        result = np.array(preds)
        # Atomic Clean before final transform
        result = np.nan_to_num(result, nan=0.0, posinf=18.0 if self._do_log else 1e9, 
                               neginf=-12.0 if self._do_log else -1e6)

        # Inverse log-transform and clamp negatives
        if self._do_log:
            result = np.expm1(result)
        
        # Final result sanitation
        return np.nan_to_num(np.maximum(result, 0), nan=0.0, posinf=1e12)

    def feature_importance(self) -> pd.Series | None:
        """Return feature importances if available."""
        if hasattr(self._model, "feature_importances_"):
            return pd.Series(self._model.feature_importances_,
                             index=self._feature_cols).sort_values(ascending=False)
        if hasattr(self._model, "coef_"):
            return pd.Series(np.abs(self._model.coef_),
                             index=self._feature_cols).sort_values(ascending=False)
        return None

    def save(self, path: str):
        joblib.dump(self, path)
        cprint(f"   Model saved  {path}", C.GREEN)

    @staticmethod
    def load(path: str):
        return joblib.load(path)


# 
# INDIVIDUAL MODEL FACTORIES
# 
def make_linear_regression(lags=14):
    return FeatureForecaster(LinearRegression(), "Linear Regression", lags)

def make_ridge(lags=14, alpha=1.0):
    return FeatureForecaster(Ridge(alpha=alpha), f"Ridge (={alpha})", lags)

def make_lasso(lags=14, alpha=0.01):
    return FeatureForecaster(Lasso(alpha=alpha, max_iter=5000), f"Lasso (={alpha})", lags)

def make_elasticnet(lags=14):
    return FeatureForecaster(ElasticNet(alpha=0.01, l1_ratio=0.5, max_iter=5000),
                              "ElasticNet", lags)

def make_decision_tree(lags=14, max_depth=8):
    return FeatureForecaster(DecisionTreeRegressor(max_depth=max_depth, random_state=42),
                              "Decision Tree", lags)

def make_random_forest(lags=14, n_est=300):
    return FeatureForecaster(RandomForestRegressor(
        n_estimators=n_est, max_depth=10, min_samples_leaf=5,
        random_state=42, n_jobs=1), "Random Forest", lags)

def make_gradient_boosting(lags=14):
    return FeatureForecaster(GradientBoostingRegressor(
        n_estimators=300, learning_rate=0.03, max_depth=4,
        subsample=0.8, min_samples_leaf=5, random_state=42),
        "Gradient Boosting", lags)

def make_extra_trees(lags=14):
    return FeatureForecaster(ExtraTreesRegressor(n_estimators=200, random_state=42, n_jobs=1),
                              "Extra Trees", lags)

def make_svr(lags=14):
    return FeatureForecaster(SVR(kernel="rbf", C=10, epsilon=0.05),
                              "SVR (RBF)", lags, scale=True, log_transform=False)

def make_knn(lags=14, k=5):
    return FeatureForecaster(KNeighborsRegressor(n_neighbors=k, weights="distance"),
                              f"KNN (k={k})", lags)

def make_bayesian_ridge(lags=14):
    return FeatureForecaster(BayesianRidge(), "Bayesian Ridge", lags)

def make_huber(lags=14):
    return FeatureForecaster(HuberRegressor(epsilon=1.35, max_iter=600, tol=1e-4),
                              "Huber Regressor", lags)

def make_mlp(lags=14):
    """
    ULTRA-STABLE MLP architecture for highly volatile retail data.
    Wraps current models in a native Pipeline to handle scaling and imprinting internally.
    """
    # Native sklearn Pipeline for maximum architecture-level stability
    pipe = Pipeline([
        ('impute', SimpleImputer(strategy='constant', fill_value=0)),
        ('scale',  RobustScaler()),
        ('mlp',    MLPRegressor(hidden_layer_sizes=(32, 16), activation="relu",
                                 solver="adam", learning_rate_init=0.001, alpha=0.1,
                                 max_iter=1000, tol=1e-4, random_state=42,
                                 early_stopping=True, validation_fraction=0.15,
                                 n_iter_no_change=20))
    ])
    
    # Passing scale=False because the pipeline handles it internally
    return FeatureForecaster(pipe, "MLP Neural Network", lags, scale=False, use_robust_scaler=True)


def make_hist_gradient_boosting(lags=14):
    return FeatureForecaster(HistGradientBoostingRegressor(
        max_iter=300, learning_rate=0.03, max_depth=5,
        min_samples_leaf=5, l2_regularization=0.1, random_state=42),
        "HistGradientBoosting", lags)

def make_xgboost(lags=14):
    if xgb is None: return None
    from xgboost import XGBRegressor
    return FeatureForecaster(XGBRegressor(
        n_estimators=500, learning_rate=0.02, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, min_child_weight=5,
        random_state=42, n_jobs=-1, verbosity=0),
        "XGBoost", lags)

def make_lightgbm(lags=14):
    if lgb is None: return None
    from lightgbm import LGBMRegressor
    return FeatureForecaster(LGBMRegressor(
        n_estimators=500, learning_rate=0.02, max_depth=5,
        subsample=0.8, colsample_bytree=0.8, min_child_samples=10,
        random_state=42, n_jobs=-1, verbose=-1),
        "LightGBM", lags)

def make_catboost(lags=14):
    if cb is None: return None
    from catboost import CatBoostRegressor
    return FeatureForecaster(CatBoostRegressor(
        iterations=800, learning_rate=0.03, depth=6,
        l2_leaf_reg=3, random_seed=42, verbose=0, bootstrap_type='Bayesian'),
        "CatBoost", lags)


# 
# 14. MANUAL AR MODEL (AutoRegressive)
# 
class ARModel(BaseForecaster):
    """Manual AutoRegressive(p) model via OLS."""
    name = "AR Model"
    def __init__(self, p: int = 14):
        self.p = p
        self._coef = None
        self._intercept = None
        self._last = None

    def fit(self, series: pd.Series):
        vals = series.values.astype(float)
        X, y = [], []
        for i in range(self.p, len(vals)):
            X.append(vals[i-self.p:i][::-1])
            y.append(vals[i])
        X, y = np.array(X), np.array(y)
        # OLS solution: beta = (X'X)^-1 X'y
        Xb = np.hstack([np.ones((len(X),1)), X])
        try:
            beta = np.linalg.lstsq(Xb, y)[0]
        except np.linalg.LinAlgError:
            beta = np.zeros(self.p + 1)
        self._intercept = beta[0]
        self._coef      = beta[1:]
        self._last      = vals[-self.p:].tolist()
        return self

    def predict(self, steps: int) -> np.ndarray:
        history = list(self._last)
        preds   = []
        for _ in range(steps):
            x   = np.array(history[-self.p:][::-1])
            val = self._intercept + np.dot(self._coef, x)
            val = max(float(val), 0)
            preds.append(val)
            history.append(val)
        return np.array(preds)


# 
# 15. HOLT-WINTERS EXPONENTIAL SMOOTHING
# 
class HoltWinters(BaseForecaster):
    """Triple Exponential Smoothing (Holt-Winters)  additive seasonality."""
    name = "Holt-Winters"
    def __init__(self, alpha=0.3, beta=0.1, gamma=0.1, season_len=7):
        self.alpha = alpha
        self.beta  = beta
        self.gamma = gamma
        self.m     = season_len
        self._L = self._T = self._S = None

    def fit(self, series: pd.Series):
        y = series.values.astype(float)
        m = self.m
        if len(y) < 2 * m:
            m = max(2, len(y) // 4)
            self.m = m

        # Initialise
        L = np.mean(y[:m])
        T = (np.mean(y[m:2*m]) - L) / m if len(y) >= 2*m else 0
        S = [y[i] - L for i in range(m)]

        for t in range(len(y)):
            s_t = S[t % m]
            L_new = self.alpha * (y[t] - s_t) + (1 - self.alpha) * (L + T)
            T_new = self.beta  * (L_new - L)  + (1 - self.beta)  * T
            S[t % m] = self.gamma * (y[t] - L) + (1 - self.gamma) * s_t
            L, T = L_new, T_new

        self._L, self._T, self._S, self._n = L, T, S, len(y)
        return self

    def predict(self, steps: int) -> np.ndarray:
        L, T, S, n, m = self._L, self._T, self._S, self._n, self.m
        return np.array([max(L + h*T + S[(n + h - 1) % m], 0)
                         for h in range(1, steps + 1)])


# 
# 16. ENSEMBLE (Voting / Stacking)
# 
class EnsembleForecaster(BaseForecaster):
    """
    Weighted average ensemble of multiple forecasters.
    Weights are determined by inverse RMSE on a validation split.
    """
    name = "Weighted Ensemble"

    def __init__(self, forecasters: list, val_ratio: float = 0.15):
        self.forecasters = forecasters
        self.val_ratio   = val_ratio
        self._weights    = None

    def fit(self, series: pd.Series):
        n = len(series)
        val_n = max(7, int(n * self.val_ratio))
        train_s, val_s = series.iloc[:-val_n], series.iloc[-val_n:]

        def _get_weight(fc, tr_s, v_s, v_n):
            try:
                fc.fit(tr_s)
                preds = fc.predict(v_n)
                rmse = np.sqrt(np.mean((v_s.values - preds[:v_n]) ** 2))
                return 1 / (rmse + 1e-9)
            except Exception:
                return 0.0

        weights = [_get_weight(fc, train_s, val_s, val_n) for fc in self.forecasters]

        total = sum(weights) or 1
        self._weights = [w / total for w in weights]

        def _refit(fc, s):
            try:
                fc.fit(s)
            except Exception:
                pass
            return fc

        self.forecasters = [_refit(fc, series) for fc in self.forecasters]
        return self

    def predict(self, steps: int) -> np.ndarray:
        agg = np.zeros(steps)
        for fc, w in zip(self.forecasters, self._weights):
            if w > 0:
                try:
                    preds = fc.predict(steps)
                    agg += w * preds[:steps]
                except Exception:
                    pass
        return np.maximum(agg, 0)


# 
# 17. STACKING ENSEMBLE (God Level)
# 
class StackingForecaster(BaseForecaster):
    """
    Advanced Stacking ensemble using a Ridge meta-learner.
    Uses TimeSeriesSplit to avoid data leakage during meta-training.
    """
    name = "Stacking Regressor"

    def __init__(self, base_models: list, meta_model=None):
        self.base_models = base_models
        self.meta_model  = meta_model or Ridge(alpha=1.0)
        self._is_fitted   = False

    def fit(self, series: pd.Series):
        from sklearn.model_selection import TimeSeriesSplit
        n = len(series)
        tscv = TimeSeriesSplit(n_splits=5)
        
        # 1. Generate OOF (Out-Of-Fold) predictions for meta-learner
        X_meta = []
        y_meta = []
        
        series_vals = series.values.astype(float)
        
        for train_idx, val_idx in tscv.split(series):
            train_s = series.iloc[train_idx]
            val_s   = series.iloc[val_idx]
            
            fold_preds = []
            for m in self.base_models:
                try:
                    m.fit(train_s)
                    p = m.predict(len(val_s))[:len(val_s)]
                    # Ensure predictions match the fold size
                    if len(p) < len(val_s):
                        p = np.pad(p, (0, len(val_s) - len(p)), mode='edge')
                    fold_preds.append(p)
                except Exception as e:
                    # Fallback to mean if model fails on small fold
                    fold_preds.append(np.full(len(val_idx), train_s.mean()))
            
            X_meta.append(np.column_stack(fold_preds))
            y_meta.append(series_vals[val_idx])

        X_meta = np.vstack(X_meta)
        y_meta = np.concatenate(y_meta)

        # 2. Train meta-learner
        self.meta_model.fit(X_meta, y_meta)

        # 3. Re-fit all base models on full series
        for m in self.base_models:
            m.fit(series)
        
        self._is_fitted = True
        return self

    def predict(self, steps: int) -> np.ndarray:
        base_preds = []
        for m in self.base_models:
            base_preds.append(m.predict(steps)[:steps])
        
        X_test = np.column_stack(base_preds)
        final_preds = self.meta_model.predict(X_test)
        return np.maximum(final_preds, 0)


# 
# MODEL REGISTRY  all models available for benchmarking
# 
def get_all_models(lags: int = 0, series: pd.Series = None) -> list:
    """Return all forecaster instances for full benchmark."""
    # God Level: Auto-detect lags if set to 0 and series is provided
    if lags == 0 and series is not None:
        lags = detect_optimal_lags(series)
        cprint(f"   Statistical Auto-Lag Detected: {lags} periods", C.CYAN)
    elif lags == 0:
        lags = 14
    return [m for m in [
        NaiveForecaster(window=7),
        SeasonalNaiveForecaster(season_len=lags),
        ARModel(p=lags),
        HoltWinters(alpha=0.3, beta=0.1, gamma=0.1, season_len=7),
        make_linear_regression(lags),
        DifferencedForecaster(make_linear_regression(lags)),
        make_ridge(lags),
        make_lasso(lags),
        make_elasticnet(lags),
        make_bayesian_ridge(lags),
        make_huber(lags),
        make_decision_tree(lags),
        make_knn(lags),
        make_svr(lags),
        make_extra_trees(lags),
        make_random_forest(lags),
        make_gradient_boosting(lags),
        make_hist_gradient_boosting(lags),
        make_xgboost(lags),
        make_lightgbm(lags),
        make_catboost(lags),
        make_mlp(lags),
    ] if m is not None]

def get_quick_models(lags: int = 0, series: pd.Series = None) -> list:
    """Faster subset for quick benchmarking."""
    # God Level: Auto-detect lags if set to 0 and series is provided
    if lags == 0 and series is not None:
        lags = detect_optimal_lags(series)
    elif lags == 0:
        lags = 14
    return [m for m in [
        NaiveForecaster(window=7),
        SeasonalNaiveForecaster(season_len=lags),
        ARModel(p=lags),
        HoltWinters(),
        make_linear_regression(lags),
        DifferencedForecaster(make_linear_regression(lags)),
        make_ridge(lags),
        make_random_forest(lags),
        make_gradient_boosting(lags),
        make_hist_gradient_boosting(lags),
        make_xgboost(lags),
        make_lightgbm(lags),
        make_catboost(lags),
        make_mlp(lags),
    ] if m is not None]

def get_best_model(series: pd.Series, lags: int = 14, verbose: bool = True) -> BaseForecaster:
    """
    Automatically selects the best model via cross-validation.
    Returns a fitted best model.
    """
    models  = get_quick_models(lags)
    results = []
    n = len(series)
    split = int(n * 0.8)
    train_s, test_s = series.iloc[:split], series.iloc[split:]

    for m in models:
        try:
            m.fit(train_s)
            preds = m.predict(len(test_s))
            rmse  = np.sqrt(np.mean((test_s.values - preds[:len(test_s)]) ** 2))
            results.append((rmse, m))
            if verbose:
                cprint(f"    {m.name:<25} RMSE: {rmse:,.2f}", C.CYAN)
        except Exception as e:
            if verbose:
                cprint(f"    {m.name:<25} FAILED: {e}", C.RED)

    results.sort(key=lambda x: x[0])
    best = results[0][1]
    best.fit(series)   # re-fit on full data
    cprint(f"\n   Best Model: {best.name} (RMSE: {results[0][0]:,.2f})\n", C.GREEN)
    return best
