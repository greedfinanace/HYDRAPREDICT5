"""
risk_engine.py — Centralized Risk Management Module
=====================================================
Implements 5 systemic directives for Sharpe > 1.0 and drawdown minimization:
  1. Dynamic Volatility Targeting (EWMA λ=0.94)
  2. Tail Risk Hedging (Synthetic OTM Put Basket via Black-Scholes)
  3. Trailing Drawdown Kill Switch (HWM -10% hard kill)
  4. Sharpe-Optimized LightGBM Objective (custom grad/hess)
  5. MVRK Kelly Sizing (Multivariate Volatility-Regulated Kelly)
"""

import numpy as np
import pandas as pd
from scipy.stats import norm


# =============================================================================
# 1. DYNAMIC VOLATILITY TARGETING
# =============================================================================
class DynamicVolatilityTargeter:
    """
    Scales portfolio exposure inversely to forecasted volatility.
    Uses EWMA (RiskMetrics λ=0.94) to forecast variance.
    When portfolio-level variance exceeds the 80th percentile of its
    historical distribution, mechanically deleverages toward cash.
    """

    def __init__(self, ewma_lambda=0.94, target_vol=0.15, vol_cap_percentile=80,
                 lookback_days=252):
        self.ewma_lambda = ewma_lambda
        self.target_vol = target_vol          # annualized target vol (15%)
        self.vol_cap_percentile = vol_cap_percentile
        self.lookback_days = lookback_days
        self._vol_history = []

    def forecast_ewma_vol(self, returns_series):
        """
        Compute EWMA volatility forecast from a series of daily returns.
        Returns annualized volatility.
        """
        if len(returns_series) < 20:
            return self.target_vol  # not enough data, assume target

        returns = returns_series.values if hasattr(returns_series, 'values') else np.array(returns_series)
        returns = returns[~np.isnan(returns)]

        if len(returns) < 20:
            return self.target_vol

        # EWMA variance (RiskMetrics)
        lam = self.ewma_lambda
        var_t = np.var(returns[-20:])  # seed with sample variance
        for r in returns[-self.lookback_days:]:
            var_t = lam * var_t + (1 - lam) * r ** 2

        annualized_vol = np.sqrt(var_t * 252)
        return max(annualized_vol, 0.01)  # floor at 1%

    def get_scalar(self, portfolio_returns, spy_returns):
        """
        Returns a multiplier [0.0, 1.5] that scales all position sizes.
        Macro volatility targeting: exposure = target_vol / realized_vol.
        Dynamically scales portfolio inversely to rolling EWMA vol.
        """
        current_vol = self.forecast_ewma_vol(portfolio_returns)
        spy_vol = self.forecast_ewma_vol(spy_returns)

        # Track vol history
        self._vol_history.append(current_vol)

        # Macro vol targeting: direct inverse scaling
        # When realized vol is low → lever up; when high → delever
        base_scalar = self.target_vol / current_vol
        base_scalar = np.clip(base_scalar, 0.10, 1.50)  # 10% floor, 150% cap

        # Safety overlay: if SPY vol is in danger zone (> 80th pctl), reduce
        if len(self._vol_history) > 60:
            vol_threshold = np.percentile(self._vol_history, self.vol_cap_percentile)
            if spy_vol > vol_threshold:
                excess_ratio = spy_vol / vol_threshold
                regime_penalty = max(1.0 / excess_ratio, 0.25)
                base_scalar *= regime_penalty

        return float(np.clip(base_scalar, 0.10, 1.50))

    def get_asset_weight_scalars(self, asset_returns_dict):
        """
        Given a dict of {ticker: returns_series}, return per-asset inverse-vol 
        weight scalars. Higher vol → lower weight.
        """
        vols = {}
        for ticker, returns in asset_returns_dict.items():
            vols[ticker] = self.forecast_ewma_vol(returns)

        if not vols:
            return {}

        # Inverse vol weighting
        inv_vols = {t: 1.0 / v for t, v in vols.items()}
        total_inv_vol = sum(inv_vols.values())

        scalars = {t: iv / total_inv_vol * len(vols) for t, iv in inv_vols.items()}
        return scalars


# =============================================================================
# 2. TAIL RISK HEDGER (Synthetic OTM Put Basket)
# =============================================================================
class TailRiskHedger:
    """
    Simulates a basket of deep OTM put options (δ ≈ -0.10, 6-12 month expiry).
    Budget: 0.5-1% of NAV annually for premium.
    During crash (SPY drawdown > threshold), puts go ITM → convex payoff.
    """

    def __init__(self, annual_budget_pct=0.0075, delta_target=-0.10,
                 days_to_expiry=180, risk_free_rate=0.04, initial_nav=1000000):
        self.annual_budget_pct = annual_budget_pct  # 0.75% midpoint
        self.delta_target = delta_target
        self.dte = days_to_expiry
        self.rf = risk_free_rate
        self.initial_nav = initial_nav  # anchor budget to initial capital
        self._active_puts = []
        self._total_premium_paid = 0.0
        self._total_payoff = 0.0
        self._last_roll_day = 0
        self._roll_interval = 63  # roll every quarter (63 trading days)
        self._day_counter = 0

    @staticmethod
    def _bs_put_price(S, K, T, r, sigma):
        """Black-Scholes put option price."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return max(K - S, 0)
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)
        put = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        return max(put, 0)

    @staticmethod
    def _bs_put_delta(S, K, T, r, sigma):
        """Black-Scholes put delta."""
        if T <= 0 or sigma <= 0 or S <= 0:
            return -1.0 if S < K else 0.0
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        return norm.cdf(d1) - 1.0

    def _find_strike_for_delta(self, S, T, sigma, target_delta=-0.10):
        """
        Binary search for strike price that gives target delta.
        For deep OTM puts, strike << spot.
        """
        lo, hi = S * 0.50, S * 1.0  # search range: 50% to 100% of spot
        for _ in range(50):  # binary search iterations
            mid = (lo + hi) / 2
            delta = self._bs_put_delta(S, mid, T, self.rf, sigma)
            if delta < target_delta:
                lo = mid
            else:
                hi = mid
        return (lo + hi) / 2

    def update(self, spy_price, spy_vol, nav, day_index, allow_new_hedges=True):
        """
        Daily update: mark-to-market active puts, roll when expired.
        Returns (net_hedge_value, premium_cost_today).
        """
        self._day_counter = day_index
        T_years = self.dte / 252.0
        premium_today = 0.0

        # Mark-to-market existing puts
        total_put_value = 0.0
        expired = []
        for put in self._active_puts:
            remaining_T = max((put['expiry_day'] - day_index) / 252.0, 0)
            if remaining_T <= 0:
                # Expired — compute final payoff
                payoff = max(put['strike'] - spy_price, 0) * put['notional_units']
                self._total_payoff += payoff
                total_put_value += payoff
                expired.append(put)
            else:
                # MTM via BS
                mtm = self._bs_put_price(spy_price, put['strike'], remaining_T,
                                         self.rf, spy_vol) * put['notional_units']
                total_put_value += mtm

        for p in expired:
            self._active_puts.remove(p)

        # Roll: buy new puts every roll_interval days
        should_roll = (day_index - self._last_roll_day) >= self._roll_interval or len(self._active_puts) == 0
        if should_roll and allow_new_hedges and nav > 0:
            # Budget for this roll — anchored to INITIAL nav to prevent spiral
            quarterly_budget = self.initial_nav * self.annual_budget_pct / 4.0
            
            strike = self._find_strike_for_delta(spy_price, T_years, spy_vol, self.delta_target)
            put_price = self._bs_put_price(spy_price, strike, T_years, self.rf, spy_vol)

            # Cap max theoretical leverage by enforcing a minimum typical option premium ($0.05 index options)
            put_price = max(put_price, 0.05)
            
            if put_price > 0:
                notional_units = quarterly_budget / put_price
                self._active_puts.append({
                    'strike': strike,
                    'expiry_day': day_index + self.dte,
                    'notional_units': notional_units,
                    'entry_price': put_price,
                    'premium_paid': quarterly_budget,
                })
                self._total_premium_paid += quarterly_budget
                premium_today = quarterly_budget
                self._last_roll_day = day_index

        return total_put_value, premium_today

    def get_cumulative_stats(self):
        return {
            'total_premium_paid': self._total_premium_paid,
            'total_payoff': self._total_payoff,
            'net_pnl': self._total_payoff - self._total_premium_paid,
            'active_puts': len(self._active_puts),
        }

    def reset(self):
        self._active_puts = []
        self._total_premium_paid = 0.0
        self._total_payoff = 0.0
        self._last_roll_day = 0
        self._day_counter = 0


# =============================================================================
# 3. TRAILING DRAWDOWN KILL SWITCH
# =============================================================================
class TrailingDrawdownKillSwitch:
    """
    Continuously tracks absolute peak equity (high-water mark).
    Hard kill at -10% from peak: flatten ALL positions to cash.
    Re-entry when equity recovers to within 5% of HWM (hysteresis).
    Soft scaling for 0-10% range to avoid cliff effects.
    """

    def __init__(self, kill_threshold=-0.10, recovery_threshold=-0.05):
        self.kill_threshold = kill_threshold      # -10% from HWM
        self.recovery_threshold = recovery_threshold  # -5% to re-enter
        self.high_water_mark = 0.0
        self.is_killed = False
        self._kill_count = 0
        self._kill_dates = []

    def update(self, current_equity, date=None):
        """
        Update HWM and check kill switch state.
        Returns (is_killed: bool, scalar: float 0.0-1.0).
        """
        # Update high water mark
        if current_equity > self.high_water_mark:
            self.high_water_mark = current_equity

        if self.high_water_mark <= 0:
            return False, 1.0

        drawdown = (current_equity - self.high_water_mark) / self.high_water_mark

        # State machine
        if self.is_killed:
            # Check for recovery (hysteresis)
            if drawdown > self.recovery_threshold:
                self.is_killed = False
                # After recovery, start cautious — 50% deployment
                return False, 0.50
            else:
                return True, 0.0  # stay killed

        else:
            # Check for kill trigger
            if drawdown <= self.kill_threshold:
                self.is_killed = True
                self._kill_count += 1
                if date is not None:
                    self._kill_dates.append(date)
                return True, 0.0  # KILL — flatten everything

            # Soft scaling for 0-10% range
            if drawdown > -0.03:
                return False, 1.0      # within 3%: full size
            elif drawdown > -0.06:
                return False, 0.70     # 3-6%: reduce
            elif drawdown > -0.08:
                return False, 0.40     # 6-8%: cautious
            else:
                return False, 0.20     # 8-10%: minimal (approaching kill)

    def get_stats(self):
        return {
            'kill_count': self._kill_count,
            'kill_dates': self._kill_dates,
            'high_water_mark': self.high_water_mark,
            'currently_killed': self.is_killed,
        }

    def reset(self, initial_capital):
        self.high_water_mark = initial_capital
        self.is_killed = False
        self._kill_count = 0
        self._kill_dates = []


# =============================================================================
# 4. SHARPE-OPTIMIZED LIGHTGBM OBJECTIVE (Band Turnover + Fixed Hessians)
# =============================================================================
class SharpeOptimizedObjective:
    """
    Custom LightGBM objective that directly maximizes a differentiable Sharpe
    approximation with band turnover regularization.

    Design goals:
      - Convert labels to signed targets {-1, +1} to avoid one-sided gradients
      - Adaptive gradient RMS scaling to keep LightGBM updates out of the flat regime
      - Band turnover penalty: no penalty within drift band, quadratic beyond it
      - Positive, bounded Hessian with conservative floor to keep tree growth viable
    """

    def __init__(self, turnover_lambda=0.05, band_threshold=0.02,
                 target_grad_rms=0.05, min_scale=1.0, max_scale=500.0,
                 hessian_floor=1e-3):
        self.turnover_lambda = turnover_lambda
        self.band_threshold = band_threshold  # free drift band (2%)
        self.target_grad_rms = target_grad_rms
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.hessian_floor = hessian_floor
        self.prev_preds = None

    def sharpe_objective(self, y_true, y_pred):
        """
        Custom objective for LightGBM (sklearn API).
        Signature: (y_true, y_pred) -> (grad, hess)
        """
        n = len(y_true)
        if n == 0:
            return np.array([], dtype=float), np.array([], dtype=float)

        # y_pred in LGBM objective is the raw score (before sigmoid)
        y_pred = np.clip(np.asarray(y_pred, dtype=float), -20, 20)
        y_signed = np.where(np.asarray(y_true) > 0, 1.0, -1.0)

        sigmoid_preds = 1.0 / (1.0 + np.exp(-y_pred))
        sigmoid_grad = sigmoid_preds * (1.0 - sigmoid_preds)

        # Exposure in [-1, +1]; signed returns reward correct directional confidence
        exposures = (2.0 * sigmoid_preds) - 1.0
        weighted_returns = exposures * y_signed
        mean_wr = np.mean(weighted_returns)
        centered_wr = weighted_returns - mean_wr
        std_wr = np.sqrt(np.mean(centered_wr ** 2) + 1e-8)

        # dSharpe/d(weighted_returns), then chain back to raw scores
        d_sharpe_d_wr = (1.0 / std_wr) - (mean_wr * centered_wr) / (std_wr ** 3)
        d_sharpe_d_wr /= float(n)
        d_wr_d_raw = y_signed * 2.0 * sigmoid_grad
        raw_grad = -(d_sharpe_d_wr * d_wr_d_raw)

        # Adaptive scaling to avoid near-zero gradients causing one-leaf trees
        grad_rms = np.sqrt(np.mean(raw_grad ** 2) + 1e-12)
        adaptive_scale = np.clip(
            self.target_grad_rms / max(grad_rms, 1e-12),
            self.min_scale,
            self.max_scale,
        )
        grad = raw_grad * adaptive_scale

        # Band turnover penalty (in exposure space)
        if self.prev_preds is not None and len(self.prev_preds) == n:
            prev_exposure = (2.0 * self.prev_preds) - 1.0
            drift = exposures - prev_exposure
            excess = np.maximum(np.abs(drift) - self.band_threshold, 0.0)
            turnover_grad = self.turnover_lambda * 2.0 * excess * np.sign(drift)
            grad += turnover_grad * (2.0 * sigmoid_grad)

        self.prev_preds = sigmoid_preds.copy()

        # Positive bounded Hessian proxy (scaled Fisher information)
        hess = (2.0 * sigmoid_grad) * (1.0 + 0.1 * np.abs(adaptive_scale))
        hess = np.clip(hess, self.hessian_floor, 10.0)

        # Safe guards: never return NaN/Inf to LightGBM
        if not np.all(np.isfinite(grad)):
            grad = np.where(np.isfinite(grad), grad, 0.0)
        if not np.all(np.isfinite(hess)):
            hess = np.where(np.isfinite(hess), hess, 1.0)

        return grad.astype(float), hess.astype(float)

    def sharpe_eval(self, y_true, y_pred):
        """
        Custom evaluation metric for LightGBM (sklearn API).
        Signature: (y_true, y_pred) -> (name, value, is_higher_better)
        """
        y_pred = np.clip(np.asarray(y_pred, dtype=float), -20, 20)
        y_signed = np.where(np.asarray(y_true) > 0, 1.0, -1.0)
        sigmoid_preds = 1.0 / (1.0 + np.exp(-y_pred))
        exposures = (2.0 * sigmoid_preds) - 1.0

        weighted_returns = exposures * y_signed
        mean_wr = np.mean(weighted_returns)
        std_wr = np.std(weighted_returns) + 1e-8
        sharpe = mean_wr / std_wr if std_wr > 0 else 0.0
        if not np.isfinite(sharpe):
            sharpe = 0.0
        return 'sharpe_proxy', sharpe, True  # name, value, is_higher_better

    @staticmethod
    def get_lambdarank_params():
        """
        Fallback LightGBM params for LambdaRank objective.
        Natively optimizes cross-sectional sorting without vanishing gradients.
        """
        return {
            'objective': 'lambdarank',
            'metric': 'ndcg',
            'ndcg_eval_at': [5, 10],
            'n_estimators': 300,
            'max_depth': 4,
            'learning_rate': 0.03,
            'subsample': 0.8,
            'min_child_samples': 5,
            'min_child_weight': 0.0,
            'min_split_gain': 0.0,
            'reg_alpha': 0.0,
            'reg_lambda': 0.0,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1,
        }

    def reset(self):
        self.prev_preds = None


# =============================================================================
# 5. MVRK KELLY SIZER (Multivariate Volatility-Regulated Kelly)
# =============================================================================
class MVRKKellySizer:
    """
    Implements the Multivariate Volatility-Regulated Kelly formula:
        f* = (Σ⁻¹ · μ) / (1 + κ · diag(Σ))
    
    Where:
        Σ = covariance matrix of asset returns
        μ = expected return vector
        κ = idiosyncratic risk penalty (default 1.0)
    
    Highly volatile assets are automatically constrained.
    """

    def __init__(self, kappa=1.0, max_position_pct=0.05, min_position_pct=0.005,
                 shrinkage_alpha=0.5, lookback=60, fractional_kelly=0.25):
        self.kappa = kappa                    # idiosyncratic risk penalty
        self.max_position_pct = max_position_pct  # 5% max per position
        self.min_position_pct = min_position_pct  # 0.5% min
        self.shrinkage_alpha = shrinkage_alpha    # Ledoit-Wolf style shrinkage
        self.lookback = lookback
        self.fractional_kelly = fractional_kelly  # fraction of full Kelly (0.25 = quarter)

    def _shrink_covariance(self, cov_matrix):
        """
        Apply Ledoit-Wolf shrinkage toward the diagonal (identity scaled).
        Stabilizes the inverse in small-sample regime.
        """
        n = cov_matrix.shape[0]
        target = np.diag(np.diag(cov_matrix))  # diagonal target
        shrunk = (1 - self.shrinkage_alpha) * cov_matrix + self.shrinkage_alpha * target
        return shrunk

    @staticmethod
    def _sanitize_returns(values, clip_abs=0.30):
        """
        Remove non-finite points and clip extreme values for stable covariance.
        """
        arr = np.asarray(values, dtype=float)
        if arr.size == 0:
            return np.array([], dtype=float)
        arr = arr[np.isfinite(arr)]
        if arr.size == 0:
            return np.array([], dtype=float)
        return np.clip(arr, -clip_abs, clip_abs)

    def optimal_allocation(self, expected_returns, returns_history, capital, prices):
        """
        Compute MVRK Kelly-optimal position sizes.

        Parameters:
        -----------
        expected_returns : dict {ticker: float}   — model's expected 5-day return
        returns_history  : dict {ticker: np.array} — last N daily returns per asset
        capital          : float                    — available capital
        prices           : dict {ticker: float}     — current prices

        Returns:
        --------
        allocations : list of dicts with 'ticker', 'shares', 'weight', 'cost'
        """
        tickers = sorted(expected_returns.keys())
        n = len(tickers)

        if n == 0:
            return []

        if n == 1:
            # Single asset: simple Kelly
            t = tickers[0]
            mu = expected_returns[t] if np.isfinite(expected_returns[t]) else 0.0
            ret = self._sanitize_returns(returns_history.get(t, np.array([])))
            if len(ret) < 10:
                var = 0.02 ** 2
            else:
                var = np.var(ret) + 1e-8
            kelly_f = mu / (var * (1 + self.kappa))
            kelly_f *= self.fractional_kelly  # Apply fractional Kelly
            kelly_f = np.clip(kelly_f, 0, self.max_position_pct)
            dollar_alloc = capital * kelly_f
            shares = int(dollar_alloc / prices[t]) if prices[t] > 0 else 0
            if shares > 0:
                return [{'ticker': t, 'shares': shares, 'weight': kelly_f,
                         'cost': shares * prices[t]}]
            return []

        # Build return matrix (N assets x T observations)
        clean_returns = {
            t: self._sanitize_returns(returns_history.get(t, np.array([])))
            for t in tickers
        }
        min_len = min(len(clean_returns[t]) for t in tickers)
        min_len = min(min_len, self.lookback)

        if min_len < 10:
            # Not enough history — fall back to equal weight with vol scaling
            return self._equal_vol_fallback(expected_returns, returns_history,
                                           capital, prices, tickers)

        # Build returns matrix
        R = np.column_stack([clean_returns[t][-min_len:] for t in tickers])
        if not np.all(np.isfinite(R)):
            R = np.nan_to_num(R, nan=0.0, posinf=0.0, neginf=0.0)

        # Expected return vector
        mu = np.array([
            expected_returns[t] if np.isfinite(expected_returns[t]) else 0.0
            for t in tickers
        ], dtype=float)

        # Covariance matrix with shrinkage
        cov = np.cov(R, rowvar=False)
        cov = np.nan_to_num(cov, nan=0.0, posinf=0.0, neginf=0.0)
        cov += np.eye(cov.shape[0]) * 1e-8
        cov_shrunk = self._shrink_covariance(cov)

        # MVRK Kelly formula: f* = (Σ⁻¹ · μ) / (1 + κ · diag(Σ))
        try:
            cov_inv = np.linalg.inv(cov_shrunk)
        except np.linalg.LinAlgError:
            cov_inv = np.linalg.pinv(cov_shrunk)

        raw_kelly = cov_inv @ mu
        idio_penalty = 1.0 + self.kappa * np.diag(cov_shrunk)
        f_star = raw_kelly / idio_penalty

        # Apply fractional Kelly to limit concentration from uncalibrated ML signals
        f_star *= self.fractional_kelly

        # Clip to [0, max_position] — we only go long
        f_star = np.clip(f_star, 0, self.max_position_pct)

        # Normalize if total allocation exceeds 100%
        total_alloc = np.sum(f_star)
        if total_alloc > 1.0:
            f_star = f_star / total_alloc

        # Convert to shares
        allocations = []
        for i, t in enumerate(tickers):
            weight = f_star[i]
            if weight < self.min_position_pct:
                continue  # skip tiny allocations

            dollar_alloc = capital * weight
            if prices[t] <= 0:
                continue
            shares = int(dollar_alloc / prices[t])
            if shares > 0:
                allocations.append({
                    'ticker': t,
                    'shares': shares,
                    'weight': float(weight),
                    'cost': shares * prices[t],
                })

        return allocations

    def _equal_vol_fallback(self, expected_returns, returns_history,
                            capital, prices, tickers):
        """Fallback: inverse-vol equal weight when not enough covariance data."""
        vols = {}
        for t in tickers:
            ret = self._sanitize_returns(returns_history.get(t, np.array([])))
            if len(ret) < 5:
                vols[t] = 0.03  # default vol
            else:
                vols[t] = np.std(ret) * np.sqrt(252) + 1e-8

        inv_vols = {t: 1.0 / v for t, v in vols.items()}
        total_iv = sum(inv_vols.values())

        # Only allocate to assets with positive expected return
        allocations = []
        for t in tickers:
            if expected_returns[t] <= 0:
                continue
            weight = (inv_vols[t] / total_iv) * min(len(tickers) * self.max_position_pct, 0.8)
            weight = min(weight, self.max_position_pct)
            dollar_alloc = capital * weight
            if prices[t] <= 0:
                continue
            shares = int(dollar_alloc / prices[t])
            if shares > 0:
                allocations.append({
                    'ticker': t,
                    'shares': shares,
                    'weight': float(weight),
                    'cost': shares * prices[t],
                })
        return allocations
