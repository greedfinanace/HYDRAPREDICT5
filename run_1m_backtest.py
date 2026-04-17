import pandas as pd
import numpy as np
import warnings
import logging
import yfinance as yf
warnings.simplefilter("default")
logger = logging.getLogger(__name__)

from ml_pipeline import (fetch_universe_data, build_all_features,
                         make_cross_sectional_labels, add_cross_sectional_ranks,
                         UNIVERSE, ALL_ASSETS, build_earnings_exclusion_mask)
from risk_engine import (DynamicVolatilityTargeter, TailRiskHedger,
                         TrailingDrawdownKillSwitch, SharpeOptimizedObjective,
                         MVRKKellySizer)
from lightgbm import LGBMClassifier, LGBMRegressor
from sklearn.isotonic import IsotonicRegression
from sklearn.preprocessing import StandardScaler


FLAT_PRED_STD_THRESHOLD = 1e-4
MAX_RETURN_ABS_CLIP = 0.30
PROB_FLOOR = 0.30
VIX_ENTRY_CUTOFF = 28.0
HEDGE_CHEAP_VIX_THRESHOLD = 18.0
NEAR_HWM_THRESHOLD = 0.95
HARD_STOP_PCT = 0.015
TRAILING_ACTIVATE_PCT = 0.01
TRAILING_RETAIN_GAIN = 0.50
TIME_STOP_DAYS = 3
CALIBRATION_FRACTION = 0.20
MIN_CALIBRATION_SAMPLES = 80
CLASS_TREND_REPRESENTATIVES = {
    "equities": "SPY",
    "sectors": "SPY",
    "international": "EFA",
    "fixed_income": "IEF",
    "commodities": "PDBC",
    "reits": "VNQ",
}


# â”€â”€â”€ Dynamic ATR Stop (kept from prior, integrates with vol targeter) â”€â”€â”€â”€â”€â”€â”€â”€
def dynamic_atr_multiple(vix_level):
    """Adjust stop distance based on current market volatility."""
    if pd.isna(vix_level):
        return 2.0
    if vix_level < 15:   return 1.5
    elif vix_level < 20: return 2.0
    elif vix_level < 28: return 2.5
    else:                return 3.0


def _extract_close_series(data, spy, ticker):
    if ticker == "SPY":
        df = spy.copy()
    else:
        if not isinstance(data.columns, pd.MultiIndex) or ticker not in data.columns.levels[0]:
            return pd.Series(dtype=float)
        df = data[ticker].copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(0)
    if "Close" not in df.columns:
        return pd.Series(dtype=float)

    close = pd.to_numeric(df["Close"], errors="coerce").dropna()
    close.index = pd.DatetimeIndex(close.index)
    if close.index.tz is not None:
        close.index = close.index.tz_localize(None)
    return close


def _build_class_trend_filters(data, spy, target_dates):
    target_index = pd.DatetimeIndex(pd.to_datetime(list(target_dates)))
    filters = {}
    for asset_class, proxy in CLASS_TREND_REPRESENTATIVES.items():
        close = _extract_close_series(data, spy, proxy)
        if close.empty:
            filters[asset_class] = pd.Series(True, index=target_index)
            continue

        sma50 = close.rolling(50).mean()
        sma200 = close.rolling(200).mean()
        trend_ok = ((sma50 > sma200) & (close > sma200)).reindex(target_index).ffill().fillna(True)
        filters[asset_class] = trend_ok.astype(bool)
    return filters


def _regime_allows_long(asset_class, date, class_trend_filters):
    trend_series = class_trend_filters.get(asset_class)
    if trend_series is None or trend_series.empty:
        return True
    return bool(trend_series.get(date, True))


def _should_buy_tail_hedge(vix_level, current_equity, high_water_mark):
    cheap_protection = pd.notna(vix_level) and float(vix_level) < HEDGE_CHEAP_VIX_THRESHOLD
    near_high_water = high_water_mark <= 0 or current_equity >= (high_water_mark * NEAR_HWM_THRESHOLD)
    return cheap_protection or near_high_water


def _liquidate_positions(active_positions, current_day_data, date, all_trades_log, reason):
    released_cash = 0.0
    win_count = 0
    trade_count = 0

    for pos in active_positions:
        if pos["ticker"] in current_day_data.index:
            exit_price = float(current_day_data.loc[pos["ticker"], "close"])
        else:
            exit_price = pos["entry_price"]

        profit = pos["shares"] * (exit_price - pos["entry_price"])
        released_cash += pos["cost"] + profit
        trade_count += 1
        if profit > 0:
            win_count += 1

        all_trades_log.append({
            "date": date,
            "ticker": pos["ticker"],
            "return_pct": (exit_price - pos["entry_price"]) / pos["entry_price"],
            "profit": profit,
            "exit_reason": reason,
        })

    return released_cash, win_count, trade_count


def _split_train_calibration(X, y):
    n = len(y)
    calibration_size = max(MIN_CALIBRATION_SAMPLES, int(n * CALIBRATION_FRACTION))
    calibration_size = min(calibration_size, max(1, n // 3))
    if (n - calibration_size) < 50:
        calibration_size = max(20, n // 5)
    if calibration_size <= 0 or calibration_size >= n:
        return X, y, None, None
    return X[:-calibration_size], y[:-calibration_size], X[-calibration_size:], y[-calibration_size:]


def _fit_isotonic_calibrator(uncalibrated_probs, y_true):
    if uncalibrated_probs is None or y_true is None:
        return None, {}

    probs = np.asarray(uncalibrated_probs, dtype=float)
    labels = np.asarray(y_true, dtype=float)
    mask = np.isfinite(probs) & np.isfinite(labels)
    probs = probs[mask]
    labels = labels[mask]

    if len(probs) < 20 or len(np.unique(labels)) < 2 or len(np.unique(probs)) < 2:
        return None, {}

    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(probs, labels)
    calibrated = np.clip(calibrator.predict(probs), 0.0, 1.0)
    brier = float(np.mean((calibrated - labels) ** 2))
    ece = _expected_calibration_error(labels, calibrated)
    return calibrator, {"brier": brier, "ece": ece}


def _expected_calibration_error(y_true, probs, bins=10):
    y_true = np.asarray(y_true, dtype=float)
    probs = np.asarray(probs, dtype=float)
    if len(y_true) == 0:
        return 0.0

    edges = np.linspace(0.0, 1.0, bins + 1)
    ece = 0.0
    for left, right in zip(edges[:-1], edges[1:]):
        if right == 1.0:
            mask = (probs >= left) & (probs <= right)
        else:
            mask = (probs >= left) & (probs < right)
        if not np.any(mask):
            continue
        avg_prob = float(np.mean(probs[mask]))
        avg_label = float(np.mean(y_true[mask]))
        ece += (np.sum(mask) / len(y_true)) * abs(avg_prob - avg_label)
    return float(ece)


def _format_calibration_stats(stats):
    if not stats:
        return "uncalibrated"
    return f"brier={stats['brier']:.3f}, ece={stats['ece']:.3f}"


# â”€â”€â”€ MVRK-Kelly Enhanced Trade Selection â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def _select_top_candidates(signals_df, picks_per_class=2):
    selected = []
    for asset_class, tickers in UNIVERSE.items():
        class_signals = signals_df[signals_df['ticker'].isin(tickers)].copy()
        if class_signals.empty:
            continue

        class_signals = class_signals[class_signals['prob_score'] >= PROB_FLOOR].copy()
        if class_signals.empty:
            continue

        class_signals['pct_rank'] = class_signals['prob_score'].rank(pct=True, method='first')
        top = class_signals[class_signals['pct_rank'] >= 0.7].copy()
        top = top.nlargest(picks_per_class, 'pct_rank')
        if not top.empty:
            selected.append(top)

    if not selected:
        return pd.DataFrame()
    return pd.concat(selected, ignore_index=True)


def _fallback_allocations(candidates_df, effective_capital, prices, mvrk_sizer):
    if candidates_df.empty or effective_capital <= 0:
        return []

    top = candidates_df.sort_values('pct_rank', ascending=False).copy()
    max_positions = min(len(top), 8)
    if max_positions == 0:
        return []

    per_weight = min(mvrk_sizer.max_position_pct, 1.0 / max_positions)
    allocations = []
    for row in top.head(max_positions).itertuples(index=False):
        ticker = row.ticker
        price = prices.get(ticker, np.nan)
        if not np.isfinite(price) or price <= 0:
            continue
        allocations.append({
            'ticker': ticker,
            'shares': int((effective_capital * per_weight) / price),
            'weight': per_weight,
            'cost': 0.0,
        })
    return allocations


def select_trades_mvrk(signals_df, capital, regime_scalar, kill_scalar,
                       returns_history, mvrk_sizer, picks_per_class=2):
    """
    Rank within each asset class â†’ take top percentile-ranked assets â†’ size via MVRK Kelly.
    Uses cross-sectional percentile ranking instead of Z-scores to guarantee
    selections even when prediction variance is near zero.
    """
    final = _select_top_candidates(signals_df, picks_per_class=picks_per_class)
    if final.empty:
        return []
    
    # Build inputs for MVRK Kelly
    expected_returns = {}
    prices = {}
    row_lookup = {}
    for row in final.itertuples(index=False):
        t = row.ticker
        # Convert percentile rank to expected return proxy
        # pct_rank is [0.7, 1.0] for selected assets; scale to a meaningful signal
        rank_signal = (row.pct_rank - 0.5) * 2.0  # maps 0.7->0.4, 1.0->1.0
        expected_returns[t] = rank_signal * 0.01 * regime_scalar * kill_scalar
        prices[t] = row.price
        row_lookup[t] = row

    # Effective capital with scalars applied
    effective_capital = capital * regime_scalar * kill_scalar

    if effective_capital <= 0:
        return []

    # Get MVRK Kelly allocations
    allocations = mvrk_sizer.optimal_allocation(
        expected_returns, returns_history, effective_capital, prices
    )
    if not allocations:
        allocations = _fallback_allocations(final, effective_capital, prices, mvrk_sizer)

    # Enrich with trade metadata
    trades = []
    for alloc in allocations:
        ticker = alloc['ticker']
        row = row_lookup.get(ticker)
        if row is None:
            continue

        stop_dist = prices[ticker] * HARD_STOP_PCT

        if stop_dist <= 0:
            continue

        # Standard Risk Management: Risk 2% of Effective Capital on this trade
        # Risk = (Entry - Stop) * Shares -> Shares = Risk / (Entry - Stop)
        risk_per_trade = effective_capital * 0.02
        shares = int(risk_per_trade / stop_dist)
        
        # Sizing Cap: Never exceed the MVRK-optimal weight
        # Max Shares = Optimal Weight * Capital / Price
        max_shares = int((alloc['weight'] * effective_capital) / prices[ticker])
        
        shares = min(shares, max_shares)

        if shares > 0:
            trades.append({
                'ticker':       ticker,
                'asset_class':  getattr(row, 'asset_class', 'Unknown'),
                'shares':       shares,
                'stop_price':   prices[ticker] - stop_dist,
                'prob':         row.prob_score,
                'pct_rank':     row.pct_rank,
                'cost':         shares * prices[ticker],
                'entry_price':  prices[ticker],
                'kelly_weight': alloc['weight'],
                'peak_return':  0.0,
            })
    return trades


# â”€â”€â”€ Main Backtest â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
def run_universe_backtest(start_year=2010, tail_hedge_annual_budget_pct=0.0075):
    print("Fetching Universe Data...")
    data, spy, tlt, usd, btc = fetch_universe_data(start=f"{start_year}-01-01")
    
    print("Building Earnings Exclusion Mask...")
    exclusions = build_earnings_exclusion_mask(ALL_ASSETS)

    print("Building Features...")
    df = build_all_features(data, spy, tlt, usd, btc)
    
    # Feature columns â€” includes new risk features
    raw_features = [
        'rsi_14', 'mom_5d', 'mom_20d', 'vol_ratio', 'atr_pct', 
        'rs_vs_spy_5d', 'rs_vs_spy_20d', 'bb_pct_b', 'gap',
        'vol_price_divergence', 'pct_from_52w_high',
        'spy_regime', 'yield_momentum', 'flight_to_safety', 
        'usd_5d', 'inflation_regime',
        'forecasted_vol', 'signal_stability',
        # Orthogonal cross-asset features
        'hyg_spread_5d', 'hyg_spread_20d',
        'btc_mom_5d', 'btc_mom_20d',
        'hmm_regime',
    ]
    
    print("Adding Cross-Sectional Ranks...")
    df = add_cross_sectional_ranks(df, raw_features)
    
    print("Creating Labels...")
    labels = make_cross_sectional_labels(df)
    df['target'] = labels
    
    feature_cols = raw_features + [f'{c}_xrank' for c in raw_features]
    df = df.dropna(subset=feature_cols + ['target', 'atr_abs', 'close'])
    
    # VIX for regime
    vix = yf.download("^VIX", start=f"{start_year}-01-01", progress=False)
    if isinstance(vix.columns, pd.MultiIndex):
        vix = vix.droplevel(1, axis=1)
        
    dates = df.index.get_level_values(0)
    df['Year'] = dates.year
    years = sorted(df['Year'].unique())
    class_trend_filters = _build_class_trend_filters(data, spy, sorted(df.index.get_level_values(0).unique()))
    
    if len(years) < 2:
        print("Not enough historical data to run the rolling backtest.")
        return

    train_window_years = min(4, len(years) - 1)
    initial_capital = 1000000
    
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # INITIALIZE ALL 5 RISK ENGINE COMPONENTS
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    vol_targeter   = DynamicVolatilityTargeter(ewma_lambda=0.94, target_vol=0.15)
    tail_hedger    = TailRiskHedger(annual_budget_pct=tail_hedge_annual_budget_pct, delta_target=-0.10, days_to_expiry=180, initial_nav=initial_capital)
    kill_switch    = TrailingDrawdownKillSwitch(kill_threshold=-0.10, recovery_threshold=-0.05)
    sharpe_obj     = SharpeOptimizedObjective(turnover_lambda=0.05, band_threshold=0.02)
    mvrk_sizer     = MVRKKellySizer(kappa=1.0, max_position_pct=0.05, shrinkage_alpha=0.5, fractional_kelly=0.25)

    kill_switch.reset(initial_capital)
    
    # Track returns history for MVRK covariance estimation
    asset_returns_history = {t: [] for t in ALL_ASSETS}
    asset_last_close = {}
    portfolio_returns_buffer = []
    spy_returns_buffer = []
    spy_last_close = None
    
    all_trades_log = []
    daily_portfolio_value = []
    
    print(f"\nRunning Rolling Backtest ({train_window_years}-Year Train -> 1-Year Test)...")
    print("Model: LightGBM (Sharpe-Optimized Objective + Turnover Penalty)")
    print(f"Risk Engine: DynVol Targeting | Tail Hedge ({tail_hedge_annual_budget_pct*100:.2f}% annual budget) | Kill Switch | MVRK Kelly")
    print(f"{'Test Year':<9} | {'Trades':<8} | {'Win Rate':<10} | {'Return':<8} | {'Kill Events'}")
    print("-" * 75)
    
    for i in range(len(years) - train_window_years):
        train_years = years[i:i+train_window_years]
        test_year = years[i+train_window_years]
        
        train_df = df[df['Year'].isin(train_years)].copy()
        test_df = df[df['Year'] == test_year].copy()
        
        if len(train_df) < 100 or len(test_df) < 10:
            continue
        
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        # DIRECTIVE 4: Train with Sharpe-optimized objective per asset class
        # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
        models = {}
        model_path_summary = {}
        sharpe_obj.reset()
        
        for asset_class, tickers in UNIVERSE.items():
            class_train_df = train_df[train_df['asset_class'] == asset_class].copy()
            if len(class_train_df) < 50:
                continue
                
            scaler = StandardScaler()
            X_train = scaler.fit_transform(class_train_df[feature_cols])
            y_train = class_train_df['target'].values
            
            if len(np.unique(y_train)) < 2:
                continue
            
            # Use custom Sharpe objective with LGBMRegressor (unblocked hyperparams)
            try:
                X_fit, y_fit, X_cal, y_cal = _split_train_calibration(X_train, y_train)
                base_model = LGBMRegressor(
                    n_estimators=300,
                    max_depth=4,
                    learning_rate=0.03,
                    subsample=0.8,
                    reg_alpha=0.0,          # Disabled â€” was killing tiny leaf values
                    reg_lambda=0.0,         # Disabled â€” was killing tiny leaf values  
                    min_child_samples=5,    # Allow small leaves (was default 20)
                    min_child_weight=0.0,   # = min_sum_hessian_in_leaf (was 1e-3)
                    min_split_gain=0.0,     # Allow even tiny split improvements
                    random_state=42,
                    n_jobs=-1,
                    verbose=-1,
                    objective=sharpe_obj.sharpe_objective,
                )
                base_model.fit(
                    X_fit, y_fit,
                    eval_set=[(X_fit, y_fit)],
                    eval_metric=sharpe_obj.sharpe_eval,
                    callbacks=[],
                )
                
                # Verify model actually learned (check prediction variance)
                train_preds = base_model.predict(X_fit)
                pred_std = float(np.std(train_preds))
                if (not np.isfinite(pred_std)) or pred_std < FLAT_PRED_STD_THRESHOLD:
                    raise ValueError(f"Sharpe objective produced flat predictions (std={pred_std:.2e})")

                cal_input = 1.0 / (1.0 + np.exp(-base_model.predict(X_cal))) if X_cal is not None else None
                calibrator, cal_stats = _fit_isotonic_calibrator(cal_input, y_cal)

                models[asset_class] = {
                    'model': base_model,
                    'scaler': scaler,
                    'calibrator': calibrator,
                    'kind': 'regressor',
                }
                model_path_summary[asset_class] = (
                    f"SHARPE(std={pred_std:.2e}, {_format_calibration_stats(cal_stats)})"
                )
            except Exception as e:
                # Fallback to Standard Binary Classifier (natively robust)
                try:
                    X_fit, y_fit, X_cal, y_cal = _split_train_calibration(X_train, y_train)
                    fallback = LGBMClassifier(
                        n_estimators=100, max_depth=4, learning_rate=0.05,
                        subsample=0.8, importance_type='gain',
                        random_state=42, n_jobs=-1, verbose=-1,
                        objective='binary'
                    )
                    fallback.fit(X_fit, y_fit)
                    if hasattr(fallback, 'predict_proba') and 1 in fallback.classes_:
                        idx = list(fallback.classes_).index(1)
                        fb_probs = fallback.predict_proba(X_fit)[:, idx]
                        fb_std = float(np.std(fb_probs))
                        cal_input = fallback.predict_proba(X_cal)[:, idx] if X_cal is not None else None
                    else:
                        fb_std = 0.0
                        cal_input = None
                    calibrator, cal_stats = _fit_isotonic_calibrator(cal_input, y_cal)
                    models[asset_class] = {
                        'model': fallback,
                        'scaler': scaler,
                        'calibrator': calibrator,
                        'kind': 'classifier',
                    }
                    reason = str(e).replace('\n', ' ')[:80]
                    model_path_summary[asset_class] = (
                        f"FALLBACK(std={fb_std:.2e}, {_format_calibration_stats(cal_stats)}, reason={reason})"
                    )
                except Exception:
                    continue

        if model_path_summary:
            path_msg = "; ".join(
                f"{ac}:{msg}" for ac, msg in sorted(model_path_summary.items())
            )
            print(f"  [{test_year}] Model path -> {path_msg}")
        
        # Generate calibrated probabilities
        test_df['prob_score'] = 0.0
        for asset_class, model_info in models.items():
            model = model_info['model']
            scaler = model_info['scaler']
            calibrator = model_info['calibrator']
            model_kind = model_info['kind']
            class_test_mask = test_df['asset_class'] == asset_class
            if not class_test_mask.any():
                continue
            
            X_test_class = scaler.transform(test_df.loc[class_test_mask, feature_cols])

            if model_kind == 'regressor':
                raw_preds = model.predict(X_test_class)
                base_probs = 1.0 / (1.0 + np.exp(-raw_preds))
            elif hasattr(model, 'predict_proba') and 1 in model.classes_:
                idx = list(model.classes_).index(1)
                base_probs = model.predict_proba(X_test_class)[:, idx]
            else:
                continue

            if calibrator is not None:
                prob_scores = np.clip(calibrator.predict(base_probs), 0.0, 1.0)
            else:
                prob_scores = np.clip(base_probs, 0.0, 1.0)

            test_df.loc[class_test_mask, 'prob_score'] = prob_scores
            continue
            
            
        test_dates = sorted(test_df.index.get_level_values(0).unique())
        
        df_dates = pd.DataFrame(index=test_dates)
        df_dates['week'] = df_dates.index.isocalendar().week
        df_dates['year'] = df_dates.index.isocalendar().year
        weekly_starts = df_dates.groupby(['year', 'week']).head(1).index.tolist()
        
        capital = initial_capital if i == 0 else daily_portfolio_value[-1]['capital']
        year_trades = 0
        winning_trades = 0
        year_start_capital = capital
        year_kill_events = 0
        
        active_positions = [] 
        day_idx_offset = len(daily_portfolio_value)
        
        for d_idx, date in enumerate(test_dates):
            current_day_data = test_df.xs(date, level=0)
            dt_no_tz = pd.to_datetime(date).tz_localize(None)
            
            # Get VIX and SPY for today
            try:
                current_vix = vix.loc[:dt_no_tz]['Close'].iloc[-1]
            except Exception:
                current_vix = 20
            try:
                current_spy = spy.loc[:dt_no_tz]['Close'].iloc[-1]
            except Exception:
                current_spy = 100
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # DIRECTIVE 3: CHECK KILL SWITCH FIRST (before ANY trade logic)
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            mtm_invested = 0
            for pos in active_positions:
                if pos['ticker'] in current_day_data.index:
                    mtm_invested += pos['shares'] * current_day_data.loc[pos['ticker'], 'close']
                else:
                    mtm_invested += pos['cost']
            
            current_equity = capital + mtm_invested
            is_killed, kill_scalar = kill_switch.update(current_equity, date)
            
            if is_killed:
                # â•â•â•â•â•â•â•â•â•â• KILL SWITCH ACTIVATED â•â•â•â•â•â•â•â•â•â•
                # Flatten ALL positions to cash immediately
                released_cash, win_count, closed_count = _liquidate_positions(
                    active_positions, current_day_data, date, all_trades_log, "KILL_SWITCH"
                )
                capital += released_cash
                winning_trades += win_count
                year_trades += closed_count

                active_positions = []
                year_kill_events += 1
                
                # Still update tail hedge MTM even when killed
                spy_vol = current_vix / 100.0 if current_vix else 0.20
                allow_new_hedges = _should_buy_tail_hedge(
                    current_vix, capital, kill_switch.high_water_mark
                )
                hedge_val, premium = tail_hedger.update(
                    current_spy, spy_vol, capital, day_idx_offset + d_idx,
                    allow_new_hedges=allow_new_hedges
                )
                capital -= premium
                
                daily_portfolio_value.append({
                    'date': date, 'capital': capital + hedge_val,
                    'status': 'KILLED'
                })
                
                # Update returns buffer
                if len(daily_portfolio_value) >= 2:
                    prev_cap = daily_portfolio_value[-2]['capital']
                    if prev_cap > 0:
                        portfolio_returns_buffer.append(
                            (daily_portfolio_value[-1]['capital'] / prev_cap) - 1
                        )
                
                continue  # Skip all trading logic

            allow_new_entries = True
            if current_vix > VIX_ENTRY_CUTOFF:
                released_cash, win_count, closed_count = _liquidate_positions(
                    active_positions, current_day_data, date, all_trades_log, "VIX_GATE"
                )
                capital += released_cash
                winning_trades += win_count
                year_trades += closed_count
                active_positions = []
                allow_new_entries = False
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # UPDATE ASSET RETURNS HISTORY (for MVRK covariance)
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            for ticker_name in current_day_data.index:
                if ticker_name in asset_returns_history:
                    close_val = float(current_day_data.loc[ticker_name, 'close'])
                    if not np.isfinite(close_val) or close_val <= 0:
                        continue
                    prev_close = asset_last_close.get(ticker_name)
                    if prev_close is not None and prev_close > 0:
                        day_ret = (close_val / prev_close) - 1.0
                        if np.isfinite(day_ret):
                            asset_returns_history[ticker_name].append(
                                float(np.clip(day_ret, -MAX_RETURN_ABS_CLIP, MAX_RETURN_ABS_CLIP))
                            )
                            if len(asset_returns_history[ticker_name]) > 252:
                                asset_returns_history[ticker_name] = asset_returns_history[ticker_name][-252:]
                    asset_last_close[ticker_name] = close_val
            
            # Track SPY returns for vol targeter
            try:
                spy_close_today = float(spy.loc[:dt_no_tz]['Close'].iloc[-1])
                if np.isfinite(spy_close_today) and spy_close_today > 0:
                    if spy_last_close is not None and spy_last_close > 0:
                        spy_ret = (spy_close_today / spy_last_close) - 1.0
                        if np.isfinite(spy_ret):
                            spy_returns_buffer.append(
                                float(np.clip(spy_ret, -MAX_RETURN_ABS_CLIP, MAX_RETURN_ABS_CLIP))
                            )
                            if len(spy_returns_buffer) > 252:
                                spy_returns_buffer = spy_returns_buffer[-252:]
                    spy_last_close = spy_close_today
            except Exception:
                logger.debug("SPY return buffer update failed for %s", date, exc_info=True)
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # 1. UPDATE PRICES & CHECK STOPS/EXITS
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            positions_to_close = []
            for pos in active_positions:
                pos['days_held'] += 1
                ticker = pos['ticker']
                
                if ticker in current_day_data.index:
                    current_price = float(current_day_data.loc[ticker, 'close'])
                    current_return = (current_price / pos['entry_price']) - 1.0
                    pos['peak_return'] = max(pos.get('peak_return', 0.0), current_return)

                    hard_stop_hit = current_return <= -HARD_STOP_PCT
                    trail_hit = (
                        pos['peak_return'] >= TRAILING_ACTIVATE_PCT and
                        current_return <= (pos['peak_return'] * TRAILING_RETAIN_GAIN)
                    )
                    time_stop_hit = pos['days_held'] >= TIME_STOP_DAYS and current_return <= 0.0

                    if hard_stop_hit:
                        positions_to_close.append((pos, current_price, 'HARD_STOP'))
                    elif trail_hit:
                        positions_to_close.append((pos, current_price, 'TRAIL_STOP'))
                    elif time_stop_hit:
                        positions_to_close.append((pos, current_price, 'TIME_STOP'))
            
            # Close positions
            for pos, exit_price, exit_reason in positions_to_close:
                return_pct = (exit_price - pos['entry_price']) / pos['entry_price']
                profit = pos['shares'] * (exit_price - pos['entry_price'])
                capital += pos['cost'] + profit
                
                all_trades_log.append({
                    'date': date, 'ticker': pos['ticker'],
                    'return_pct': return_pct, 'profit': profit,
                    'exit_reason': exit_reason
                })
                
                if profit > 0:
                    winning_trades += 1
                year_trades += 1
                active_positions.remove(pos)
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # 2. OPEN NEW POSITIONS (weekly rebalance)
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            if allow_new_entries and date in weekly_starts:
                # DIRECTIVE 1: Dynamic Volatility Targeting scalar
                spy_ret_series = pd.Series(spy_returns_buffer[-252:]) if len(spy_returns_buffer) > 20 else pd.Series([0.0] * 20)
                port_ret_series = pd.Series(portfolio_returns_buffer[-252:]) if len(portfolio_returns_buffer) > 20 else pd.Series([0.0] * 20)
                
                vol_scalar = vol_targeter.get_scalar(port_ret_series, spy_ret_series)
                
                # Combined scalar: vol targeting * kill switch soft scalar
                combined_scalar = vol_scalar * kill_scalar
                
                if combined_scalar > 0:
                    invested = sum(p['cost'] for p in active_positions)
                    available_capital = capital - invested
                    
                    if available_capital > capital * 0.1:
                        # Prepare signals
                        cols_to_get = ['close', 'atr_abs', 'prob_score', 'asset_class']
                        day_signals = current_day_data[[c for c in cols_to_get if c in current_day_data.columns]].reset_index()
                        day_signals = day_signals.rename(columns={'close': 'price'})
                        day_signals['vix_level'] = current_vix
                        
                        # Earnings exclusion
                        dt_no_tz_date = dt_no_tz.date()
                        mask = day_signals['ticker'].apply(
                            lambda t: not exclusions.get((t, dt_no_tz_date), False)
                        )
                        day_signals = day_signals[mask]
                        day_signals = day_signals[
                            day_signals['asset_class'].apply(
                                lambda ac: _regime_allows_long(ac, date, class_trend_filters)
                            )
                        ]
                        candidate_count = len(_select_top_candidates(day_signals, picks_per_class=2))
                        
                        # Build sanitized returns dict for MVRK
                        returns_dict = {}
                        for t in day_signals['ticker'].values:
                            hist = np.asarray(asset_returns_history.get(t, []), dtype=float)
                            hist = hist[np.isfinite(hist)]
                            hist = np.clip(hist, -MAX_RETURN_ABS_CLIP, MAX_RETURN_ABS_CLIP)
                            if len(hist) >= 10:
                                returns_dict[t] = hist[-60:]
                            else:
                                returns_dict[t] = np.array([0.0] * 10)
                        
                        # DIRECTIVE 5: MVRK Kelly-sized trade selection
                        new_trades = select_trades_mvrk(
                            day_signals, available_capital,
                            vol_scalar, kill_scalar,
                            returns_dict, mvrk_sizer,
                            picks_per_class=2
                        )
                        if candidate_count > 0 and not new_trades:
                            print(
                                f"  [WARN {pd.Timestamp(date).date()}] {candidate_count} candidates selected but 0 executable trades after sizing."
                            )
                        
                        for trade in new_trades:
                            if available_capital >= trade['cost']:
                                capital -= trade['cost']
                                available_capital -= trade['cost']
                                trade['days_held'] = 0
                                trade['peak_return'] = 0.0
                                active_positions.append(trade)
                                year_trades += 1
            
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            # DIRECTIVE 2: TAIL RISK HEDGE â€” Update daily MTM
            # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
            spy_vol = current_vix / 100.0 if current_vix else 0.20
            hedge_nav = capital + mtm_invested
            allow_new_hedges = _should_buy_tail_hedge(
                current_vix, hedge_nav, kill_switch.high_water_mark
            )
            hedge_val, premium = tail_hedger.update(
                current_spy, spy_vol, hedge_nav,
                day_idx_offset + d_idx,
                allow_new_hedges=allow_new_hedges
            )
            capital -= premium  # deduct premium cost
            
            # Record daily MTM value
            mtm_invested = 0
            for pos in active_positions:
                if pos['ticker'] in current_day_data.index:
                    mtm_invested += pos['shares'] * current_day_data.loc[pos['ticker'], 'close']
                else:
                    mtm_invested += pos['cost']
            
            total_value = capital + mtm_invested + hedge_val
            
            daily_portfolio_value.append({
                'date': date,
                'capital': total_value,
                'status': 'ACTIVE'
            })
            
            # Update portfolio returns buffer
            if len(daily_portfolio_value) >= 2:
                prev_cap = daily_portfolio_value[-2]['capital']
                if prev_cap > 0:
                    portfolio_returns_buffer.append(
                        (total_value / prev_cap) - 1
                    )
        
        year_end_capital = daily_portfolio_value[-1]['capital']
        year_return = (year_end_capital / year_start_capital) - 1
        win_rate = (winning_trades / year_trades * 100) if year_trades > 0 else 0
        
        print(f"{test_year:<9} | {year_trades:<8} | {win_rate:>5.1f}%     | {year_return*100:>7.2f}% | {year_kill_events}")
    
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    # FINAL ANALYSIS (Enhanced Reporting)
    # â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•â•
    if not daily_portfolio_value:
        return
        
    portfolio_df = pd.DataFrame(daily_portfolio_value).set_index('date')
    portfolio_df['daily_return'] = portfolio_df['capital'].pct_change()
    
    final_capital = portfolio_df['capital'].iloc[-1]
    total_return = (final_capital / initial_capital) - 1
    
    years_traded = len(portfolio_df) / 252
    annualized_return = (1 + total_return) ** (1 / years_traded) - 1 if years_traded > 0 else 0
    
    volatility = portfolio_df['daily_return'].std() * (252 ** 0.5)
    sharpe = (portfolio_df['daily_return'].mean() * 252) / volatility if volatility > 0 else 0
    
    downside_returns = portfolio_df[portfolio_df['daily_return'] < 0]['daily_return']
    downside_volatility = downside_returns.std() * (252 ** 0.5)
    sortino = (portfolio_df['daily_return'].mean() * 252) / downside_volatility if downside_volatility > 0 else 0
    
    running_max = portfolio_df['capital'].cummax()
    drawdown = (portfolio_df['capital'] - running_max) / running_max
    max_dd = drawdown.min()
    
    # Calmar Ratio  
    calmar = annualized_return / abs(max_dd) if max_dd != 0 else 0
    
    # Tail Ratio (95th percentile gain / 5th percentile loss)
    returns = portfolio_df['daily_return'].dropna()
    p95 = returns.quantile(0.95) if len(returns) > 0 else 0
    p05 = abs(returns.quantile(0.05)) if len(returns) > 0 else 1
    tail_ratio = p95 / p05 if p05 > 0 else 0
    
    trades_df = pd.DataFrame(all_trades_log)
    total_trades = len(trades_df)
    overall_win_rate = (trades_df['profit'] > 0).mean() * 100 if total_trades > 0 else 0
    
    # Kill switch stats
    ks_stats = kill_switch.get_stats()
    
    # Tail hedge stats
    hedge_stats = tail_hedger.get_cumulative_stats()
    
    print("\n" + "=" * 75)
    print("TOTAL PORTFOLIO PERFORMANCE (ADVANCED RISK ENGINE)")
    print("=" * 75)
    print(f"Initial Capital:         ${initial_capital:,.2f}")
    print(f"Final Capital:           ${final_capital:,.2f}")
    print(f"Total Return:            {total_return*100:.2f}%")
    print(f"Annualized Return:       {annualized_return*100:.2f}%")
    print(f"Sharpe Ratio:            {sharpe:.3f}")
    print(f"Sortino Ratio:           {sortino:.3f}")
    print(f"Calmar Ratio:            {calmar:.3f}")
    print(f"Tail Ratio (95/5):       {tail_ratio:.3f}")
    print(f"Max Drawdown:            {max_dd*100:.2f}%")
    print(f"Annualized Volatility:   {volatility*100:.2f}%")
    print(f"Total Trades:            {total_trades}")
    print(f"Avg Trades/Year:         {total_trades / years_traded:.1f}")
    print(f"Overall Win Rate:        {overall_win_rate:.1f}%")
    print("-" * 75)
    print("RISK ENGINE DIAGNOSTICS")
    print("-" * 75)
    print(f"Kill Switch Triggers:    {ks_stats['kill_count']}")
    if ks_stats['kill_dates']:
        print(f"Kill Dates:              {', '.join(str(d)[:10] for d in ks_stats['kill_dates'][:5])}")
    print(f"Tail Hedge Premium Paid: ${hedge_stats['total_premium_paid']:,.2f}")
    print(f"Tail Hedge Payoff:       ${hedge_stats['total_payoff']:,.2f}")
    print(f"Tail Hedge Net P&L:      ${hedge_stats['net_pnl']:,.2f}")
    print(f"MVRK kappa (idio penalty): {mvrk_sizer.kappa}")
    print(f"Vol Target:              {vol_targeter.target_vol*100:.1f}%")
    print("=" * 75)

if __name__ == "__main__":
    run_universe_backtest()
