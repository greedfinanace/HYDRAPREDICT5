import pandas as pd
import numpy as np
import yfinance as yf
import ta
import warnings
import logging
from pathlib import Path
from typing import Optional

warnings.simplefilter("default")
logger = logging.getLogger(__name__)

try:
    from hmmlearn.hmm import GaussianHMM
    HMM_AVAILABLE = True
except ImportError:
    HMM_AVAILABLE = False
    print("Warning: hmmlearn not installed. HMM regime detection disabled. Install with: pip install hmmlearn")

UNIVERSE = {
    'equities': [
        'AAPL','MSFT','GOOGL','AMZN','NVDA','META','TSLA','JPM','JNJ','V',
        'UNH','XOM','PG','MA','HD','CVX','MRK','ABBV','KO','AVGO','COST',
        'TMO','MCD','WMT','BAC','ABT','CSCO','ACN','CRM','ADBE','NKE',
        'TXN','NEE','PM','QCOM','BMY','HON','AMGN','UPS','LOW','SBUX',
        'GS','INTU','ELV','MDT','LIN','DHR','RTX','PEP','BRK-B'
    ],
    'sectors': ['XLK','XLE','XLF','XLV','XLI','XLP','XLY','XLB','XLU','XLRE','XLC'],
    'international': ['EFA','EEM','VGK','EWJ','FXI','EWZ','INDA','EWC','EWA'],
    'fixed_income': ['TLT','IEF','SHY','HYG','LQD','EMB','TIP','BNDX'],
    'commodities': ['GLD','SLV','USO','UNG','PDBC','CORN','WEAT','COPX'],
    'reits': ['VNQ','IYR','REM','KBWY','SRVR'],
}

ALL_ASSETS = [t for assets in UNIVERSE.values() for t in assets]

def get_asset_class(ticker):
    for ac, tickers in UNIVERSE.items():
        if ticker in tickers:
            return ac
    return 'Unknown'

PROJECT_ROOT = Path(__file__).resolve().parent
LOCAL_DATA_DIR = PROJECT_ROOT / "data"
_LOCAL_REQUIRED_TICKERS = ALL_ASSETS + ["SPY", "TLT", "UUP", "BTC-USD"]
_REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
_TICKER_ALIASES = {
    "BTC-USD": ["BTC-USD", "BTC", "BTCUSD"],
}

def _parse_daily_csv(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if df.empty:
        return None

    date_col = next((c for c in df.columns if c.strip().lower() == "date"), None)
    if date_col is None:
        return None

    df = df.rename(columns={date_col: "Date"})
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    if df.empty:
        return None

    df = df.sort_values("Date")
    df = df.set_index("Date")
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)

    col_map = {}
    for required in _REQUIRED_COLUMNS:
        match = next((c for c in df.columns if c.strip().lower() == required.lower()), None)
        if match:
            col_map[match] = required

    df = df.rename(columns=col_map)
    if "Adj Close" not in df.columns and "Close" in df.columns:
        df["Adj Close"] = df["Close"]

    if not all(col in df.columns for col in ("Open", "High", "Low", "Close", "Volume")):
        return None

    df = df[_REQUIRED_COLUMNS]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["Close"])

    if df.empty:
        return None

    return df[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]

def _parse_hourly_txt(path: Path) -> Optional[pd.DataFrame]:
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if df.empty:
        return None

    def _clean_column(col: str) -> str:
        return col.strip().strip("<>").upper()

    df.columns = [_clean_column(c) for c in df.columns]
    required_cols = {"DATE", "TIME", "OPEN", "HIGH", "LOW", "CLOSE", "VOL"}
    if not required_cols.issubset(df.columns):
        return None

    df["DATETIME"] = pd.to_datetime(
        df["DATE"].astype(str).str.zfill(8) + df["TIME"].astype(str).str.zfill(6),
        format="%Y%m%d%H%M%S",
        errors="coerce"
    )
    df = df.dropna(subset=["DATETIME"])
    if df.empty:
        return None

    df = df.set_index("DATETIME")
    df = df.sort_index()
    df = df[["OPEN", "HIGH", "LOW", "CLOSE", "VOL"]]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(subset=["OPEN"])
    if df.empty:
        return None

    daily = df.resample("D").agg({
        "OPEN": "first",
        "HIGH": "max",
        "LOW": "min",
        "CLOSE": "last",
        "VOL": "sum",
    })
    daily = daily.dropna(subset=["OPEN"])
    if daily.empty:
        return None

    daily = daily.rename(columns={
        "OPEN": "Open",
        "HIGH": "High",
        "LOW": "Low",
        "CLOSE": "Close",
        "VOL": "Volume",
    })
    daily["Adj Close"] = daily["Close"]
    daily = daily[["Open", "High", "Low", "Close", "Adj Close", "Volume"]]
    return daily

def _read_local_price_frame(path: Path) -> Optional[pd.DataFrame]:
    if not path.exists():
        return None

    suffix = path.suffix.lower()
    if suffix == ".csv":
        return _parse_daily_csv(path)
    if suffix == ".txt":
        return _parse_hourly_txt(path)
    return None

def _build_ticker_index():
    if not LOCAL_DATA_DIR.exists():
        return {}

    expected_keys = set(t.upper() for t in _LOCAL_REQUIRED_TICKERS)
    for aliases in _TICKER_ALIASES.values():
        expected_keys.update(alias.upper() for alias in aliases)

    index = {}
    for path in LOCAL_DATA_DIR.rglob("*"):
        if not path.is_file():
            continue
        suffix = path.suffix.lower()
        if suffix not in {".csv", ".txt"}:
            continue

        candidate = path.name.split(".")[0].upper()
        # Handle ticker name transformations
        candidate = candidate.replace("_", "-")
        if candidate in expected_keys and candidate not in index:
            index[candidate] = path

        if len(index) >= len(expected_keys):
            break

    return index

def _load_local_market_data():
    ticker_index = _build_ticker_index()
    if not ticker_index:
        return None, {}, list(_LOCAL_REQUIRED_TICKERS)

    frames = {}
    missing = []
    for ticker in _LOCAL_REQUIRED_TICKERS:
        df = None
        candidates = [ticker] + _TICKER_ALIASES.get(ticker, [])
        for candidate in candidates:
            path = ticker_index.get(candidate.upper())
            if path is None:
                continue
            df = _read_local_price_frame(path)
            if df is not None:
                break

        if df is None:
            missing.append(ticker)
        else:
            frames[ticker] = df

    if missing:
        return None, {}, missing

    multi_frames = []
    for ticker in ALL_ASSETS:
        df = frames[ticker].copy()
        df.columns = pd.MultiIndex.from_product([[ticker], df.columns])
        multi_frames.append(df)

    data = pd.concat(multi_frames, axis=1).sort_index(axis=1)
    return data, frames, []
def fetch_universe_data(start="2010-01-01"):
    local_data, local_frames, missing = _load_local_market_data()
    if local_data is not None:
        print(f"Using local data cache from {LOCAL_DATA_DIR}")
        spy = local_frames["SPY"]
        tlt = local_frames["TLT"]
        usd = local_frames["UUP"]
        btc = local_frames["BTC-USD"]
        return local_data, spy, tlt, usd, btc

    if missing:
        sample = ", ".join(missing[:6])
        print(f"Local data cache incomplete ({len(missing)} files missing: {sample}{'...' if len(missing) > 6 else ''}); falling back to yfinance.")

    print(f"Downloading data for {len(ALL_ASSETS)} tickers...")
    data = yf.download(ALL_ASSETS, start=start, group_by='ticker', auto_adjust=True, progress=False)

    print("Downloading benchmark data (SPY, TLT, UUP)...")
    spy = yf.download("SPY", start=start, auto_adjust=True, progress=False)
    tlt = yf.download("TLT", start=start, auto_adjust=True, progress=False)
    # Using UUP for USD
    usd = yf.download("UUP", start=start, auto_adjust=True, progress=False)

    # Orthogonal feature sources (not tradable, feature-only)
    print("Downloading orthogonal feature sources (BTC-USD)...")
    btc = yf.download("BTC-USD", start=start, auto_adjust=True, progress=False)

    if isinstance(spy.columns, pd.MultiIndex): spy = spy.droplevel(1, axis=1)
    if isinstance(tlt.columns, pd.MultiIndex): tlt = tlt.droplevel(1, axis=1)
    if isinstance(usd.columns, pd.MultiIndex): usd = usd.droplevel(1, axis=1)
    if isinstance(btc.columns, pd.MultiIndex): btc = btc.droplevel(1, axis=1)

    return data, spy, tlt, usd, btc

def build_earnings_exclusion_mask(universe):
    """
    Returns a dict: {(ticker, date): True} for dates within ±2 days of earnings.
    Uses yfinance earnings_dates (will only have recent history).
    """
    print("Building earnings exclusion mask (fetching recent earnings dates)...")
    exclusions = {}
    for ticker in universe:
        try:
            info = yf.Ticker(ticker)
            dates = info.get_earnings_dates()
            if dates is None or dates.empty:
                continue
            for earn_date in dates.index:
                # remove timezone to match our other dates
                if earn_date.tzinfo is not None:
                    earn_date_dt = earn_date.tz_localize(None).date()
                else:
                    earn_date_dt = earn_date.date()
                    
                for delta in range(-1, 4):   # exclude 1 day before → 3 days after
                    excl_date = pd.Timestamp(earn_date_dt) + pd.Timedelta(days=delta)
                    exclusions[(ticker, excl_date)] = True
        except Exception:
            continue
    return exclusions

def fit_hmm_regimes(spy_df, n_states=2, lookback=504):
    """
    Fit a 2-state Gaussian HMM on SPY daily returns to detect market regimes.
    State 0 = low-vol (bull), State 1 = high-vol (bear).
    Returns a Series aligned to spy_df.index with regime labels.
    """
    if not HMM_AVAILABLE:
        return pd.Series(0, index=spy_df.index, name='hmm_regime')
    
    spy_returns = spy_df['Close'].pct_change().dropna()
    
    if len(spy_returns) < 100:
        return pd.Series(0, index=spy_df.index, name='hmm_regime')
    
    try:
        returns_data = spy_returns.values.reshape(-1, 1)
        model = GaussianHMM(n_components=n_states, covariance_type='full',
                           n_iter=100, random_state=42)
        model.fit(returns_data[-lookback:])
        states = model.predict(returns_data)
        
        # Ensure state 0 = low-vol (bull), state 1 = high-vol (bear)
        state_vols = [np.sqrt(model.covars_[i][0, 0]) for i in range(n_states)]
        if state_vols[0] > state_vols[1]:
            states = 1 - states
        
        regime_series = pd.Series(states, index=spy_returns.index, name='hmm_regime')
        # Reindex to full spy_df index, forward-fill
        regime_series = regime_series.reindex(spy_df.index).ffill().fillna(0).astype(int)
        return regime_series
    except Exception as e:
        print(f"HMM fitting failed: {e}. Using default regime=0.")
        return pd.Series(0, index=spy_df.index, name='hmm_regime')


def make_features_by_class(ticker, ticker_df, asset_class, spy_df, tlt_df, usd_df,
                           btc_df=None, hyg_df=None, hmm_regimes=None):
    f = pd.DataFrame(index=ticker_df.index)
    close = ticker_df['Close']
    volume = ticker_df['Volume']

    if len(close) < 252: return f

    # EWMA Volatility Forecast (RiskMetrics λ=0.94)
    daily_ret = close.pct_change()
    ewma_var = daily_ret.ewm(alpha=0.06, adjust=False).var()  # α = 1 - λ = 0.06
    f['forecasted_vol'] = np.sqrt(ewma_var * 252)
    
    # Signal stability: autocorrelation of 5-day momentum (higher = more persistent signal)
    mom5 = close.pct_change(5)
    f['signal_stability'] = mom5.rolling(20).apply(
        lambda x: x.autocorr(lag=1) if len(x.dropna()) > 5 else 0, raw=False
    ).fillna(0)

    # Universal features
    f['rsi_14'] = ta.momentum.RSIIndicator(close, window=14).rsi()
    f['mom_5d'] = close.pct_change(5)
    f['mom_20d'] = close.pct_change(20)
    
    atr = ta.volatility.AverageTrueRange(ticker_df['High'], ticker_df['Low'], close, window=14).average_true_range()
    f['atr_pct'] = atr / close
    f['atr_abs'] = atr
    
    high_52w = close.rolling(252).max()
    f['pct_from_52w_high'] = (close / high_52w.replace(0, np.nan)) - 1
    
    vol_ma20 = volume.rolling(20).mean()
    f['vol_ratio'] = volume / vol_ma20.replace(0, np.nan)

    # Basic features to keep from prior setup
    f['bb_pct_b'] = ta.volatility.BollingerBands(close).bollinger_pband()
    f['gap'] = (ticker_df['Open'] - close.shift(1)) / close.shift(1).replace(0, np.nan)
    
    price_change_5d = close.pct_change(5)
    volume_change_5d = volume.pct_change(5)
    f['vol_price_divergence'] = price_change_5d - volume_change_5d * 0.1

    # Apply asset-class specific logic
    if asset_class in ('equities', 'sectors', 'international', 'reits'):
        f['rs_vs_spy_5d'] = close.pct_change(5) - spy_df['Close'].pct_change(5)
        f['rs_vs_spy_20d'] = close.pct_change(20) - spy_df['Close'].pct_change(20)
        spy_sma200 = spy_df['Close'].rolling(200).mean()
        f['spy_regime'] = (spy_df['Close'] > spy_sma200).astype(int)

    if asset_class == 'fixed_income':
        f['rs_vs_spy_5d'] = close.pct_change(5) - spy_df['Close'].pct_change(5)
        f['yield_momentum'] = -close.pct_change(20)
        f['flight_to_safety'] = (spy_df['Close'].pct_change(5) < -0.02).astype(int)
        # Re-zero volume ratio for bonds
        f['vol_ratio'] = 1.0 

    if asset_class == 'commodities':
        f['usd_5d'] = usd_df['Close'].pct_change(5)
        f['rs_vs_spy_5d'] = close.pct_change(5) - spy_df['Close'].pct_change(5)
        f['inflation_regime'] = (tlt_df['Close'].pct_change(20) < -0.01).astype(int)

    # ═══════════════════════════════════════════════════════════════════
    # ORTHOGONAL CROSS-ASSET FEATURES (all asset classes)
    # ═══════════════════════════════════════════════════════════════════
    # HYG credit spread proxy: falling HYG = widening spreads = risk-off
    if hyg_df is not None and 'Close' in hyg_df.columns and len(hyg_df) > 20:
        hyg_close = hyg_df['Close'].reindex(ticker_df.index, method='ffill')
        f['hyg_spread_5d'] = -hyg_close.pct_change(5)   # inverted: positive = risk-off
        f['hyg_spread_20d'] = -hyg_close.pct_change(20)
    
    # BTC momentum as non-linear leading indicator
    if btc_df is not None and 'Close' in btc_df.columns and len(btc_df) > 20:
        btc_close = btc_df['Close'].reindex(ticker_df.index, method='ffill')
        f['btc_mom_5d'] = btc_close.pct_change(5)
        f['btc_mom_20d'] = btc_close.pct_change(20)
    
    # HMM regime state (0=bull, 1=bear)
    if hmm_regimes is not None:
        f['hmm_regime'] = hmm_regimes.reindex(ticker_df.index, method='ffill').fillna(0).astype(int)

    f['next_5d_return'] = close.pct_change(5).shift(-5)
    f['close'] = close

    return f

def build_all_features(data, spy_df, tlt_df, usd_df, btc_df=None):
    print("Building features for all tickers...")
    
    # Fit HMM regimes on SPY
    print("Fitting HMM regime model on SPY...")
    hmm_regimes = fit_hmm_regimes(spy_df)
    
    # Extract HYG data from universe if available (it's in fixed_income)
    hyg_df = None
    if isinstance(data.columns, pd.MultiIndex) and 'HYG' in data.columns.levels[0]:
        hyg_df = data['HYG'].copy()
        if isinstance(hyg_df.columns, pd.MultiIndex):
            hyg_df.columns = hyg_df.columns.droplevel(0)
    
    all_features = []

    if isinstance(data.columns, pd.MultiIndex):
        for ticker in ALL_ASSETS:
            if ticker in data.columns.levels[0]:
                ticker_df = data[ticker].copy()
                if isinstance(ticker_df.columns, pd.MultiIndex):
                    ticker_df.columns = ticker_df.columns.droplevel(0)

                asset_class = get_asset_class(ticker)
                feats = make_features_by_class(ticker, ticker_df, asset_class,
                                              spy_df, tlt_df, usd_df,
                                              btc_df=btc_df, hyg_df=hyg_df,
                                              hmm_regimes=hmm_regimes)
                if feats.empty:
                    continue
                    
                feats['ticker'] = ticker
                feats['asset_class'] = asset_class
                feats = feats.reset_index()
                
                date_col = 'Date' if 'Date' in feats.columns else feats.columns[0]
                feats = feats.set_index([date_col, 'ticker'])
                all_features.append(feats)
    else:
        print("Warning: Unexpected data format from yfinance")
        return pd.DataFrame()

    df = pd.concat(all_features)
    df = df.dropna(subset=['rsi_14', 'mom_5d', 'atr_pct', 'pct_from_52w_high'])
    
    # Fill NAs for new risk features
    for rf in ['forecasted_vol', 'signal_stability']:
        if rf in df.columns:
            df[rf] = df[rf].fillna(0)
    
    # Fill NAs for class-specific features only
    class_feats = ['rs_vs_spy_5d', 'rs_vs_spy_20d', 'spy_regime', 'yield_momentum',
                   'flight_to_safety', 'usd_5d', 'inflation_regime']
    for cf in class_feats:
        if cf in df.columns:
            df[cf] = df[cf].fillna(0)
    
    # Fill NAs for orthogonal features
    ortho_feats = ['hyg_spread_5d', 'hyg_spread_20d', 'btc_mom_5d', 'btc_mom_20d', 'hmm_regime']
    for of_ in ortho_feats:
        if of_ in df.columns:
            df[of_] = df[of_].fillna(0)
            
    return df

def make_cross_sectional_labels(df, top_pct=0.25):
    print("Creating cross-sectional labels...")
    
    # Grouping by level=0 (Date) AND 'asset_class'
    ranked = df.groupby([df.index.get_level_values(0), 'asset_class'])['next_5d_return'].rank(pct=True)
    
    labels = (ranked >= (1 - top_pct)).astype(int)
    labels.loc[df['next_5d_return'].isna()] = np.nan
    return labels

def add_cross_sectional_ranks(df, feature_cols):
    print("Adding cross-sectional rank features...")
    ranked = df.groupby([df.index.get_level_values(0), 'asset_class'])[feature_cols].rank(pct=True)
    ranked.columns = [f'{c}_xrank' for c in feature_cols]
    return pd.concat([df, ranked], axis=1)

if __name__ == "__main__":
    data, spy, tlt, usd, btc = fetch_universe_data(start="2020-01-01")
    df = build_all_features(data, spy, tlt, usd, btc)
    print(df.head())
    print(f"\nNew orthogonal features present: {[c for c in df.columns if c in ['hyg_spread_5d','hyg_spread_20d','btc_mom_5d','btc_mom_20d','hmm_regime']]}")
