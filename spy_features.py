import pandas as pd
import yfinance as yf

# Fetch SPY data
spy = yf.download("SPY", start="2000-01-01", progress=False)
if isinstance(spy.columns, pd.MultiIndex):
    spy = spy.droplevel(1, axis=1)

# Calculate rolling returns (stationary)
for window in [5, 10, 20]:
    spy[f"return_{window}d"] = spy["Close"].pct_change(window) * 100

# Price distance from 50-day SMA (percentage)
spy["sma_50"] = spy["Close"].rolling(50).mean()
spy["price_sma50_pct"] = ((spy["Close"] - spy["sma_50"]) / spy["sma_50"]) * 100

# Rolling 20-day historical volatility (annualized)
spy["volatility_20d"] = spy["Close"].pct_change().rolling(20).std() * (252 ** 0.5) * 100

# Fetch VIX and 10-year Treasury yield
vix = yf.download("^VIX", start="2000-01-01", progress=False)
if isinstance(vix.columns, pd.MultiIndex):
    vix = vix.droplevel(1, axis=1)
spy["vix_pct_change"] = vix["Close"].pct_change() * 100

treasury = yf.download("^TNX", start="2000-01-01", progress=False)
if isinstance(treasury.columns, pd.MultiIndex):
    treasury = treasury.droplevel(1, axis=1)
spy["treasury_10y_pct_change"] = treasury["Close"].pct_change() * 100

# Select features and drop NaN
features = spy[["return_5d", "return_10d", "return_20d", "price_sma50_pct", 
                "volatility_20d", "vix_pct_change", "treasury_10y_pct_change"]].dropna()

print("Columns:", features.columns.tolist())
print(features.head(10))
print(f"\nShape: {features.shape}")
print(f"Date range: {features.index[0]} to {features.index[-1]}")
print("\nNaN count per column:")
print(features.isna().sum())