"""
Multi-Asset Market Data Ingestion Pipeline
Fetches spot prices, realized volatility, and option chains for Top 10 S&P 500 stocks via yfinance.
"""

import os
import time
import numpy as np
import pandas as pd
from scipy.stats import norm
import yfinance as yf

TOP_10_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "TSLA", "JPM", "LLY", "AVGO"
]

DEFAULT_RISK_FREE_RATE = 0.045


def fetch_ticker_metadata(ticker_symbol: str, period: str = "1y"):
    """
    Fetches spot price and annual realized volatility for a given ticker.
    """
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period=period)
    if hist.empty:
        raise ValueError(f"No price history found for {ticker_symbol}")

    spot_price = float(hist["Close"].iloc[-1])
    returns = hist["Close"].pct_change().dropna()
    realized_vol = float(returns.std() * np.sqrt(252))
    return ticker, spot_price, realized_vol


def fetch_top10_sp500_options(max_expirations_per_ticker: int = 4, max_options_per_ticker: int = 80):
    """
    Fetches real option chains for Top 10 S&P 500 stocks and constructs a normalized dataset.
    Returns:
        df_all: pd.DataFrame with normalized moneyness (m = S/K), tau, sigma, and normalized market price v = V/K
        ticker_stats: dict with spot and realized volatility per ticker
    """
    all_options = []
    ticker_stats = {}

    print(f"Fetching real option chains for Top 10 S&P 500 stocks...")
    t0 = time.time()

    for sym in TOP_10_TICKERS:
        try:
            ticker, spot, vol = fetch_ticker_metadata(sym)
            ticker_stats[sym] = {
                "spot": spot,
                "realized_vol": vol,
                "risk_free_rate": DEFAULT_RISK_FREE_RATE
            }
            print(f"  [+] {sym:5s} | Spot: ${spot:7.2f} | Realized Vol: {vol*100:4.1f}% | Option Expirations: {len(ticker.options)}")

            expirations = ticker.options
            count_ticker = 0

            for exp in expirations[:max_expirations_per_ticker]:
                if count_ticker >= max_options_per_ticker:
                    break
                try:
                    chain = ticker.option_chain(exp)
                    calls = chain.calls
                    exp_date = pd.to_datetime(exp)
                    today = pd.to_datetime("today")
                    tau = max(float((exp_date - today).days) / 365.0, 0.02)

                    for _, row in calls.iterrows():
                        K = float(row["strike"])
                        bid = float(row["bid"])
                        ask = float(row["ask"])
                        moneyness = spot / K

                        if bid > 0 and ask > 0 and 0.70 <= moneyness <= 1.30:
                            mid_price = 0.5 * (bid + ask)
                            norm_price = mid_price / K
                            all_options.append({
                                "ticker": sym,
                                "S": spot,
                                "K": K,
                                "moneyness": moneyness,
                                "tau": tau,
                                "sigma": vol,
                                "r": DEFAULT_RISK_FREE_RATE,
                                "market_price": mid_price,
                                "normalized_price": norm_price,
                                "implied_vol": float(row.get("impliedVolatility", vol))
                            })
                            count_ticker += 1
                            if count_ticker >= max_options_per_ticker:
                                break
                except Exception as e:
                    continue

        except Exception as err:
            print(f"  [!] Failed fetching {sym}: {err}")
            continue

    df_all = pd.DataFrame(all_options)
    elapsed = time.time() - t0
    print(f"\n[+] Successfully ingested {len(df_all)} option contracts across {len(ticker_stats)} stocks in {elapsed:.1f}s.")
    return df_all, ticker_stats


if __name__ == "__main__":
    df, stats = fetch_top10_sp500_options(max_expirations_per_ticker=3, max_options_per_ticker=50)
    print("\nSample records:")
    print(df.head())
    print("\nRecords per ticker:")
    print(df["ticker"].value_counts())
