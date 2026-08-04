"""
Real Market Data Ingestion via yfinance
"""

import numpy as np
import yfinance as yf

def fetch_market_parameters(ticker_symbol: str = "NVDA") -> tuple:
    ticker = yf.Ticker(ticker_symbol)
    hist = ticker.history(period="1y")
    
    if hist.empty:
        S0, sigma = 120.0, 0.35
    else:
        S0 = float(hist["Close"].iloc[-1])
        daily_returns = hist["Close"].pct_change().dropna()
        sigma = float(daily_returns.std() * np.sqrt(252))
        
    r = 0.045
    return S0, sigma, r
