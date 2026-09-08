"""
Polygon.io / Massive Official Market Data Provider for Options and Equities
Loads POLYGON_API_KEY securely from .env and fetches official options reference contracts,
closing prices, and microstructure metadata.
"""

import os
import json
import time
import urllib.request
import pandas as pd
from typing import Optional, Dict, List


def load_api_key() -> Optional[str]:
    """
    Loads Polygon / Massive API key from environment or .env file.
    """
    key = os.getenv("POLYGON_API_KEY") or os.getenv("MASSIVE_API_KEY")
    if key:
        return key

    # Search in .env file in parent directories
    env_paths = [".env", os.path.join(os.path.dirname(__file__), "..", ".env")]
    for path in env_paths:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line.startswith("POLYGON_API_KEY=") or line.startswith("MASSIVE_API_KEY="):
                        return line.split("=", 1)[1].strip()
    return None


def fetch_polygon_options_contracts(
    ticker: str = "AAPL",
    contract_type: str = "call",
    limit: int = 50
) -> List[Dict]:
    """
    Fetches official options reference contracts from Polygon.io.
    Endpoint: /v3/reference/options/contracts
    """
    api_key = load_api_key()
    if not api_key:
        print("[WARN] No Polygon API key found in environment or .env.")
        return []

    url = (
        f"https://api.polygon.io/v3/reference/options/contracts?"
        f"underlying_ticker={ticker.upper()}&contract_type={contract_type}&limit={limit}&apiKey={api_key}"
    )

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Deep-Stochastic-Volatility/2.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data.get("results", [])
    except Exception as e:
        print(f"[ERROR] Failed to fetch Polygon contracts for {ticker}: {e}")
        return []


def fetch_polygon_option_prev_close(option_ticker: str) -> Optional[Dict]:
    """
    Fetches official previous closing price, volume, and VWAP for a specific option contract.
    Endpoint: /v2/aggs/ticker/{optionTicker}/prev
    """
    api_key = load_api_key()
    if not api_key:
        return None

    url = f"https://api.polygon.io/v2/aggs/ticker/{option_ticker}/prev?apiKey={api_key}"

    try:
        req = urllib.request.Request(url, headers={"User-Agent": "Deep-Stochastic-Volatility/2.0"})
        with urllib.request.urlopen(req, timeout=10) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            results = data.get("results", [])
            if results:
                return results[0]
            return None
    except Exception as e:
        print(f"[ERROR] Failed to fetch prev close for {option_ticker}: {e}")
        return None


if __name__ == "__main__":
    key = load_api_key()
    masked_key = (key[:6] + "..." + key[-4:]) if key else "None"
    print(f"[+] Loaded Polygon API Key: {masked_key}")

    print("[+] Fetching AAPL options contracts from Polygon...")
    contracts = fetch_polygon_options_contracts("AAPL", limit=5)
    print(f"[OK] Fetched {len(contracts)} contracts:")
    for c in contracts:
        print(f"  Ticker: {c.get('ticker')} | Strike: ${c.get('strike_price')} | Expiry: {c.get('expiration_date')} | Style: {c.get('exercise_style')}")

    if contracts:
        sample_opt = contracts[0].get("ticker")
        print(f"\n[+] Fetching official previous close bar for {sample_opt}...")
        bar = fetch_polygon_option_prev_close(sample_opt)
        if bar:
            print(f"[OK] Close Price: ${bar.get('c')} | Volume: {bar.get('v')} | VWAP: ${bar.get('vw')}")
