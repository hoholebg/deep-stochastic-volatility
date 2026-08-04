import pandas as pd
from src.market_data import fetch_market_parameters
from src.structured_products import StructuredProductPricer

def main():
    ticker = "NVDA"
    print(f"=== Deep Stochastic Volatility: Real Market Structured Products ({ticker}) ===")
    
    S0, sigma, r = fetch_market_parameters(ticker)
    pricer = StructuredProductPricer(S0=S0, K=S0, T=1.0, r=r, sigma=sigma)

    asian_res = pricer.price_asian_option_mc(n_sims=50000)
    autocall_res = pricer.price_phoenix_autocall_mc(autocall_barrier=1.0, protection_barrier=0.60)

    print(f"\nReal Market Results for {ticker} (S0 = ${S0:.2f}, Vol = {sigma*100:.1f}%):")
    print(f"  Asian Call Option Fair Value:        ${asian_res['price']:.2f} ({asian_res['elapsed_ms']:.1f} ms)")
    print(f"  Phoenix Autocall Note Fair Value:   ${autocall_res['autocall_note_price']:.2f} ({autocall_res['elapsed_ms']:.1f} ms)")

if __name__ == "__main__":
    main()
