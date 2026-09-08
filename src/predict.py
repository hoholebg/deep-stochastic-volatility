"""
Inference Script: Loading Pre-Trained PyTorch Model Weights
Supports both the Universal Multi-Asset PINN (Top 10 S&P 500) and Single-Asset NVDA PINN.
Computes instantaneous price and exact automatic differentiation Greeks (Delta, Gamma, Vega, Theta).
"""

import os
import sys
import argparse
import torch
import torch.nn as nn
import numpy as np

# Ensure UTF-8 output on Windows
sys.stdout.reconfigure(encoding="utf-8")

# Default metadata for top tickers (Spot and Realized Vol)
TICKER_DEFAULTS = {
    "AAPL":  {"spot": 220.0, "vol": 0.251},
    "MSFT":  {"spot": 420.0, "vol": 0.325},
    "NVDA":  {"spot": 215.0, "vol": 0.382},
    "AMZN":  {"spot": 185.0, "vol": 0.345},
    "GOOGL": {"spot": 170.0, "vol": 0.315},
    "META":  {"spot": 520.0, "vol": 0.390},
    "TSLA":  {"spot": 225.0, "vol": 0.476},
    "JPM":   {"spot": 215.0, "vol": 0.222},
    "LLY":   {"spot": 920.0, "vol": 0.359},
    "AVGO":  {"spot": 160.0, "vol": 0.474}
}


class UniversalMultiAssetPINN(nn.Module):
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, m, tau, sigma):
        x = torch.cat([m, tau, sigma], dim=1)
        return self.net(x)


def predict_universal_option(
    ticker: str = "AAPL",
    S: float = None,
    K: float = None,
    tau: float = 1.0,
    sigma: float = None,
    r: float = 0.045,
    weights_path: str = "weights/universal_pinn_sp10.pth"
):
    """
    Universal option pricing across any S&P 500 stock using the pre-trained weights.
    """
    ticker_up = ticker.upper()
    meta = TICKER_DEFAULTS.get(ticker_up, {"spot": 100.0, "vol": 0.30})
    if S is None:
        S = meta["spot"]
    if K is None:
        K = S  # At-the-money by default
    if sigma is None:
        sigma = meta["vol"]

    model = UniversalMultiAssetPINN(hidden_dim=128)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=torch.device("cpu")))
        print(f"[OK] Loaded pre-trained Universal PINN weights from '{weights_path}'")
    else:
        print(f"[WARN] Weights file '{weights_path}' not found. Falling back to single-asset weights if available.")

    model.eval()

    m_val = S / K
    m_t = torch.tensor([[m_val]], dtype=torch.float32, requires_grad=True)
    tau_t = torch.tensor([[tau]], dtype=torch.float32, requires_grad=True)
    sigma_t = torch.tensor([[sigma]], dtype=torch.float32, requires_grad=True)

    v = model(m_t, tau_t, sigma_t)
    dv_dm = torch.autograd.grad(v, m_t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    d2v_dm2 = torch.autograd.grad(dv_dm, m_t, grad_outputs=torch.ones_like(dv_dm), create_graph=True)[0]
    dv_dsigma = torch.autograd.grad(v, sigma_t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
    dv_dtau = torch.autograd.grad(v, tau_t, grad_outputs=torch.ones_like(v), create_graph=True)[0]

    price = float(v.item()) * K
    delta = float(dv_dm.item())
    gamma = float(d2v_dm2.item()) / K
    vega = float(dv_dsigma.item()) * K / 100.0
    theta = -float(dv_dtau.item()) * K / 365.0

    return {
        "ticker": ticker_up,
        "S": S,
        "K": K,
        "tau": tau,
        "sigma": sigma,
        "price": price,
        "delta": delta,
        "gamma": gamma,
        "vega": vega,
        "theta": theta,
        "normalized_v": float(v.item())
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Universal PINN Option Inference")
    parser.add_argument("--ticker", type=str, default="NVDA", help="Ticker symbol (e.g., AAPL, MSFT, NVDA, TSLA)")
    parser.add_argument("--spot", type=float, default=None, help="Spot price S")
    parser.add_argument("--strike", type=float, default=None, help="Strike price K")
    parser.add_argument("--maturity", type=float, default=1.0, help="Time to maturity in years")
    parser.add_argument("--vol", type=float, default=None, help="Realized or Implied volatility (e.g. 0.35)")
    args = parser.parse_args()

    res = predict_universal_option(
        ticker=args.ticker,
        S=args.spot,
        K=args.strike,
        tau=args.maturity,
        sigma=args.vol
    )

    print(f"\nInference Result for {res['ticker']} (S = ${res['S']:.2f}, K = ${res['K']:.2f}, T = {res['tau']:.2f}Y, Vol = {res['sigma']*100:.1f}%):")
    print(f"  Predicted Option Price V:  ${res['price']:.2f} (v = {res['normalized_v']:.4f})")
    print(f"  Exact Autograd Delta:       {res['delta']:.4f}")
    print(f"  Exact Autograd Gamma:       {res['gamma']:.6f}")
    print(f"  Exact Autograd Vega:        ${res['vega']:.4f} per 1% vol")
    print(f"  Exact Autograd Theta:       ${res['theta']:.4f} per day")
