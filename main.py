import time
import torch
import numpy as np
import pandas as pd
from src.analytical import bs_call_price, bs_call_delta
from src.fdm_solver import fdm_black_scholes
from src.pinn_solver import BlackScholesPINN

def main():
    print("=== Deep Stochastic Volatility: PINN vs Analytical vs FDM Benchmark ===")
    
    K, T, r, sigma = 100.0, 1.0, 0.05, 0.20
    model = BlackScholesPINN(hidden_dim=64)

    S_test = np.array([80.0, 90.0, 100.0, 110.0, 120.0])
    S_torch = torch.tensor(S_test, dtype=torch.float32).unsqueeze(1).requires_grad_(True)
    tau_torch = torch.full((len(S_test), 1), T, dtype=torch.float32).requires_grad_(True)

    bs_prices = bs_call_price(S_test, K, T, r, sigma)
    bs_deltas = bs_call_delta(S_test, K, T, r, sigma)

    _, pinn_deltas_t, _ = model.pde_residual(S_torch, tau_torch, K, r, sigma)
    pinn_prices_t = model(S_torch, tau_torch)

    df = pd.DataFrame({
        "Spot (S)": S_test,
        "Exact BS Price ($)": np.round(bs_prices, 4),
        "Exact BS Delta": np.round(bs_deltas, 4),
        "PINN Price ($)": np.round(pinn_prices_t.detach().numpy().flatten(), 4),
        "PINN Autograd Delta": np.round(pinn_deltas_t.detach().numpy().flatten(), 4)
    })

    print("\nBenchmark Evaluation Grid:")
    print(df.to_string(index=False))

if __name__ == "__main__":
    main()
