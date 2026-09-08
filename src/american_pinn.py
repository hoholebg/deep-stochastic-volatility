"""
American Option Free-Boundary PINN and CRR Binomial Tree Solver
Solves the American Put Variational Inequality:
  min( L_BS[u], u - h(m) ) = 0
where:
  L_BS[u] = du/dtau - (0.5 * sigma^2 * m^2 * d2u/dm2 + r * m * du/dm - r * u)
  h(m) = max(1 - m, 0)
Benchmarked against a 1,000-step Cox-Ross-Rubinstein (CRR) Binomial Tree.
"""

import numpy as np
import torch
import torch.nn as nn


class AmericanOptionPINN(nn.Module):
    """
    Physics-Informed Neural Network for American Put Option Pricing
    incorporating the free-boundary obstacle condition (Variational Inequality).

    Inputs:
      m:   Normalized Moneyness (S / K)
      tau: Time to Expiration (T - t)
    Output:
      u:   Normalized Option Value (V / K)
    """
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, m: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        x = torch.cat([m, tau], dim=1)
        return self.net(x)

    def pde_residual(self, m: torch.Tensor, tau: torch.Tensor, r: float = 0.045, sigma: float = 0.20):
        """
        Computes Black-Scholes PDE operator L_BS[u] via PyTorch Autograd.
        """
        m.requires_grad_(True)
        tau.requires_grad_(True)

        u = self.forward(m, tau)

        du_dtau = torch.autograd.grad(u, tau, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        du_dm = torch.autograd.grad(u, m, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        d2u_dm2 = torch.autograd.grad(du_dm, m, grad_outputs=torch.ones_like(du_dm), create_graph=True)[0]

        bs_operator = du_dtau - (0.5 * (sigma ** 2) * (m ** 2) * d2u_dm2 + r * m * du_dm - r * u)
        return bs_operator, u, du_dm, d2u_dm2


def crr_american_put(S0: float, K: float, tau: float, r: float, sigma: float, N: int = 1000) -> float:
    """
    Computes exact American Put Option price using an N-step Cox-Ross-Rubinstein Binomial Tree.
    """
    if tau <= 1e-7:
        return max(K - S0, 0.0)

    dt = tau / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    j = np.arange(N + 1)
    ST = S0 * (u ** j) * (d ** (N - j))
    values = np.maximum(K - ST, 0.0)

    for step in range(N - 1, -1, -1):
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])
        S_step = S0 * (u ** np.arange(step + 1)) * (d ** (step - np.arange(step + 1)))
        values = np.maximum(values, np.maximum(K - S_step, 0.0))

    return float(values[0])


def crr_european_put(S0: float, K: float, tau: float, r: float, sigma: float, N: int = 1000) -> float:
    """
    Computes European Put Option price using an N-step Cox-Ross-Rubinstein Binomial Tree.
    """
    if tau <= 1e-7:
        return max(K - S0, 0.0)

    dt = tau / N
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    p = (np.exp(r * dt) - d) / (u - d)
    discount = np.exp(-r * dt)

    j = np.arange(N + 1)
    ST = S0 * (u ** j) * (d ** (N - j))
    values = np.maximum(K - ST, 0.0)

    for step in range(N - 1, -1, -1):
        values = discount * (p * values[1:] + (1.0 - p) * values[:-1])

    return float(values[0])


def early_exercise_premium(S0: float, K: float, tau: float, r: float, sigma: float, N: int = 1000) -> float:
    """
    Calculates Early Exercise Premium (EEP) = V_American - V_European.
    """
    p_am = crr_american_put(S0, K, tau, r, sigma, N=N)
    p_eu = crr_european_put(S0, K, tau, r, sigma, N=N)
    return max(0.0, p_am - p_eu)
