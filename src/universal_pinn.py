"""
Universal Multi-Asset Physics-Informed Neural Network (PINN)
Solves the Dimensionless Black-Scholes PDE across multiple underlying assets,
moneyness levels m = S/K, maturities tau, and volatility regimes sigma.
"""

import torch
import torch.nn as nn


class UniversalMultiAssetPINN(nn.Module):
    """
    Universal PINN pricing operator:
      Input:  (m = S/K, tau = T - t, sigma) [3 dimensions]
      Output: v = V/K (Normalized option price) [1 dimension]
    """
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, m: torch.Tensor, tau: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        """
        Forward pass: outputs dimensionless option price v = V/K.
        """
        x = torch.cat([m, tau, sigma], dim=1)
        return self.net(x)

    def pde_residual(self, m: torch.Tensor, tau: torch.Tensor, sigma: torch.Tensor, r: float = 0.045):
        """
        Evaluates the dimensionless Black-Scholes PDE residual using autograd:
          R_PDE = dv/dtau - [ 0.5 * sigma^2 * m^2 * d2v/dm2 + r * m * dv/dm - r * v ]
        """
        m.requires_grad_(True)
        tau.requires_grad_(True)
        sigma.requires_grad_(True)

        v = self.forward(m, tau, sigma)

        dv_dtau = torch.autograd.grad(v, tau, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        dv_dm = torch.autograd.grad(v, m, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        d2v_dm2 = torch.autograd.grad(dv_dm, m, grad_outputs=torch.ones_like(dv_dm), create_graph=True)[0]

        pde_res = dv_dtau - (0.5 * (sigma ** 2) * (m ** 2) * d2v_dm2 + r * m * dv_dm - r * v)
        return pde_res, dv_dm, d2v_dm2

    def compute_price_and_greeks(self, S: float, K: float, tau: float, sigma: float, r: float = 0.045):
        """
        Infers option price V and exact Greeks (Delta, Gamma, Vega, Theta) via autograd.
        """
        self.eval()
        device = next(self.parameters()).device
        m_val = S / K
        m_t = torch.tensor([[m_val]], dtype=torch.float32, device=device, requires_grad=True)
        tau_t = torch.tensor([[tau]], dtype=torch.float32, device=device, requires_grad=True)
        sigma_t = torch.tensor([[sigma]], dtype=torch.float32, device=device, requires_grad=True)

        v = self.forward(m_t, tau_t, sigma_t)
        dv_dm = torch.autograd.grad(v, m_t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        d2v_dm2 = torch.autograd.grad(dv_dm, m_t, grad_outputs=torch.ones_like(dv_dm), create_graph=True)[0]
        dv_dsigma = torch.autograd.grad(v, sigma_t, grad_outputs=torch.ones_like(v), create_graph=True)[0]
        dv_dtau = torch.autograd.grad(v, tau_t, grad_outputs=torch.ones_like(v), create_graph=True)[0]

        # Reconstructed dollar amounts
        price = float(v.item()) * K
        delta = float(dv_dm.item()) # dV/dS = (K*dv)/(K*dm) = dv/dm
        gamma = float(d2v_dm2.item()) / K # d2V/dS2 = (d2v/dm2) / K
        vega = float(dv_dsigma.item()) * K / 100.0 # Per 1% vol change
        theta = -float(dv_dtau.item()) * K / 365.0 # Per day decay

        return {
            "price": price,
            "delta": delta,
            "gamma": gamma,
            "vega": vega,
            "theta": theta,
            "normalized_v": float(v.item())
        }
