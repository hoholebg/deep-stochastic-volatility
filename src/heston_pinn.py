"""
Heston 2D Stochastic Volatility PINN and Semi-Analytical Fourier Solver
Solves the 2D Heston PDE in dimensionless space (m = S/K, v, tau):
  du/dtau = 0.5 * v * m^2 * d2u/dm2 + rho * xi * v * m * d2u/(dm dv)
            + 0.5 * xi^2 * v * d2u/dv2 + r * m * du/dm + kappa * (theta - v) * du/dv - r * u
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.integrate import quad


class HestonPINN(nn.Module):
    """
    2D Physics-Informed Neural Network for Heston Stochastic Volatility.
    Inputs:
      m:   Moneyness (S / K)
      v:   Instantaneous Variance
      tau: Time to Maturity (T - t)
    Output:
      u:   Normalized Option Price (V / K)
    """
    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, m: torch.Tensor, v: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        x = torch.cat([m, v, tau], dim=1)
        return self.net(x)

    def pde_residual(
        self,
        m: torch.Tensor,
        v: torch.Tensor,
        tau: torch.Tensor,
        kappa: float = 2.0,
        theta: float = 0.04,
        xi: float = 0.30,
        rho: float = -0.70,
        r: float = 0.045
    ):
        """
        Evaluates the 2D Heston PDE residual using autograd for spatial, variance,
        temporal, and cross-derivatives d2u/(dm dv).
        """
        m.requires_grad_(True)
        v.requires_grad_(True)
        tau.requires_grad_(True)

        u = self.forward(m, v, tau)

        du_dtau = torch.autograd.grad(u, tau, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        du_dm = torch.autograd.grad(u, m, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        d2u_dm2 = torch.autograd.grad(du_dm, m, grad_outputs=torch.ones_like(du_dm), create_graph=True)[0]

        du_dv = torch.autograd.grad(u, v, grad_outputs=torch.ones_like(u), create_graph=True)[0]
        d2u_dv2 = torch.autograd.grad(du_dv, v, grad_outputs=torch.ones_like(du_dv), create_graph=True)[0]

        # Exact cross derivative d2u / (dm dv) via autograd
        d2u_dm_dv = torch.autograd.grad(du_dm, v, grad_outputs=torch.ones_like(du_dm), create_graph=True)[0]

        pde_res = du_dtau - (
            0.5 * v * (m ** 2) * d2u_dm2
            + rho * xi * v * m * d2u_dm_dv
            + 0.5 * (xi ** 2) * v * d2u_dv2
            + r * m * du_dm
            + kappa * (theta - v) * du_dv
            - r * u
        )
        return pde_res, du_dm, d2u_dm2, du_dv


def heston_characteristic_function(phi, S0, K, tau, r, v0, kappa=2.0, theta=0.04, xi=0.30, rho=-0.70, j=1):
    """
    Heston (1993) characteristic function P_j(phi).
    """
    u = 0.5 if j == 1 else -0.5
    b = kappa - rho * xi if j == 1 else kappa
    a = kappa * theta
    x = np.log(S0)
    d = np.sqrt((rho * xi * 1j * phi - b) ** 2 - (xi ** 2) * (2 * u * 1j * phi - phi ** 2))
    g = (b - rho * xi * 1j * phi + d) / (b - rho * xi * 1j * phi - d)

    C = r * 1j * phi * tau + (a / (xi ** 2)) * (
        (b - rho * xi * 1j * phi + d) * tau - 2 * np.log((1 - g * np.exp(d * tau)) / (1 - g))
    )
    D = ((b - rho * xi * 1j * phi + d) / (xi ** 2)) * ((1 - np.exp(d * tau)) / (1 - g * np.exp(d * tau)))
    return np.exp(C + D * v0 + 1j * phi * x)


def heston_analytical_call_price(S0, K, tau, r, v0, kappa=2.0, theta=0.04, xi=0.30, rho=-0.70):
    """
    Semi-analytical European Call price under Heston model using Fourier inversion.
    """
    def integrand1(phi):
        cf = heston_characteristic_function(phi, S0, K, tau, r, v0, kappa, theta, xi, rho, j=1)
        return (np.exp(-1j * phi * np.log(K)) * cf / (1j * phi)).real

    def integrand2(phi):
        cf = heston_characteristic_function(phi, S0, K, tau, r, v0, kappa, theta, xi, rho, j=2)
        return (np.exp(-1j * phi * np.log(K)) * cf / (1j * phi)).real

    int1, _ = quad(integrand1, 1e-6, 100, limit=200)
    int2, _ = quad(integrand2, 1e-6, 100, limit=200)

    P1 = 0.5 + (1.0 / np.pi) * int1
    P2 = 0.5 + (1.0 / np.pi) * int2

    price = S0 * P1 - K * np.exp(-r * tau) * P2
    return max(0.0, float(price))
