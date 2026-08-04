"""
Physics-Informed Neural Networks (PINNs) PyTorch Architecture for Black-Scholes PDE
"""

import torch
import torch.nn as nn

class BlackScholesPINN(nn.Module):
    """
    Neural Network solving Black-Scholes PDE:
    ∂V/∂τ = 0.5 * σ² * S² * ∂²V/∂S² + r * S * ∂V/∂S - r * V
    where τ = T - t.
    """
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, S: torch.Tensor, tau: torch.Tensor) -> torch.Tensor:
        x = torch.cat([S, tau], dim=1)
        return self.net(x)

    def pde_residual(self, S: torch.Tensor, tau: torch.Tensor, K: float, r: float, sigma: float):
        S.requires_grad_(True)
        tau.requires_grad_(True)

        V = self.forward(S, tau)

        dV_dtau = torch.autograd.grad(V, tau, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        dV_dS = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        d2V_dS2 = torch.autograd.grad(dV_dS, S, grad_outputs=torch.ones_like(dV_dS), create_graph=True)[0]

        pde_loss = dV_dtau - (0.5 * (sigma ** 2) * (S ** 2) * d2V_dS2 + r * S * dV_dS - r * V)
        return pde_loss, dV_dS, d2V_dS2
