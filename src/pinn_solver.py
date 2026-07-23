"""
Physics-Informed Neural Networks (PINNs) for Black-Scholes PDE Solving
"""

import torch
import torch.nn as nn

class BlackScholesPINN(nn.Module):
    """
    Neural Network solving the Black-Scholes Partial Differential Equation:
    ∂V/∂t + 0.5 * σ² * S² * ∂²V/∂S² + r * S * ∂V/∂S - r * V = 0
    """
    
    def __init__(self, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), # Inputs: (S, t)
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1)    # Output: V(S, t)
        )

    def forward(self, S: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        inputs = torch.cat([S, t], dim=1)
        return self.net(inputs)

    def pde_residual(self, S: torch.Tensor, t: torch.Tensor, r: float = 0.05, sigma: float = 0.20) -> torch.Tensor:
        """Computes residual of Black-Scholes PDE using Automatic Differentiation."""
        S.requires_grad_(True)
        t.requires_grad_(True)

        V = self.forward(S, t)

        dV_dt = torch.autograd.grad(V, t, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        dV_dS = torch.autograd.grad(V, S, grad_outputs=torch.ones_like(V), create_graph=True)[0]
        d2V_dS2 = torch.autograd.grad(dV_dS, S, grad_outputs=torch.ones_like(dV_dS), create_graph=True)[0]

        residual = dV_dt + 0.5 * (sigma ** 2) * (S ** 2) * d2V_dS2 + r * S * dV_dS - r * V
        return residual
