"""
Inference Script: Loading Pre-Trained PyTorch Model Weights
"""

import os
import torch
import torch.nn as nn
import numpy as np

class RealMarketPINN(nn.Module):
    def __init__(self, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.SiLU(),
            nn.Linear(hidden_dim, 1)
        )

    def forward(self, S, tau):
        return self.net(torch.cat([S, tau], dim=1))

def predict_option_price_and_greeks(S_val: float, tau_val: float, weights_path: str = "weights/pinn_bs_nvda.pth"):
    model = RealMarketPINN(hidden_dim=128)
    if os.path.exists(weights_path):
        model.load_state_dict(torch.load(weights_path, map_location=torch.device('cpu')))
        print(f"Loaded pre-trained PyTorch weights from '{weights_path}'")
    else:
        print("Model weights file not found. Running untrained network.")

    model.eval()
    S_t = torch.tensor([[S_val]], dtype=torch.float32, requires_grad=True)
    tau_t = torch.tensor([[tau_val]], dtype=torch.float32, requires_grad=True)

    V = model(S_t, tau_t)
    Delta = torch.autograd.grad(V, S_t, grad_outputs=torch.ones_like(V), create_graph=True)[0]
    Gamma = torch.autograd.grad(Delta, S_t, grad_outputs=torch.ones_like(Delta), create_graph=True)[0]

    return {
        "price": float(V.item()),
        "delta": float(Delta.item()),
        "gamma": float(Gamma.item())
    }

if __name__ == "__main__":
    res = predict_option_price_and_greeks(211.94, 1.0)
    print(f"\nInference Result for NVDA (S = 211.94, T = 1.0):")
    print(f"  Predicted Option Price V:  ${res['price']:.2f}")
    print(f"  Exact Autograd Delta:       {res['delta']:.4f}")
    print(f"  Exact Autograd Gamma:       {res['gamma']:.6f}")
