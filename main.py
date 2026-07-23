import torch
from src.pinn_solver import BlackScholesPINN

def main():
    print("=== PyTorch Physics-Informed Neural Networks (PINNs) Option Pricing ===")
    
    model = BlackScholesPINN(hidden_dim=32)
    
    # Sample inputs (Stock price S = 100, Time t = 0.5 years)
    S = torch.tensor([[100.0], [105.0], [95.0]], requires_grad=True)
    t = torch.tensor([[0.5], [0.5], [0.5]], requires_grad=True)
    
    predicted_prices = model(S, t)
    residuals = model.pde_residual(S, t)

    print("Predicted Option Prices V(S, t):")
    for s_val, price in zip(S.detach().numpy().flatten(), predicted_prices.detach().numpy().flatten()):
        print(f"  S = ${s_val:.1f} -> NN Option Value = ${price:.4f}")
        
    print(f"\nMean PDE Residual Loss: {torch.mean(residuals**2).item():.6e}")

if __name__ == "__main__":
    main()
