"""
Training and Benchmarking: 2D Heston Stochastic Volatility PINN & American Option Variational Inequality PINN

1. Heston 2D PINN:
   - Solves 2D Heston PDE with cross-derivative autograd d2u / (dm dv).
   - Evaluates against semi-analytical Fourier inversion benchmark.
   - Saves weights to 'weights/heston_pinn.pth'.
   - Generates 'assets/heston_benchmark.png'.

2. American Option Free-Boundary PINN:
   - Solves American Put Variational Inequality obstacle problem:
       min( L_BS[u], u - h(m) ) = 0
   - Evaluates against a 1,000-step Cox-Ross-Rubinstein (CRR) Binomial Tree.
   - Isolates the Early Exercise Premium (EEP) = V_American - V_European.
   - Saves weights to 'weights/american_pinn.pth'.
   - Generates 'assets/american_option_benchmark.png'.
"""

import os
import sys
import time
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.heston_pinn import HestonPINN, heston_analytical_call_price
from src.american_pinn import AmericanOptionPINN, crr_american_put, crr_european_put

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

os.makedirs("weights", exist_ok=True)
os.makedirs("assets", exist_ok=True)
os.makedirs("data", exist_ok=True)


# ==============================================================================
# 1. TRAIN & BENCHMARK HESTON 2D STOCHASTIC VOLATILITY PINN
# ==============================================================================

def train_heston(epochs: int = 1500):
    print("=" * 70)
    print("STAGE 1: 2D HESTON STOCHASTIC VOLATILITY PINN TRAINING")
    print("=" * 70)

    # Market & Heston Model Parameters
    kappa = 2.0      # Mean-reversion speed
    theta = 0.04     # Long-term variance (20% long-term vol)
    xi = 0.30        # Volatility of variance
    rho = -0.70      # Spot-vol correlation (leverage effect)
    r = 0.045        # Risk-free rate
    K = 100.0        # Reference strike

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[+] Execution Device: {device} ({gpu_name})")

    # 1. Collocation Points in Dimensionless Domain (m, v, tau)
    N_pde = 2000
    m_pde = (torch.rand(N_pde, 1, device=device) * 1.0 + 0.50)        # m in [0.50, 1.50]
    v_pde = (torch.rand(N_pde, 1, device=device) * 0.15 + 0.01)       # v in [0.01, 0.16] (vol 10% - 40%)
    tau_pde = (torch.rand(N_pde, 1, device=device) * 0.95 + 0.05)     # tau in [0.05, 1.00]

    # 2. Initial Condition (tau = 0)
    N_ic = 600
    m_ic = (torch.rand(N_ic, 1, device=device) * 1.0 + 0.50)
    v_ic = (torch.rand(N_ic, 1, device=device) * 0.15 + 0.01)
    tau_ic = torch.zeros(N_ic, 1, device=device)
    u_ic_target = torch.relu(m_ic - 1.0)              # Call payoff max(m - 1, 0)

    # 3. Boundary Conditions
    # Deep OTM boundary (m = 0.3) -> u = 0
    N_bc = 300
    m_bc_lo = torch.ones(N_bc, 1, device=device) * 0.30
    v_bc_lo = (torch.rand(N_bc, 1, device=device) * 0.15 + 0.01)
    tau_bc_lo = (torch.rand(N_bc, 1, device=device) * 1.0)
    u_bc_lo_target = torch.zeros(N_bc, 1, device=device)

    # Deep ITM boundary (m = 1.8) -> u = m - exp(-r * tau)
    m_bc_hi = torch.ones(N_bc, 1, device=device) * 1.80
    v_bc_hi = (torch.rand(N_bc, 1, device=device) * 0.15 + 0.01)
    tau_bc_hi = (torch.rand(N_bc, 1, device=device) * 1.0)
    u_bc_hi_target = m_bc_hi - torch.exp(-r * tau_bc_hi)

    model = HestonPINN(hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"[INFO] Training Heston PINN on {gpu_name} ({epochs} Epochs)...")
    t0 = time.time()
    loss_history = []
    pde_history = []
    ic_history = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # PDE residual with full autograd cross derivatives
        pde_res, _, _, _ = model.pde_residual(
            m_pde, v_pde, tau_pde,
            kappa=kappa, theta=theta, xi=xi, rho=rho, r=r
        )
        loss_pde = torch.mean(pde_res ** 2)

        # Initial condition loss
        u_ic = model(m_ic, v_ic, tau_ic)
        loss_ic = torch.mean((u_ic - u_ic_target) ** 2)

        # Boundary losses
        u_bc_lo = model(m_bc_lo, v_bc_lo, tau_bc_lo)
        loss_bc_lo = torch.mean((u_bc_lo - u_bc_lo_target) ** 2)

        u_bc_hi = model(m_bc_hi, v_bc_hi, tau_bc_hi)
        loss_bc_hi = torch.mean((u_bc_hi - u_bc_hi_target) ** 2)

        loss = loss_pde + 40.0 * loss_ic + 15.0 * loss_bc_lo + 15.0 * loss_bc_hi
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())
        pde_history.append(loss_pde.item())
        ic_history.append(loss_ic.item())

        if epoch % 300 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:4d}/{epochs} | Total Loss: {loss.item():.6f} | PDE: {loss_pde.item():.6f} | IC: {loss_ic.item():.6f}")

    train_time = time.time() - t0
    print(f"[OK] Heston PINN trained in {train_time:.2f} seconds.")

    # Save weights
    torch.save(model.state_dict(), "weights/heston_pinn.pth")
    print("[OK] Model weights saved to 'weights/heston_pinn.pth'")

    # 4. Benchmarking vs Semi-Analytical Fourier Inversion
    model.eval()
    test_spots = np.linspace(75.0, 125.0, 11)
    v0_eval = 0.04   # 20% instantaneous volatility
    tau_eval = 0.50  # 6-month horizon

    fourier_prices = []
    pinn_prices = []
    deltas = []

    for s in test_spots:
        p_fourier = heston_analytical_call_price(s, K, tau_eval, r, v0_eval, kappa, theta, xi, rho)
        fourier_prices.append(p_fourier)

        s_t = torch.tensor([[s / K]], dtype=torch.float32, device=device, requires_grad=True)
        v_t = torch.tensor([[v0_eval]], dtype=torch.float32, device=device)
        tau_t = torch.tensor([[tau_eval]], dtype=torch.float32, device=device)

        u_pred = model(s_t, v_t, tau_t)
        p_pinn = u_pred.item() * K
        pinn_prices.append(p_pinn)

        # Autograd Delta: dV/dS = du/dm
        d_val = torch.autograd.grad(u_pred, s_t)[0].item()
        deltas.append(d_val)

    fourier_prices = np.array(fourier_prices)
    pinn_prices = np.array(pinn_prices)
    abs_errors = np.abs(pinn_prices - fourier_prices)
    mae_heston = np.mean(abs_errors)
    rmse_heston = np.sqrt(np.mean(abs_errors ** 2))

    print(f"\n[BENCHMARK] Heston 2D PINN vs Fourier Semi-Analytical:")
    print(f"  MAE:  ${mae_heston:.4f}")
    print(f"  RMSE: ${rmse_heston:.4f}")
    print(f"  Max Absolute Error: ${np.max(abs_errors):.4f}")

    # 5. Multi-Variance Profile for Visualization
    v_profiles = [0.02, 0.04, 0.09]  # 14.1% vol, 20.0% vol, 30.0% vol
    curve_spots = np.linspace(70.0, 130.0, 40)
    profile_curves = {}
    for v_val in v_profiles:
        pinn_curve = []
        fourier_curve = []
        for s in curve_spots:
            f_p = heston_analytical_call_price(s, K, tau_eval, r, v_val, kappa, theta, xi, rho)
            with torch.no_grad():
                s_t = torch.tensor([[s / K]], dtype=torch.float32, device=device)
                v_t = torch.tensor([[v_val]], dtype=torch.float32, device=device)
                tau_t = torch.tensor([[tau_eval]], dtype=torch.float32, device=device)
                pred_p = model(s_t, v_t, tau_t).item() * K
            pinn_curve.append(pred_p)
            fourier_curve.append(f_p)
        profile_curves[v_val] = (np.array(pinn_curve), np.array(fourier_curve))

    # 6. Generate Dark-Themed Figure
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 5.5), facecolor="#000000")

    # Panel 1: Loss Convergence
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_facecolor("#000000")
    ax1.plot(loss_history, color="#00e5ff", lw=1.8, label="Total Loss")
    ax1.plot(pde_history, color="#ff9100", lw=1.5, alpha=0.85, label="2D PDE Residual")
    ax1.plot(ic_history, color="#00e676", lw=1.4, alpha=0.75, label="Payoff IC")
    ax1.set_yscale("log")
    ax1.set_title("Heston 2D PINN Loss Convergence", color="#ffffff", fontsize=12, pad=12)
    ax1.set_xlabel("Epoch", color="#ffffff", fontsize=11)
    ax1.set_ylabel("Loss (Log Scale)", color="#ffffff", fontsize=11)
    ax1.legend(loc="upper right", frameon=False, labelcolor="#ffffff")
    ax1.grid(False)
    for spine in ax1.spines.values():
        spine.set_color("#ffffff")
        spine.set_linewidth(1.5)
    ax1.tick_params(colors="#ffffff", width=1.5)

    # Panel 2: Model vs Fourier Curves Across Variance Regimes
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_facecolor("#000000")
    palette = ["#00e5ff", "#7c4dff", "#ff9100"]
    for i, v_val in enumerate(v_profiles):
        vol_pct = np.sqrt(v_val) * 100
        p_pinn, p_four = profile_curves[v_val]
        ax2.plot(curve_spots, p_four, color=palette[i], linestyle="--", lw=1.5, alpha=0.6, label=f"Fourier (Vol={vol_pct:.0f}%)")
        ax2.plot(curve_spots, p_pinn, color=palette[i], lw=2.2, label=f"Heston PINN (Vol={vol_pct:.0f}%)")

    ax2.set_title("Call Pricing across Stochastic Volatility Regimes", color="#ffffff", fontsize=12, pad=12)
    ax2.set_xlabel("Underlying Spot Price ($)", color="#ffffff", fontsize=11)
    ax2.set_ylabel("Option Price ($)", color="#ffffff", fontsize=11)
    ax2.legend(loc="upper left", frameon=False, fontsize=8.5, labelcolor="#ffffff")
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_color("#ffffff")
        spine.set_linewidth(1.5)
    ax2.tick_params(colors="#ffffff", width=1.5)

    # Panel 3: Absolute Pricing Error & Autograd Delta
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_facecolor("#000000")
    ax3_twin = ax3.twinx()
    ax3_twin.set_facecolor("#000000")

    b1 = ax3.bar(test_spots - 0.5, abs_errors, width=1.0, color="#ff5252", alpha=0.85, label="Pricing Error ($)", edgecolor="#ffffff", lw=1.0)
    l1 = ax3_twin.plot(test_spots, deltas, color="#00e676", lw=2.2, marker="o", markersize=5, label="Autograd Delta (dV/dS)")

    ax3.set_title(f"Heston Absolute Error (MAE: ${mae_heston:.2f}) & Autograd Delta", color="#ffffff", fontsize=12, pad=12)
    ax3.set_xlabel("Spot Price ($)", color="#ffffff", fontsize=11)
    ax3.set_ylabel("Absolute Error ($)", color="#ffffff", fontsize=11)
    ax3_twin.set_ylabel("Delta", color="#00e676", fontsize=11)
    ax3.set_ylim(0, max(abs_errors) * 1.4 + 0.1)
    ax3_twin.set_ylim(-0.05, 1.05)

    ax3.grid(False)
    ax3_twin.grid(False)
    for spine in ax3.spines.values():
        spine.set_color("#ffffff")
        spine.set_linewidth(1.5)
    for spine in ax3_twin.spines.values():
        spine.set_color("#ffffff")
        spine.set_linewidth(1.5)
    ax3.tick_params(colors="#ffffff", width=1.5)
    ax3_twin.tick_params(colors="#00e676", width=1.5)

    plt.tight_layout()
    chart_path = "assets/heston_benchmark.png"
    plt.savefig(chart_path, dpi=200, facecolor="#000000", edgecolor="none")
    plt.close()
    print(f"[OK] Heston benchmark graphic saved to '{chart_path}'")

    return {
        "model": "Heston 2D PINN",
        "mae": float(mae_heston),
        "rmse": float(rmse_heston),
        "max_error": float(np.max(abs_errors)),
        "training_time_seconds": float(train_time)
    }


# ==============================================================================
# 2. TRAIN & BENCHMARK AMERICAN OPTION FREE-BOUNDARY PINN
# ==============================================================================

def train_american(epochs: int = 1500):
    print("\n" + "=" * 70)
    print("STAGE 2: AMERICAN OPTION FREE-BOUNDARY PINN TRAINING")
    print("=" * 70)

    # Market Parameters
    r = 0.045        # Risk-free rate
    sigma = 0.20     # Volatility
    K = 100.0        # Strike price

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[+] Execution Device: {device} ({gpu_name})")

    # 1. Collocation Points in Dimensionless Domain (m, tau)
    N_pde = 2500
    m_pde = (torch.rand(N_pde, 1, device=device) * 1.20 + 0.40)       # Moneyness m in [0.40, 1.60]
    tau_pde = (torch.rand(N_pde, 1, device=device) * 1.00)            # tau in [0.00, 1.00 year]

    # 2. Initial Condition Domain (tau = 0)
    N_ic = 800
    m_ic = (torch.rand(N_ic, 1, device=device) * 1.20 + 0.40)
    tau_ic = torch.zeros(N_ic, 1, device=device)
    u_ic_target = torch.relu(1.0 - m_ic)             # American Put Payoff: max(1 - m, 0)

    # 3. Boundary Conditions
    # Deep OTM boundary (m = 1.70) -> Put value u = 0
    N_bc = 300
    m_bc_hi = torch.ones(N_bc, 1, device=device) * 1.70
    tau_bc_hi = (torch.rand(N_bc, 1, device=device) * 1.00)
    u_bc_hi_target = torch.zeros(N_bc, 1, device=device)

    # Deep ITM boundary (m = 0.30) -> Put value u = 1 - m
    m_bc_lo = torch.ones(N_bc, 1, device=device) * 0.30
    tau_bc_lo = (torch.rand(N_bc, 1, device=device) * 1.00)
    u_bc_lo_target = 1.0 - m_bc_lo

    model = AmericanOptionPINN(hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"[INFO] Training American Option PINN on {gpu_name} ({epochs} Epochs)...")
    t0 = time.time()
    loss_history = []
    pde_history = []
    obstacle_history = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # PDE residual & value prediction
        pde_res, u_pred, _, _ = model.pde_residual(m_pde, tau_pde, r=r, sigma=sigma)
        payoff = torch.relu(1.0 - m_pde)

        # Free-Boundary Obstacle Penalization: u >= payoff everywhere
        obstacle_violation = torch.relu(payoff - u_pred)
        loss_obstacle = torch.mean(obstacle_violation ** 2)

        # In continuation region (u > payoff), Black-Scholes PDE operator equals 0
        continuation_mask = (u_pred > payoff).float()
        loss_pde = torch.mean((continuation_mask * pde_res) ** 2)

        # Variational Inequality Slack: L_BS[u] >= 0 everywhere
        loss_slack = torch.mean(torch.relu(-pde_res) ** 2)

        # Initial Condition loss at tau = 0
        u_ic = model(m_ic, tau_ic)
        loss_ic = torch.mean((u_ic - u_ic_target) ** 2)

        # Boundary losses
        u_bc_hi = model(m_bc_hi, tau_bc_hi)
        loss_bc_hi = torch.mean((u_bc_hi - u_bc_hi_target) ** 2)

        u_bc_lo = model(m_bc_lo, tau_bc_lo)
        loss_bc_lo = torch.mean((u_bc_lo - u_bc_lo_target) ** 2)

        loss = loss_pde + 60.0 * loss_obstacle + 10.0 * loss_slack + 40.0 * loss_ic + 20.0 * loss_bc_hi + 20.0 * loss_bc_lo
        loss.backward()
        optimizer.step()
        scheduler.step()

        loss_history.append(loss.item())
        pde_history.append(loss_pde.item())
        obstacle_history.append(loss_obstacle.item())

        if epoch % 300 == 0 or epoch == epochs:
            print(f"  Epoch {epoch:4d}/{epochs} | Total Loss: {loss.item():.6f} | PDE: {loss_pde.item():.6f} | Obstacle: {loss_obstacle.item():.6f}")

    train_time = time.time() - t0
    print(f"[OK] American Option PINN trained in {train_time:.2f} seconds.")

    # Save weights
    torch.save(model.state_dict(), "weights/american_pinn.pth")
    print("[OK] Model weights saved to 'weights/american_pinn.pth'")

    # 4. Benchmarking vs 1,000-step CRR Binomial Tree
    model.eval()
    test_spots = np.linspace(60.0, 140.0, 17)
    tau_eval = 1.00  # 1-year maturity

    crr_am_prices = []
    crr_eu_prices = []
    pinn_am_prices = []

    for s in test_spots:
        # CRR 1,000-step American and European benchmark
        p_am_crr = crr_american_put(s, K, tau_eval, r, sigma, N=1000)
        p_eu_crr = crr_european_put(s, K, tau_eval, r, sigma, N=1000)
        crr_am_prices.append(p_am_crr)
        crr_eu_prices.append(p_eu_crr)

        # PINN prediction with intrinsic value floor
        with torch.no_grad():
            m_t = torch.tensor([[s / K]], dtype=torch.float32, device=device)
            tau_t = torch.tensor([[tau_eval]], dtype=torch.float32, device=device)
            u_norm = model(m_t, tau_t).item()
            p_am_pinn = max(K - s, u_norm * K)
            pinn_am_prices.append(p_am_pinn)

    crr_am_prices = np.array(crr_am_prices)
    crr_eu_prices = np.array(crr_eu_prices)
    pinn_am_prices = np.array(pinn_am_prices)

    abs_errors = np.abs(pinn_am_prices - crr_am_prices)
    mae_american = np.mean(abs_errors)
    rmse_american = np.sqrt(np.mean(abs_errors ** 2))

    # Early Exercise Premium (EEP) Isolation
    eep_crr = crr_am_prices - crr_eu_prices
    eep_pinn = pinn_am_prices - crr_eu_prices

    print(f"\n[BENCHMARK] American Option PINN vs CRR Binomial Tree (1,000 steps):")
    print(f"  American Put MAE:  ${mae_american:.4f}")
    print(f"  American Put RMSE: ${rmse_american:.4f}")
    print(f"  Max Absolute Error: ${np.max(abs_errors):.4f}")
    print(f"  Max Early Exercise Premium (EEP): ${np.max(eep_crr):.4f} at S = ${test_spots[np.argmax(eep_crr)]:.1f}")

    # 5. Multi-Maturity EEP Structure for Visualization
    tau_list = [0.25, 0.50, 1.00]
    dense_spots = np.linspace(50.0, 130.0, 50)
    eep_profiles = {}

    for tau_v in tau_list:
        eep_v = []
        for s in dense_spots:
            am = crr_american_put(s, K, tau_v, r, sigma, N=500)
            eu = crr_european_put(s, K, tau_v, r, sigma, N=500)
            eep_v.append(max(0.0, am - eu))
        eep_profiles[tau_v] = np.array(eep_v)

    # 6. Generate Dark-Themed Figure
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 5.5), facecolor="#000000")

    # Panel 1: Loss Convergence
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_facecolor("#000000")
    ax1.plot(loss_history, color="#00e5ff", lw=1.8, label="Total Loss")
    ax1.plot(pde_history, color="#ff9100", lw=1.5, alpha=0.85, label="PDE Residual Loss")
    ax1.plot(obstacle_history, color="#e040fb", lw=1.5, alpha=0.85, label="Obstacle Penalty Loss")
    ax1.set_yscale("log")
    ax1.set_title("American Option PINN Loss Convergence", color="#ffffff", fontsize=12, pad=12)
    ax1.set_xlabel("Epoch", color="#ffffff", fontsize=11)
    ax1.set_ylabel("Loss (Log Scale)", color="#ffffff", fontsize=11)
    ax1.legend(loc="upper right", frameon=False, labelcolor="#ffffff")
    ax1.grid(False)
    for spine in ax1.spines.values():
        spine.set_color("#ffffff")
        spine.set_linewidth(1.5)
    ax1.tick_params(colors="#ffffff", width=1.5)

    # Panel 2: American Put Price Curve & Free-Boundary Obstacle
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_facecolor("#000000")
    ax2.plot(test_spots, crr_am_prices, color="#ffffff", linestyle="--", lw=1.8, label="CRR Binomial Tree (1,000 steps)")
    ax2.plot(test_spots, pinn_am_prices, color="#00e5ff", lw=2.4, label="American Option PINN")
    ax2.plot(test_spots, np.maximum(K - test_spots, 0.0), color="#ff5252", linestyle=":", lw=1.5, label="Intrinsic Payoff Floor h(S)")
    ax2.set_title("American Put Pricing vs CRR 1,000-Step Tree", color="#ffffff", fontsize=12, pad=12)
    ax2.set_xlabel("Spot Price ($)", color="#ffffff", fontsize=11)
    ax2.set_ylabel("Option Price ($)", color="#ffffff", fontsize=11)
    ax2.legend(loc="upper right", frameon=False, fontsize=9, labelcolor="#ffffff")
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_color("#ffffff")
        spine.set_linewidth(1.5)
    ax2.tick_params(colors="#ffffff", width=1.5)

    # Panel 3: Early Exercise Premium (EEP) Across Maturities
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_facecolor("#000000")
    colors_eep = ["#00e5ff", "#7c4dff", "#ff9100"]
    for i, tau_v in enumerate(tau_list):
        ax3.plot(dense_spots, eep_profiles[tau_v], color=colors_eep[i], lw=2.0, label=f"EEP (T - t = {tau_v:.2f} yr)")

    ax3.scatter(test_spots, eep_pinn, color="#00e676", s=28, zorder=5, label="PINN Isolated EEP")
    ax3.set_title("Early Exercise Premium (EEP) Dynamics", color="#ffffff", fontsize=12, pad=12)
    ax3.set_xlabel("Spot Price ($)", color="#ffffff", fontsize=11)
    ax3.set_ylabel("Early Exercise Premium ($)", color="#ffffff", fontsize=11)
    ax3.legend(loc="upper right", frameon=False, fontsize=9, labelcolor="#ffffff")
    ax3.grid(False)
    for spine in ax3.spines.values():
        spine.set_color("#ffffff")
        spine.set_linewidth(1.5)
    ax3.tick_params(colors="#ffffff", width=1.5)

    plt.tight_layout()
    chart_path = "assets/american_option_benchmark.png"
    plt.savefig(chart_path, dpi=200, facecolor="#000000", edgecolor="none")
    plt.close()
    print(f"[OK] American option benchmark graphic saved to '{chart_path}'")

    return {
        "model": "American Option PINN",
        "mae": float(mae_american),
        "rmse": float(rmse_american),
        "max_error": float(np.max(abs_errors)),
        "max_eep": float(np.max(eep_crr)),
        "training_time_seconds": float(train_time)
    }


# ==============================================================================
# MAIN EXECUTION
# ==============================================================================

if __name__ == "__main__":
    t_start = time.time()

    heston_metrics = train_heston(epochs=1500)
    american_metrics = train_american(epochs=1500)

    total_duration = time.time() - t_start

    # Save comprehensive metrics json
    combined_report = {
        "heston_stochastic_volatility": heston_metrics,
        "american_option_free_boundary": american_metrics,
        "total_runtime_seconds": float(total_duration)
    }

    with open("data/advanced_models_benchmark_metrics.json", "w") as f:
        json.dump(combined_report, f, indent=2)

    print("\n" + "=" * 70)
    print("TRAINING & BENCHMARKING COMPLETE")
    print(f"Total Runtime: {total_duration:.2f} seconds")
    print(f"Metrics saved to 'data/advanced_models_benchmark_metrics.json'")
    print("=" * 70)
