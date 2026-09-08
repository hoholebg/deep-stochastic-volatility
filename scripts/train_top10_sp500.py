"""
Training Script: Universal Multi-Asset PINN on Top 10 S&P 500 Stocks
Trains on real option chains across diverse volatility regimes (AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, JPM, LLY, AVGO).
Evaluates generalization, in-sample vs out-of-sample errors, and per-ticker performance.
"""

import os
import sys
import time
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.stdout.reconfigure(encoding='utf-8')

# Ensure local imports work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from src.multi_asset_data import fetch_top10_sp500_options, DEFAULT_RISK_FREE_RATE
from src.universal_pinn import UniversalMultiAssetPINN

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)

def train_universal_pinn(epochs: int = 5000):
    print("=" * 70)
    print("UNIVERSAL MULTI-ASSET PINN TRAINING: TOP 10 S&P 500 STOCKS")
    print("=" * 70)

    # 1. Ingest Market Data
    df_all, ticker_stats = fetch_top10_sp500_options(max_expirations_per_ticker=4, max_options_per_ticker=70)
    
    # Save ingested data snapshot
    os.makedirs("data", exist_ok=True)
    df_all.to_csv("data/top10_sp500_options_dataset.csv", index=False)
    print(f"\n[+] Saved option chain dataset ({len(df_all)} rows) to 'data/top10_sp500_options_dataset.csv'")

    # 2. Train / Test Split (80% Train, 20% Out-of-Sample Test)
    shuffled_indices = np.random.permutation(len(df_all))
    train_size = int(0.80 * len(df_all))
    train_idx = shuffled_indices[:train_size]
    test_idx = shuffled_indices[train_size:]

    df_train = df_all.iloc[train_idx].copy()
    df_test = df_all.iloc[test_idx].copy()
    print(f"[+] Dataset Split: {len(df_train)} Train contracts | {len(df_test)} Out-of-Sample Test contracts")

    # 3. Device Selection (CUDA on NVIDIA GTX 1070 if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    gpu_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
    print(f"[+] Execution Device: {device} ({gpu_name})")

    # Market Tensors (Train)
    m_train = torch.tensor(df_train["moneyness"].values, dtype=torch.float32, device=device).unsqueeze(1)
    tau_train = torch.tensor(df_train["tau"].values, dtype=torch.float32, device=device).unsqueeze(1)
    sigma_train = torch.tensor(df_train["sigma"].values, dtype=torch.float32, device=device).unsqueeze(1)
    v_train_target = torch.tensor(df_train["normalized_price"].values, dtype=torch.float32, device=device).unsqueeze(1)

    # Market Tensors (Test)
    m_test = torch.tensor(df_test["moneyness"].values, dtype=torch.float32, device=device).unsqueeze(1)
    tau_test = torch.tensor(df_test["tau"].values, dtype=torch.float32, device=device).unsqueeze(1)
    sigma_test = torch.tensor(df_test["sigma"].values, dtype=torch.float32, device=device).unsqueeze(1)

    # 4. PDE Collocation Domain (Meshfree Dimensionless Space)
    N_pde = 4000
    m_pde = (torch.rand(N_pde, 1, device=device) * 0.9 + 0.60)       # moneyness m in [0.60, 1.50]
    tau_pde = (torch.rand(N_pde, 1, device=device) * 1.20)           # tau in [0.00, 1.20 years]
    sigma_pde = (torch.rand(N_pde, 1, device=device) * 0.45 + 0.15)  # sigma in [0.15, 0.60]

    # 5. Initial Condition Domain (Payoff at tau = 0)
    N_ic = 1000
    m_ic = (torch.rand(N_ic, 1, device=device) * 0.9 + 0.60)
    tau_ic = torch.zeros(N_ic, 1, device=device)
    sigma_ic = (torch.rand(N_ic, 1, device=device) * 0.45 + 0.15)
    v_ic_target = torch.relu(m_ic - 1.0) # dimensionless call payoff: max(m - 1, 0)

    # 6. Initialize Model, Optimizer & Scheduler
    model = UniversalMultiAssetPINN(hidden_dim=128).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)

    print(f"\n[INFO] Training Universal Multi-Asset PINN on {gpu_name} ({epochs} Epochs)...")
    t0 = time.time()
    loss_history = []
    pde_history = []
    mkt_history = []

    for epoch in range(1, epochs + 1):
        optimizer.zero_grad()

        # A. PDE Residual Loss
        pde_res, _, _ = model.pde_residual(m_pde, tau_pde, sigma_pde, r=DEFAULT_RISK_FREE_RATE)
        loss_pde = torch.mean(pde_res ** 2)

        # B. Initial Condition Loss
        v_ic_pred = model(m_ic, tau_ic, sigma_ic)
        loss_ic = torch.mean((v_ic_pred - v_ic_target) ** 2)

        # C. Multi-Asset Real Market Data Loss
        v_train_pred = model(m_train, tau_train, sigma_train)
        loss_mkt = torch.mean((v_train_pred - v_train_target) ** 2)

        # Total Composite Loss
        total_loss = loss_pde + 20.0 * loss_ic + 10.0 * loss_mkt
        total_loss.backward()
        optimizer.step()
        scheduler.step()

        loss_val = float(total_loss.item())
        loss_history.append(loss_val)
        pde_history.append(float(loss_pde.item()))
        mkt_history.append(float(loss_mkt.item()))

        if epoch % 1000 == 0 or epoch == epochs:
            print(f"   [Epoch {epoch:4d}/{epochs}] Total Loss: {loss_val:.6f} | PDE: {loss_pde.item():.6f} | IC: {loss_ic.item():.6f} | Mkt: {loss_mkt.item():.6f}")

    train_duration = time.time() - t0
    print(f"\n[OK] Training Completed in {train_duration:.2f}s | Final Loss: {loss_history[-1]:.6f}")

    # 7. Save Model Weights
    os.makedirs("weights", exist_ok=True)
    weights_path = "weights/universal_pinn_sp10.pth"
    torch.save(model.state_dict(), weights_path)
    print(f"[OK] Universal model weights saved to '{weights_path}'")

    # 8. Evaluation & Metrics Calculation
    model.eval()
    with torch.no_grad():
        # Predict Train
        v_pred_train = model(m_train, tau_train, sigma_train).squeeze().cpu().numpy()
        df_train["pred_norm_price"] = v_pred_train
        df_train["pred_price"] = df_train["pred_norm_price"] * df_train["K"]
        df_train["abs_error"] = np.abs(df_train["pred_price"] - df_train["market_price"])

        # Predict Out-of-Sample Test
        v_pred_test = model(m_test, tau_test, sigma_test).squeeze().cpu().numpy()
        df_test["pred_norm_price"] = v_pred_test
        df_test["pred_price"] = df_test["pred_norm_price"] * df_test["K"]
        df_test["abs_error"] = np.abs(df_test["pred_price"] - df_test["market_price"])

    train_mae = df_train["abs_error"].mean()
    train_rmse = np.sqrt((df_train["abs_error"] ** 2).mean())
    test_mae = df_test["abs_error"].mean()
    test_rmse = np.sqrt((df_test["abs_error"] ** 2).mean())

    print("\n" + "=" * 70)
    print("EMPIRICAL EVALUATION METRICS (TOP 10 S&P 500)")
    print("=" * 70)
    print(f"In-Sample Train (80%)  : MAE = ${train_mae:.2f} | RMSE = ${train_rmse:.2f}")
    print(f"Out-of-Sample Test (20%): MAE = ${test_mae:.2f} | RMSE = ${test_rmse:.2f}")

    # Per-Ticker Performance Breakdown
    print("\nPer-Ticker Breakdown (Out-of-Sample Test & Combined):")
    ticker_breakdown = []
    for sym in ticker_stats.keys():
        sub_test = df_test[df_test["ticker"] == sym]
        sub_all = pd.concat([df_train[df_train["ticker"] == sym], sub_test])
        mae_sym = sub_all["abs_error"].mean()
        rel_err_sym = (sub_all["abs_error"] / sub_all["market_price"]).mean() * 100.0
        spot_sym = ticker_stats[sym]["spot"]
        vol_sym = ticker_stats[sym]["realized_vol"] * 100.0
        ticker_breakdown.append({
            "Ticker": sym,
            "Spot": f"${spot_sym:.2f}",
            "Vol": f"{vol_sym:.1f}%",
            "Contracts": len(sub_all),
            "MAE ($)": mae_sym,
            "Rel Error (%)": rel_err_sym
        })
        print(f"  {sym:5s} (Spot ${spot_sym:7.2f}, Vol {vol_sym:4.1f}%): MAE = ${mae_sym:5.2f} | Rel Error = {rel_err_sym:4.1f}%")

    # 9. Plotting Benchmark Graphic (Dark Theme, No Emojis, No Grids)
    plt.style.use("dark_background")
    fig = plt.figure(figsize=(18, 5.5), facecolor="#000000")

    # Panel 1: Loss Convergence
    ax1 = fig.add_subplot(1, 3, 1)
    ax1.set_facecolor("#000000")
    ax1.plot(loss_history, color="#00e5ff", lw=1.8, label="Total Loss")
    ax1.plot(pde_history, color="#ff9100", lw=1.5, alpha=0.85, label="PDE Residual Loss")
    ax1.set_yscale("log")
    ax1.set_title("Training Loss Convergence (5000 Epochs)", color="#ffffff", fontsize=12, pad=12)
    ax1.set_xlabel("Epoch", color="#ffffff", fontsize=11)
    ax1.set_ylabel("Loss (Log Scale)", color="#ffffff", fontsize=11)
    ax1.legend(loc="upper right", frameon=False, labelcolor="#ffffff")
    ax1.grid(False)
    for spine in ax1.spines.values():
        spine.set_color("#ffffff")
        spine.set_linewidth(1.5)
    ax1.tick_params(colors="#ffffff", width=1.5)

    # Panel 2: Predicted Price vs Market Mid Price (Test Set)
    ax2 = fig.add_subplot(1, 3, 2)
    ax2.set_facecolor("#000000")
    colors = plt.cm.tab10(np.linspace(0, 1, len(ticker_stats)))
    for i, sym in enumerate(ticker_stats.keys()):
        sub = df_test[df_test["ticker"] == sym]
        if not sub.empty:
            ax2.scatter(sub["market_price"], sub["pred_price"], label=sym, s=36, alpha=0.85, color=colors[i])

    max_p = max(df_test["market_price"].max(), df_test["pred_price"].max())
    ax2.plot([0, max_p], [0, max_p], color="#ffffff", linestyle="--", lw=1.2, label="Parity Line (y = x)")
    ax2.set_title("Out-of-Sample Pricing: Model vs Market", color="#ffffff", fontsize=12, pad=12)
    ax2.set_xlabel("Actual Market Mid Price ($)", color="#ffffff", fontsize=11)
    ax2.set_ylabel("Universal PINN Price ($)", color="#ffffff", fontsize=11)
    ax2.legend(loc="upper left", frameon=False, fontsize=8, labelcolor="#ffffff", ncol=2)
    ax2.grid(False)
    for spine in ax2.spines.values():
        spine.set_color("#ffffff")
        spine.set_linewidth(1.5)
    ax2.tick_params(colors="#ffffff", width=1.5)

    # Panel 3: MAE Across S&P 500 Top 10
    ax3 = fig.add_subplot(1, 3, 3)
    ax3.set_facecolor("#000000")
    tickers = [tb["Ticker"] for tb in ticker_breakdown]
    maes = [tb["MAE ($)"] for tb in ticker_breakdown]
    bars = ax3.bar(tickers, maes, color="#7c4dff", edgecolor="#ffffff", lw=1.2, width=0.6)
    for bar in bars:
        yval = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2.0, yval + 0.1, f"${yval:.2f}", ha="center", va="bottom", color="#ffffff", fontsize=9)
    ax3.set_title("Mean Absolute Error Across Top 10 S&P 500", color="#ffffff", fontsize=12, pad=12)
    ax3.set_xlabel("Ticker", color="#ffffff", fontsize=11)
    ax3.set_ylabel("MAE ($)", color="#ffffff", fontsize=11)
    ax3.set_ylim(0, max(maes) * 1.25)
    ax3.grid(False)
    for spine in ax3.spines.values():
        spine.set_color("#ffffff")
        spine.set_linewidth(1.5)
    ax3.tick_params(colors="#ffffff", width=1.5)

    plt.tight_layout()
    os.makedirs("assets", exist_ok=True)
    chart_path = "assets/top10_sp500_benchmark.png"
    plt.savefig(chart_path, dpi=200, facecolor="#000000", edgecolor="none")
    plt.close()
    print(f"[OK] Benchmark chart saved to '{chart_path}'")

    # Save summary json
    summary_report = {
        "train_contracts": len(df_train),
        "test_contracts": len(df_test),
        "train_mae": float(train_mae),
        "train_rmse": float(train_rmse),
        "test_mae": float(test_mae),
        "test_rmse": float(test_rmse),
        "tickers": ticker_breakdown,
        "training_time_seconds": float(train_duration)
    }
    with open("data/top10_sp500_evaluation_metrics.json", "w") as f:
        json.dump(summary_report, f, indent=2)
    print(f"[OK] Metrics report saved to 'data/top10_sp500_evaluation_metrics.json'")

    return summary_report

if __name__ == "__main__":
    train_universal_pinn(epochs=5000)
