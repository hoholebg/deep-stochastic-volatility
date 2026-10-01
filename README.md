# Deep Stochastic Volatility: Physics-Informed Neural Networks (PINNs) for Derivatives Pricing

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-GTX%201070-green.svg)](https://developer.nvidia.com/cuda-zone)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A quantitative research library implementing Physics-Informed Neural Networks (PINNs) in PyTorch to solve derivative pricing partial differential equations (PDEs) and variational inequalities:
- **Universal Multi-Asset Black-Scholes PDE**: Parameterized over normalized moneyness $m = S/K$, maturity $\tau$, and volatility $\sigma$, calibrated on option chains across the Top 10 S&P 500 equities.
- **2D Heston Stochastic Volatility PDE**: Resolving the 2D operator with cross-derivative autograd $\frac{\partial^2 u}{\partial m \partial v}$, benchmarked against semi-analytical Fourier inversion.
- **American Option Free-Boundary Variational Inequality**: Solving the obstacle problem $\min(\mathcal{L}_{\text{BS}}[u], u - h(m)) = 0$ via continuous penalization, benchmarked against a 1,000-step Cox-Ross-Rubinstein (CRR) binomial tree.

---

## Pre-Trained Model Weights & Model Specifications

- **Universal Multi-Asset Weights**: [`weights/universal_pinn_sp10.pth`](weights/universal_pinn_sp10.pth)
- **2D Heston Stochastic Volatility Weights**: [`weights/heston_pinn.pth`](weights/heston_pinn.pth)
- **American Option Free-Boundary Weights**: [`weights/american_pinn.pth`](weights/american_pinn.pth)
- **Single-Asset NVDA Weights**: [`weights/pinn_bs_nvda.pth`](weights/pinn_bs_nvda.pth)
- **State Space**: Moneyness $m = S/K \in [0.40, 1.60]$, Maturity $\tau \in [0, 1.20]$, Volatility $\sigma \in [0.15, 0.60]$, Variance $v \in [0.01, 0.16]$
- **Architecture**: Deep MLP with 4 Hidden Layers $\times$ 128 Neurons (SiLU Activations)
- **Hardware Acceleration**: Automatic CUDA detection with native support for NVIDIA GPUs (benchmarked on NVIDIA GeForce GTX 1070 8GB).

---

## Critical Methodology & Benchmark Scope

> [!NOTE]
> **Analytical Baseline vs PINN Scope**: For standard 1D European call options under constant volatility, the closed-form Black-Scholes formula evaluates in **0.17 ms** with exact analytical precision ($0.00 error), outperforming any neural network in both speed and numerical accuracy. The 1D Black-Scholes PDE serves strictly as a pedagogical validation test to verify that the neural network solver correctly reproduces the analytical solution.
>
> The actual quantitative utility of PINNs arises in problems where closed forms do not exist or where numerical methods scale unfavorably:
> 1. **2D Heston Stochastic Volatility**: Solving the 2D PDE with spot-variance correlation autograd cross-derivative $\frac{\partial^2 u}{\partial m \partial v}$ achieves an MAE of **$0.2950** against semi-analytical Fourier quadrature, with batch inference of **0.33 ms** (GTX 1070 GPU) / **1.02 ms** (CPU) vs **1,850 ms** for 2D Alternating Direction Implicit (ADI) finite differences.
> 2. **American Options (Free-Boundary Variational Inequality)**: Where no closed form exists, achieving an MAE of **$0.8287** against a 1,000-step Cox-Ross-Rubinstein binomial tree, evaluating in **0.37 ms** (GPU) / **1.01 ms** (CPU) vs **14,200 ms** for sequential CRR trees across 1,000 contracts.

---

## Error Decomposition: PDE Approximation vs Market Misspecification

To avoid conflating numerical approximation error with structural model error, the evaluation on the 700 S&P 10 option contracts is decomposed into two distinct components:

| Evaluation Metric | Comparison | MAE ($) | RMSE ($) | Description |
| :--- | :--- | :--- | :--- | :--- |
| **Numerical PDE Approximation Error** | $V_{\text{PINN}} \text{ vs } V_{\text{Analytical\_BS}}$ | **$1.12** | **$1.55** | Pure solver error: difference between the neural network output and exact Black-Scholes evaluated at the exact same $\sigma$. |
| **Structural Model Misspecification** | $V_{\text{Analytical\_BS}} \text{ vs } V_{\text{Market\_Mid}}$ | **$1.69** | **$2.79** | Structural gap: discrepancy of standard Black-Scholes with flat volatility against real market quotes displaying implied volatility skew. |
| **Total Calibration Error to Market** | $V_{\text{PINN}} \text{ vs } V_{\text{Market\_Mid}}$ | **$1.34** | **$2.20** | Combined fit of the regularized PINN to market mid-quotes (In-Sample: **$1.28**, Out-of-Sample: **$1.58**). |

> [!WARNING]
> **Validation Methodology Note**: The 20% out-of-sample test set corresponds to spatial interpolation on held-out strikes and maturities from the same cross-sectional market snapshot (September 8, 2026). While demonstrating spatial generalization across moneyness and maturity, cross-sectional evaluation does not substitute for out-of-time (temporal) validation or leave-one-asset-out (LOAO) transfer testing.

---

## Standardized Numerical Benchmark Summary

*All benchmarks evaluated on a standardized batch of 1,000 contracts, with 50 warm-up runs, averaged over 100 timed iterations.*

| Pricing Solver / Method | Mean Abs Error (MAE) | Relative Error | CPU Inference (1,000 contracts) | GPU Inference (GTX 1070) | Greeks Computation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Black-Scholes (Exact Closed-Form)** | **$0.0000** | **0.00%** | **0.17 ms** | N/A (NumPy CPU) | Direct analytical formula |
| **Universal PINN (Top 10 S&P 500)** | **$1.2800** (Train) | **1.85%** | **1.00 ms** | **0.37 ms** | **Exact via autograd** ($\Delta, \Gamma, \text{Vega}, \Theta$) |
| **Single-Asset PINN (NVDA baseline)**| **$1.5461** | **3.25%** | **0.57 ms** | **0.25 ms** | Exact via autograd ($\Delta, \Gamma$) |
| **Heston 2D PINN (Stochastic Vol)** | **$0.2950** (vs Fourier) | **1.14%** | **1.02 ms** | **0.33 ms** | **Exact via autograd** ($\Delta, \Gamma, \partial V / \partial v$) |
| **American Option PINN (Free Boundary)**| **$0.8287** (vs CRR 1k) | **1.45%** | **1.01 ms** | **0.37 ms** | **Exact via autograd** ($\Delta$, Optimal Exercise Flag) |
| **Monte Carlo Simulation (50k Paths)** | **$0.0443** | **0.54%** | **84.57 ms** | ~12.50 ms | Bump-and-reprice (~350 ms, stochastic noise) |
| **Finite Difference (Crank-Nicolson 1D)**| **$0.0019** | **0.03%** | **724.22 ms** | N/A | Spatial grid discretization |
| **CRR Binomial Tree (1,000 Steps)** | **$0.0000** (Reference) | **0.00%** | **14,200 ms** | N/A | Backward induction on 1,000 trees |

*Speedup baseline: The Universal PINN on CPU (1.00 ms) is approximately 85x faster than 50k-path Monte Carlo (84.57 ms), and 230x faster on GTX 1070 GPU (0.37 ms).*

---

## Visual Benchmarks & Analytical Summaries

### 1. PINN vs Black-Scholes vs FDM vs Monte Carlo Benchmark
*Evaluated on European Call Option (K = 100, T = 1.0 Year, r = 5.0%, σ = 20.0%)*

![PINN Benchmark Comparison](assets/pinn_benchmark_comparison.png)

**Analysis**: The PINN architecture converges directly to the Black-Scholes analytical solution without spatial grid discretization. Exact option Deltas ($\Delta = \partial V / \partial S$) are computed instantaneously using PyTorch automatic differentiation (`autograd`). In standardized batch inference, the neural network evaluates 1,000 option prices in 1.00 ms (CPU) / 0.37 ms (GPU), representing an approximate 85x speedup over 50,000-path Monte Carlo simulations (84.57 ms).

---

### 2. Universal Multi-Asset PINN: Top 10 S&P 500 Stocks Benchmark
*Calibrated across 700 Real Option Contracts on AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, JPM, LLY, and AVGO (Market snapshot: September 8, 2026)*

![Universal PINN S&P 500 Benchmark](assets/top10_sp500_benchmark.png)

**Analysis**: By parameterizing the PINN over dimensionless moneyness ($m = S/K$) and asset volatility ($\sigma$), a single universal network prices the entire cross-section of S&P 500 mega-caps across diverse volatility regimes ($22.2\%$ to $47.6\%$). The model cuts NVDA pricing error to $\$0.71$ (more than a 2x improvement over the single-asset baseline) and achieves an out-of-sample test MAE of $\$1.58$ on unseen market strikes and maturities from the same cross-sectional snapshot.

---

### 3. Structured Products & Path-Dependent Simulations
*Evaluated on Phoenix Autocallable Notes (100% Autocall Trigger, 60% Protection Barrier) and Asian Call Options (Market snapshot: August 4, 2026, NVDA Spot S0 = $217.56, σ = 45.1%)*

![Structured Products Benchmark](assets/structured_products_benchmark.png)

**Analysis**: Calibrated on market data for NVDA ($S_0 = \$217.56$, $\sigma = 45.1\%$), the numerical simulation engine evaluates multi-period path-dependent payoffs. The Phoenix Autocall note simulation reveals a 71.1% early redemption probability and a 7.6% capital barrier breach rate at maturity, illustrating the behavior of discontinuous early exercise triggers and conditional coupons.

---

### 4. Implied Volatility Surface Calibration
*Calibrated across Strikes K/S0 ∈ [80%, 120%] and Maturities T ∈ [1 Month, 1 Year]*

![Volatility Surface and Skew](assets/volatility_surface_skew.png)

**Analysis**: The local volatility surface $\sigma(K, T)$ captures the characteristic equity volatility smile and skew observed across short-dated options (1M) to long-dated maturities (1Y). By incorporating local volatility dynamics into the PINN residual loss, the network penalizes pricing inconsistencies across the strike-maturity grid, though soft PDE penalization does not mathematically guarantee hard absence of butterfly or calendar spread arbitrage.

---

### 5. 2D Heston Stochastic Volatility PINN Benchmark
*Benchmarked against Fourier Inversion Semi-Analytical Solution across Stochastic Volatility Regimes*

![Heston 2D PINN Benchmark](assets/heston_benchmark.png)

**Analysis**: The 2D Heston PINN resolves the full cross-derivative term $\rho \xi v m \frac{\partial^2 u}{\partial m \partial v}$ governing spot-variance correlation without spatial grid mesh errors. Benchmarked against semi-analytical Fourier inversion, the network attains an MAE of $\$0.2950$ across volatility regimes while computing exact automatic differentiation Deltas and variance sensitivities.

---

### 6. American Option Free-Boundary PINN and Early Exercise Premium (EEP)
*Benchmarked against a 1,000-Step Cox-Ross-Rubinstein (CRR) Binomial Tree*

![American Option Free-Boundary Benchmark](assets/american_option_benchmark.png)

**Analysis**: The American Option PINN models the early exercise problem via the Black-Scholes variational inequality $\min(\mathcal{L}_{\text{BS}}[u], u - h(m)) = 0$ using continuous obstacle penalization $\lambda \mathbb{E}[\max(0, h(m) - u)^2]$, encouraging the option price to respect the intrinsic payoff floor. Evaluated against a 1,000-step Cox-Ross-Rubinstein binomial tree, the network achieves an MAE of $\$0.8287$ and isolates an Early Exercise Premium of up to $\$4.35$ on in-the-money puts.

---

## Empirical Performance Across Top 10 S&P 500 Equities

*Market snapshot as of September 8, 2026. Spot prices and realized volatilities calculated over 252 trading days.*

| Ticker | Company Name | Spot Price ($) | Realized Vol (%) | Option Contracts | Mean Abs Error (MAE) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **AAPL** | Apple Inc. | $316.22 | 25.1% | 70 | **$0.61** |
| **NVDA** | NVIDIA Corp. | $225.73 | 38.2% | 70 | **$0.71** |
| **AMZN** | Amazon.com Inc. | $256.97 | 34.5% | 70 | **$0.71** |
| **JPM** | JPMorgan Chase & Co. | $353.51 | 22.2% | 70 | **$0.87** |
| **GOOGL** | Alphabet Inc. | $338.36 | 31.5% | 70 | **$0.89** |
| **TSLA** | Tesla Inc. | $368.16 | 47.6% | 70 | **$1.31** |
| **MSFT** | Microsoft Corp. | $493.95 | 32.5% | 70 | **$1.50** |
| **AVGO** | Broadcom Inc. | $368.56 | 47.4% | 70 | **$1.74** |
| **META** | Meta Platforms Inc. | $613.48 | 39.0% | 70 | **$2.02** |
| **LLY** | Eli Lilly and Co. | $1,123.91 | 35.9% | 70 | **$3.00** |

---

## Technical Notes & Reference Material

- **[Educational and Technical Guide](docs/educational_guide_pinns_quant.md)**: Mathematical derivation of the Black-Scholes PDE operator, PINN loss decomposition, and PyTorch `autograd` implementation.
- **[Technical Working Paper](docs/arxiv_paper_draft_pinn_derivatives.md)**: Research draft discussing physics-informed neural network architectures for path-dependent derivatives and stochastic volatility PDEs.

---

## Quickstart & Model Inference

```bash
# Install dependencies
pip install -r requirements.txt

# 1. Universal Multi-Asset PINN Inference (S&P 500 mega-caps, runs on CUDA if available)
python src/predict.py --model universal --ticker AAPL
python src/predict.py --model universal --ticker NVDA
python src/predict.py --model universal --ticker LLY

# 2. 2D Heston Stochastic Volatility PINN Inference
python src/predict.py --model heston --spot 100 --strike 100 --maturity 0.5 --variance 0.04

# 3. American Option Free-Boundary PINN Inference & Early Exercise Decision
python src/predict.py --model american --spot 90 --strike 100 --maturity 1.0

# 4. Retrain Models with CUDA acceleration
python scripts/train_top10_sp500.py
python scripts/train_heston_and_american.py
```
