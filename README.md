# Deep Stochastic Volatility: Physics-Informed Neural Networks (PINNs) for Derivatives Pricing

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![CUDA](https://img.shields.io/badge/CUDA-GTX%201070-green.svg)](https://developer.nvidia.com/cuda-zone)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A quantitative research library implementing Physics-Informed Neural Networks (PINNs) in PyTorch to solve derivative pricing partial differential equations (PDEs) and variational inequalities:
- **Universal Multi-Asset Black-Scholes PINN**: Parameterized over normalized moneyness $m = S/K$, maturity $\tau$, and volatility $\sigma$, calibrated as a physics-regularized model on option chains across the Top 10 S&P 500 equities.
- **2D Heston Stochastic Volatility PDE**: Resolving the 2D operator with cross-derivative autograd $\frac{\partial^2 u}{\partial m \partial v}$, benchmarked against semi-analytical Fourier inversion.
- **American Option Free-Boundary Variational Inequality**: Solving the obstacle problem $\min(\mathcal{L}_{\text{BS}}[u], u - h(m)) = 0$ via continuous penalization, benchmarked against a 1,000-step Cox-Ross-Rubinstein (CRR) binomial tree.

---

## Pre-Trained Model Weights & Specifications

- **Universal Multi-Asset Weights**: [`weights/universal_pinn_sp10.pth`](weights/universal_pinn_sp10.pth)
- **2D Heston Stochastic Volatility Weights**: [`weights/heston_pinn.pth`](weights/heston_pinn.pth)
- **American Option Free-Boundary Weights**: [`weights/american_pinn.pth`](weights/american_pinn.pth)
- **Single-Asset NVDA Weights**: [`weights/pinn_bs_nvda.pth`](weights/pinn_bs_nvda.pth)
- **State Space**: Moneyness $m = S/K \in [0.40, 1.60]$, Maturity $\tau \in [0, 1.20]$, Volatility $\sigma \in [0.15, 0.60]$, Variance $v \in [0.01, 0.16]$
- **Architecture**: Deep MLP with 4 Hidden Layers $\times$ 128 Neurons (SiLU Activations)
- **Hardware Acceleration**: Automatic CUDA detection with native support for NVIDIA GPUs (benchmarked on NVIDIA GeForce GTX 1070 8GB).

---

## Critical Methodology, Scope & Baseline Comparisons

### 1. European Call Options: Analytical Black-Scholes vs PINN
For standard 1D European calls under constant volatility, the closed-form Black-Scholes formula evaluates in **0.17 ms** with exact mathematical precision ($0.00 error), outperforming any neural network in both speed and accuracy. The 1D Black-Scholes PDE serves strictly as a pedagogical sanity check to confirm that the neural network solver correctly reproduces the analytical solution.

### 2. 2D Heston Stochastic Volatility: Fourier Inversion vs PINN
- **Fourier Quadrature**: Semi-analytical Fourier inversion (`scipy.integrate.quad`) evaluates in **3.65 ms per contract** (~3,650 ms sequentially for 1,000 contracts).
- **Heston PINN Forward Pass**: Evaluates 1,000 contracts simultaneously in **1.02 ms (CPU) / 0.33 ms (GPU)**, with an MAE of **$0.2950** against the Fourier benchmark.
- **Critical Trade-Off**: The Heston PINN was trained on **fixed model parameters** ($\kappa = 2.0, \theta = 0.04, \xi = 0.30, \rho = -0.70, r = 0.045$). Training required **141.6 s (CPU) / 20.5 s (GPU)**. If market parameters change, the PINN must be retrained, whereas Fourier inversion or the COS method requires zero training time and evaluates arbitrary parameters immediately.

### 3. American Put Options: Binomial Trees vs PINN
- **Cox-Ross-Rubinstein (CRR) Tree**: An $N=1,000$-step tree requires 500,000 backward induction steps per contract (~14 ms/tree in sequential NumPy, ~30 ms/tree in Numba). For 1,000 contracts, sequential evaluation takes 14 to 30 seconds.
- **American PINN Forward Pass**: Evaluates 1,000 contracts simultaneously in **1.01 ms (CPU) / 0.37 ms (GPU)** with an MAE of **$0.8287** against the 1,000-step CRR benchmark, after an offline training phase of **87.1 s**.
- **Alternative Baselines**: Fast analytical approximations (e.g., Bjerksund-Stensland 2002) evaluate American options in ~0.02 ms with high precision.

---

## Loss Function Formulation & Error Decomposition

### Exact Loss Function of the Multi-Asset Model
The Universal Multi-Asset PINN is trained using a composite loss balancing physics adherence and empirical market calibration:

$$\mathcal{L}_{\text{total}} = w_{\text{PDE}} \mathcal{L}_{\text{PDE}} + w_{\text{IC}} \mathcal{L}_{\text{IC}} + w_{\text{Market}} \mathcal{L}_{\text{Market}}$$

where:
- $\mathcal{L}_{\text{PDE}} = \frac{1}{N_{\text{PDE}}} \sum_{i=1}^{N_{\text{PDE}}} \left[ \frac{\partial v}{\partial \tau} - \left( \frac{1}{2}\sigma^2 m^2 \frac{\partial^2 v}{\partial m^2} + r m \frac{\partial v}{\partial m} - r v \right) \right]^2$ on 4,000 synthetic collocation points ($w_{\text{PDE}} = 1.0$).
- $\mathcal{L}_{\text{IC}} = \frac{1}{N_{\text{IC}}} \sum_{i=1}^{N_{\text{IC}}} (v(m, 0, \sigma) - \max(m - 1, 0))^2$ on 1,000 payoff points ($w_{\text{IC}} = 20.0$).
- $\mathcal{L}_{\text{Market}} = \frac{1}{N_{\text{Train}}} \sum_{i=1}^{N_{\text{Train}}} (v(m_i, \tau_i, \sigma_i) - v_i^{\text{Market}})^2$ on 560 real option quotes ($w_{\text{Market}} = 10.0$).

Because $w_{\text{Market}} > 0$, the network acts as a **physics-regularized regression calibrator**, not a pure boundary-value PDE solver.

### Triangle Inequality & Error Breakdown (700 S&P 10 Contracts)
By the triangle inequality in metric spaces:

$$\| V_{\text{PINN}} - V_{\text{Market}} \| \le \| V_{\text{PINN}} - V_{\text{Analytical\_BS}} \| + \| V_{\text{Analytical\_BS}} - V_{\text{Market}} \|$$

| Evaluation Metric | Comparison | MAE ($) | RMSE ($) | Quantitative Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **Numerical PDE Approximation Error** | $V_{\text{PINN}} \text{ vs } V_{\text{Analytical\_BS}}(\sigma)$ | **$1.12** | **$1.55** | Difference between the network output and exact Black-Scholes at the exact same $\sigma$. Measures departure from pure PDE compliance. |
| **Structural Model Misspecification** | $V_{\text{Analytical\_BS}}(\sigma) \text{ vs } V_{\text{Market}}$ | **$1.69** | **$2.79** | Inherent gap of flat-volatility Black-Scholes against real options trading with implied volatility smile/skew. |
| **Total Calibration Error to Market** | $V_{\text{PINN}} \text{ vs } V_{\text{Market}}$ | **$1.34** | **$2.20** | Fit of the regularized PINN to market quotes (In-Sample: **$1.28**, Out-of-Sample: **$1.58**). |

> [!NOTE]
> Numerically, $\$1.34 \le \$1.12 + \$1.69 = \$2.81$. The PINN is closer to the market (\$1.34) than pure Black-Scholes (\$1.69) because the market data loss term pulls the network toward observed prices, trading off strict PDE compliance (\$1.12 discrepancy) to absorb part of the market skew.

> [!WARNING]
> **Validation Scope**: The 20% out-of-sample test set (MAE = \$1.58) corresponds to spatial interpolation on held-out strikes and maturities from the same cross-sectional market snapshot (September 8, 2026). This does not substitute for out-of-time (temporal) validation or leave-one-asset-out (LOAO) transfer testing.

---

## Standardized Numerical Benchmark Summary

*All benchmarks evaluated on a standardized batch of 1,000 contracts, with 50 warm-up runs, averaged over 100 timed iterations. Hardware: CPU vs CPU for primary ratios.*

| Pricing Solver / Method | Mean Abs Error (MAE) | Relative Error | CPU Inference (1,000 contracts) | GPU Inference (GTX 1070) | Greeks Computation |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Black-Scholes (Exact Closed-Form)** | **$0.0000** | **0.00%** | **0.17 ms** | N/A (NumPy CPU) | Direct analytical formula |
| **Universal PINN (Top 10 S&P 500)** | **$1.2800** (Train) | **4.45%** (Median) | **1.00 ms** | **0.37 ms** | **Exact via autograd** ($\Delta, \Gamma, \text{Vega}, \Theta$) |
| **Single-Asset PINN (NVDA baseline)**| **$1.5461** | **3.25%** | **0.57 ms** | **0.25 ms** | Exact via autograd ($\Delta, \Gamma$) |
| **Heston 2D PINN (Stochastic Vol)** | **$0.2950** (vs Fourier) | **1.14%** | **1.02 ms** | **0.33 ms** | **Exact via autograd** ($\Delta, \Gamma, \partial V / \partial v$) |
| **American Option PINN (Free Boundary)**| **$0.8287** (vs CRR 1k) | **1.45%** | **1.01 ms** | **0.37 ms** | **Exact via autograd** ($\Delta$, Optimal Exercise Flag) |
| **Heston Fourier Quadrature** | **$0.0000** (Reference) | **0.00%** | **3,650 ms** | N/A | Numerical quadrature (`quad`) |
| **Monte Carlo Simulation (50k Paths)** | **$0.0443** | **0.54%** | **84.57 ms** | ~12.50 ms | Bump-and-reprice (~350 ms, stochastic noise) |
| **Finite Difference (Crank-Nicolson 1D)**| **$0.0019** | **0.03%** | **724.22 ms** | N/A | Spatial grid discretization |
| **CRR Binomial Tree (1,000 Steps)** | **$0.0000** (Reference) | **0.00%** | **14,200 ms** | N/A | Sequential NumPy implementation |

*Speedup baseline (CPU vs CPU): The Universal PINN on CPU (1.00 ms) evaluates 1,000 contracts approximately 85x faster than 50k-path Monte Carlo (84.57 ms).*

---

## Empirical Performance Across Top 10 S&P 500 Equities

*Market snapshot as of September 8, 2026. Because stock prices range from \$225 to \$1,124, errors are reported both in dollars and normalized by spot price (basis points of spot, where 100 bps = 1.0%).*

| Ticker | Company Name | Spot Price ($) | Realized Vol (%) | Contracts | Dollar MAE ($) | Error in % of Spot (bps) | Median Rel Error (%) |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **AAPL** | Apple Inc. | $316.22 | 25.1% | 70 | **$0.61** | **19.4 bps** | 3.12% |
| **JPM** | JPMorgan Chase & Co. | $353.51 | 22.2% | 70 | **$0.87** | **24.7 bps** | 4.05% |
| **GOOGL** | Alphabet Inc. | $338.36 | 31.5% | 70 | **$0.89** | **26.4 bps** | 3.88% |
| **LLY** | Eli Lilly and Co. | $1,123.91 | 35.9% | 70 | **$3.00** | **26.7 bps** | 4.30% |
| **AMZN** | Amazon.com Inc. | $256.97 | 34.5% | 70 | **$0.71** | **27.7 bps** | 4.15% |
| **MSFT** | Microsoft Corp. | $493.95 | 32.5% | 70 | **$1.50** | **30.3 bps** | 4.45% |
| **NVDA** | NVIDIA Corp. | $225.73 | 38.2% | 70 | **$0.71** | **31.7 bps** | 4.52% |
| **META** | Meta Platforms Inc. | $613.48 | 39.0% | 70 | **$2.02** | **32.9 bps** | 4.80% |
| **TSLA** | Tesla Inc. | $368.16 | 47.6% | 70 | **$1.31** | **35.7 bps** | 5.20% |
| **AVGO** | Broadcom Inc. | $368.56 | 47.4% | 70 | **$1.74** | **47.2 bps** | 5.95% |
| **Average** | *Cross-Sectional Mean* | — | — | **700** | **$1.34** | **30.3 bps** | **4.45%** |

> [!NOTE]
> Normalized by spot, model accuracy is consistent across the entire universe: the mean error is **30.3 basis points of the underlying stock price** (0.30%). On Eli Lilly (`LLY`), the \$3.00 dollar MAE represents only **26.7 bps** of its \$1,123.91 spot price. Median relative error on option premium is **4.45%** across all 700 contracts (median is reported to prevent deep OTM penny options from distorting unweighted arithmetic averages).

---

## Visual Benchmarks & Analytical Summaries

### 1. PINN vs Black-Scholes vs FDM vs Monte Carlo Benchmark
*Evaluated on European Call Option (K = 100, T = 1.0 Year, r = 5.0%, σ = 20.0%)*

![PINN Benchmark Comparison](assets/pinn_benchmark_comparison.png)

**Analysis**: The PINN architecture converges to the Black-Scholes analytical solution without spatial grid discretization. Exact option Deltas ($\Delta = \partial V / \partial S$) are computed instantaneously using PyTorch automatic differentiation (`autograd`). In standardized CPU batch inference, the neural network evaluates 1,000 option prices in 1.00 ms, approximately 85x faster than 50,000-path Monte Carlo simulations (84.57 ms).

---

### 2. Universal Multi-Asset PINN: Top 10 S&P 500 Stocks Benchmark
*Calibrated across 700 Real Option Contracts on AAPL, MSFT, NVDA, AMZN, GOOGL, META, TSLA, JPM, LLY, and AVGO (Market snapshot: September 8, 2026)*

![Universal PINN S&P 500 Benchmark](assets/top10_sp500_benchmark.png)

**Analysis**: By parameterizing the PINN over dimensionless moneyness ($m = S/K$) and asset volatility ($\sigma$), a single universal network prices the entire cross-section of S&P 500 mega-caps across diverse volatility regimes ($22.2\%$ to $47.6\%$). The model cuts NVDA pricing error to $\$0.71$ (more than a 2x improvement over the single-asset baseline) and achieves an average error of 30.3 basis points of spot across the cross-sectional test set.

---

### 3. Structured Products Simulation Engine (Geometric Brownian Motion)
*Evaluated on Phoenix Autocallable Notes (100% Autocall Trigger, 60% Protection Barrier) and Asian Call Options (Market snapshot: August 4, 2026, NVDA Spot S0 = $217.56, σ = 45.1%)*

![Structured Products Benchmark](assets/structured_products_benchmark.png)

**Analysis**: The simulation engine in `src/structured_products.py` evaluates path-dependent structured payoffs using a standard Geometric Brownian Motion (GBM) Monte Carlo simulation ($dS_t = r S_t dt + \sigma S_t dW_t$) with 50,000 paths and daily discrete monitoring (252 steps). The Phoenix Autocall note simulation reveals a 71.1% early redemption probability and a 7.6% capital barrier breach rate at maturity, illustrating the behavior of discontinuous early exercise triggers and conditional coupons.

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
