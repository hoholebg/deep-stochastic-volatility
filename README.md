# 🌊 Deep Stochastic Volatility: PINNs, Neural SDEs & Volatility Skew Calibration

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![yfinance](https://img.shields.io/badge/yfinance-Market%20Data-blue.svg)](https://github.com/ranaroussi/yfinance)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Institutional quantitative research framework introducing **Physics-Informed Neural Networks (PINNs)** and **Neural Stochastic Differential Equations (Neural SDEs)** for pricing exotic options and path-dependent structured products, calibrated on real-world equity market data from Yahoo Finance (NVDA, TSLA, AAPL).

---

## 📸 Visual Showcase (3 Core Benchmarks)

### 1. ⚡ PINN vs Black-Scholes vs FDM vs Monte Carlo Benchmark
*Evaluated on European Call Option ($K=100, T=1	ext{y}, r=5\%, \sigma=20\%$)*

![PINN Benchmark Comparison](assets/pinn_benchmark_comparison.png)

---

### 2. 📊 Real Market Structured Products & Path-Dependent Options (NVDA)
*Evaluated on Phoenix Autocallable Notes ($100\%$ Autocall, $60\%$ Barrier) and Asian Call Options*

![Structured Products Benchmark](assets/structured_products_benchmark.png)

---

### 3. 🌊 Real Market Implied Volatility Surface & Skew Calibration (Dupire / Heston Fit)
*Calibrated across strikes $K \in [80\%, 120\%]$ and maturities $T \in [1	ext{m}, 1	ext{y}]$*

![Volatility Surface & Skew](assets/volatility_surface_skew.png)

---

## 📊 Numerical Performance Summary

| Pricing Solver / Method | Mean Abs Error (MAE) | Relative Error (%) | Batch Inference Time |
| :--- | :--- | :--- | :--- |
| **Black-Scholes (Exact Closed-Form)** | **$0.0000** | **0.00%** | **0.05 ms** |
| **PINN (Physics-Informed Neural Net)** | **$0.7172** | **3.25%** | **1.00 ms** *(Ultra Fast)* |
| **Finite Difference (Crank-Nicolson FDM)** | **$0.0027** | **0.03%** | **724.22 ms** |
| **Monte Carlo Simulation (30k Paths)** | **$0.0560** | **0.54%** | **183.26 ms** |

---

## 📚 Documentation & Research Papers
- 🎓 **[Educational & Technical Guide](docs/educational_guide_pinns_quant.md)**: Deep dive into Black-Scholes PDE, PINN loss functions, and PyTorch `autograd` Greeks.
- 📝 **[arXiv / SSRN Research Paper Draft](docs/arxiv_paper_draft_pinn_derivatives.md)**: Academic publication manuscript titled *"Deep Stochastic Volatility: Physics-Informed Neural Networks and Neural SDEs for Real-World Path-Dependent Derivatives"*.

---

## 🚀 Quickstart
```bash
pip install -r requirements.txt
python main.py
```
