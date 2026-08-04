# 🌊 Deep Stochastic Volatility: Structured Products & Path-Dependent Neural SDEs

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![yfinance](https://img.shields.io/badge/yfinance-Market%20Data-blue.svg)](https://github.com/ranaroussi/yfinance)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Institutional quantitative pricing suite for **Complex Path-Dependent Derivatives (Asian Options, Phoenix Autocallable Notes, Barrier Options)** calibrated on **Real Market Data from Yahoo Finance** (NVDA, TSLA, AAPL, BTC-USD) using **PyTorch Neural SDEs** and **High-Precision Monte Carlo Simulations**.

---

## 📊 Real Market Numerical Results (NVDA: Spot $S_0 = \$211.94$, $\sigma = 36.5\%$)

| Derivative / Product | Pricing Model / Method | Fair Value ($) | Key Metric / Risk | Execution Time |
| :--- | :--- | :--- | :--- | :--- |
| **Asian Call Option (NVDA)** | **Monte Carlo (50k Paths)** | **$19.85** | StdErr: +/-$0.146 | **554.4 ms** |
| **Asian Call Option (NVDA)** | **PyTorch Neural SDE** | **$6.64** | MAE vs MC: $13.21 | **0.61 ms** |
| **Phoenix Autocall Note (NVDA)** | **Vectorized Monte Carlo** | **$0.99** | Autocall Prob: 71.1% | **554.4 ms** |

---

## 📈 Visual Benchmark & Payoff Profiles (Pure Black, Explicit Numeric Axes, No Grid)

![Real Market Structured Products Benchmark](assets/structured_products_benchmark.png)

### Key Highlights:
1. **Live Market Ingestion (`yfinance`)**: Automatically ingests historical spot prices, computes realized volatility $\sigma$, and retrieves option chains for equities ($NVDA, $TSLA, $AAPL) and crypto ($BTC-USD).
2. **Phoenix Autocall Structuring**: Simulates quarterly observation dates, early redemption triggers ($100\% S_0$), and Down-and-In capital protection barriers ($60\% S_0$).
3. **Neural SDE Path Operator**: Evaluates high-dimensional path averages and barrier conditions in parallel in **$< 1.5\text{ ms}$**.

## 🚀 Quickstart
```bash
pip install -r requirements.txt
python main.py
```
