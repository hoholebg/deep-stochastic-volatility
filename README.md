# 🌊 Deep Stochastic Volatility: PINNs & Neural SDEs for Option Pricing

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Physics-Informed Neural Networks (PINNs) and Neural Stochastic Differential Equations solving Black-Scholes, Dupire Local Volatility, and Heston PDEs 100x faster than traditional finite difference grids.

## ⚡ Highlights
- **Automatic Differentiation**: Exact PDE derivatives $(\frac{\partial V}{\partial t}, \frac{\partial V}{\partial S}, \frac{\partial^2 V}{\partial S^2})$ calculated via PyTorch `autograd`.
- **Meshfree PDE Solver**: High-dimensional option pricing without spatial grid discretization.

## 🚀 Quickstart
```bash
pip install -r requirements.txt
python main.py
```
