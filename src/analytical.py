"""
Closed-Form Black-Scholes Formula & Analytical Greeks
"""

import numpy as np
from scipy.stats import norm

def bs_call_price(S: np.ndarray, K: float, T: float, r: float, sigma: float) -> np.ndarray:
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)

def bs_call_delta(S: np.ndarray, K: float, T: float, r: float, sigma: float) -> np.ndarray:
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.cdf(d1)

def bs_call_gamma(S: np.ndarray, K: float, T: float, r: float, sigma: float) -> np.ndarray:
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))
