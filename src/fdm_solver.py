"""
Crank-Nicolson / Implicit Finite Difference Method (FDM) PDE Solver
"""

import time
import numpy as np

def fdm_black_scholes(K: float, T: float, r: float, sigma: float, S_max: float = 300.0, M: int = 300, N: int = 1000):
    t0 = time.time()
    dS = S_max / M
    dt = T / N

    S_grid = np.linspace(0, S_max, M + 1)
    v = np.maximum(S_grid - K, 0.0)

    i_idx = np.arange(1, M)
    alpha = 0.5 * dt * (r * i_idx - (sigma ** 2) * (i_idx ** 2))
    beta  = 1.0 + dt * (r + (sigma ** 2) * (i_idx ** 2))
    gamma = -0.5 * dt * (r * i_idx + (sigma ** 2) * (i_idx ** 2))

    A = np.diag(beta) + np.diag(alpha[1:], -1) + np.diag(gamma[:-1], 1)

    for n in range(N):
        rhs = v[1:M]
        rhs[-1] -= gamma[-1] * (S_max - K * np.exp(-r * (n + 1) * dt))
        v[1:M] = np.linalg.solve(A, rhs)
        v[0] = 0.0
        v[M] = S_max - K * np.exp(-r * (n + 1) * dt)

    elapsed_ms = (time.time() - t0) * 1000.0
    return S_grid, v, elapsed_ms
