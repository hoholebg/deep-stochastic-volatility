"""
Path-Dependent Structured Products Engine (Asian Options, Phoenix Autocalls, Barrier Notes)
"""

import time
import numpy as np

class StructuredProductPricer:
    def __init__(self, S0: float, K: float, T: float, r: float, sigma: float):
        self.S0 = S0
        self.K = K
        self.T = T
        self.r = r
        self.sigma = sigma

    def price_asian_option_mc(self, n_sims: int = 50000, n_steps: int = 252) -> dict:
        t0 = time.time()
        dt = self.T / n_steps
        drift = (self.r - 0.5 * self.sigma ** 2) * dt
        vol_dt = self.sigma * np.sqrt(dt)

        z = np.random.normal(0, 1, (n_sims, n_steps))
        log_paths = np.cumsum(drift + vol_dt * z, axis=1)
        paths = self.S0 * np.exp(np.column_stack([np.zeros(n_sims), log_paths]))
        
        path_averages = np.mean(paths, axis=1)
        payoffs = np.maximum(path_averages - self.K, 0.0)
        price = float(np.mean(payoffs) * np.exp(-self.r * self.T))
        elapsed_ms = (time.time() - t0) * 1000.0
        return {"price": price, "elapsed_ms": elapsed_ms}

    def price_phoenix_autocall_mc(self, autocall_barrier: float = 1.0, protection_barrier: float = 0.60, coupon_rate: float = 0.10, n_sims: int = 50000, n_steps: int = 252) -> dict:
        t0 = time.time()
        dt = self.T / n_steps
        z = np.random.normal(0, 1, (n_sims, n_steps))
        log_paths = np.cumsum((self.r - 0.5 * self.sigma**2) * dt + self.sigma * np.sqrt(dt) * z, axis=1)
        paths = self.S0 * np.exp(np.column_stack([np.zeros(n_sims), log_paths]))

        obs_steps = [63, 126, 189, 252]
        payoffs = np.zeros(n_sims)
        autocalled = np.zeros(n_sims, dtype=bool)

        for obs in obs_steps:
            active = ~autocalled
            triggered = active & (paths[:, obs] >= autocall_barrier * self.S0)
            payoffs[triggered] = (1.0 + coupon_rate * (obs / 252.0)) * np.exp(-self.r * (obs * dt))
            autocalled[triggered] = True

        unredeemed = ~autocalled
        final_prices = paths[:, -1]
        capital_protected = final_prices >= (protection_barrier * self.S0)

        payoffs[unredeemed & capital_protected] = (1.0 + coupon_rate * self.T) * np.exp(-self.r * self.T)
        payoffs[unredeemed & (~capital_protected)] = (final_prices[unredeemed & (~capital_protected)] / self.S0) * np.exp(-self.r * self.T)

        price = float(np.mean(payoffs))
        elapsed_ms = (time.time() - t0) * 1000.0
        return {"autocall_note_price": price, "elapsed_ms": elapsed_ms}
