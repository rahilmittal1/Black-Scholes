# -*- coding: utf-8 -*-
"""
Option Pricing Models and Calibration
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy.stats as stats
from scipy.optimize import minimize
import time
import math
from numba import jit
from dataclasses import dataclass
from typing import Optional, Dict
import warnings
from tqdm import tqdm

warnings.filterwarnings("ignore", category=RuntimeWarning)


# ---- Numba-friendly Norm CDF ----
@jit(nopython=True)
def norm_cdf(x):
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2)))


# ---- Black-Scholes Core with Numba ----
@jit(nopython=True)
def _black_scholes_core(S, K, T, r, q, sigma, option_type_flag):
    """
    Core Black-Scholes calculation optimized with Numba.
    
    option_type_flag: 1 for call, 0 for put.
    """
    if T <= 0:
        # Option has expired: return intrinsic value.
        if option_type_flag == 1:
            return max(0, S - K)
        else:
            return max(0, K - S)
    
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (
        sigma * math.sqrt(T)
    )
    d2 = d1 - sigma * math.sqrt(T)
    
    if option_type_flag == 1:  # Call
        return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    else:  # Put
        return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)


# ---- Option Contract Data Class ----
@dataclass
class OptionContract:
    S: float  # Underlying asset price.
    K: float  # Strike price.
    T: float  # Time to maturity (in years).
    r: float  # Risk-free interest rate.
    q: float  # Dividend yield.
    sigma: Optional[float] = None  # Implied volatility (for Black–Scholes).
    option_type: str = "call"
    exercise_style: str = "european"
    
    def __post_init__(self):
        self.option_type = self.option_type.lower()
        self.exercise_style = self.exercise_style.lower()
        if self.option_type not in ["call", "put"]:
            raise ValueError("Option type must be 'call' or 'put'")
        if self.exercise_style not in ["european", "american"]:
            raise ValueError("Exercise style must be 'european' or 'american'")


# ---- Advanced Black-Scholes Model ----
class BlackScholesModel:
    """
    Advanced Black–Scholes option pricing model with support for Greeks and 
    implied volatility calculation.
    """
    
    def __init__(self, use_numba: bool = True):
        self.use_numba = use_numba
    
    def price(self, option: OptionContract) -> float:
        """
        Calculate the option price using the Black–Scholes model.
        Uses an approximation for American puts.
        """
        if option.exercise_style == "american" and option.option_type == "put":
            return self._american_put_price(option)
        
        option_type_flag = 1 if option.option_type == "call" else 0
        
        if self.use_numba:
            return _black_scholes_core(
                option.S, option.K, option.T, option.r, option.q, option.sigma,
                option_type_flag
            )
        else:
            return self._black_scholes_standard(option)
    
    def _black_scholes_standard(self, option: OptionContract) -> float:
        if option.T <= 0:
            return max(0, option.S - option.K) if option.option_type == "call" \
                else max(0, option.K - option.S)
        d1 = (np.log(option.S / option.K) +
              (option.r - option.q + 0.5 * option.sigma**2) * option.T) \
             / (option.sigma * np.sqrt(option.T))
        d2 = d1 - option.sigma * np.sqrt(option.T)
        if option.option_type == "call":
            return option.S * np.exp(-option.q * option.T) * stats.norm.cdf(d1) - \
                   option.K * np.exp(-option.r * option.T) * stats.norm.cdf(d2)
        else:
            return option.K * np.exp(-option.r * option.T) * stats.norm.cdf(-d2) - \
                   option.S * np.exp(-option.q * option.T) * stats.norm.cdf(-d1)
    
    def _american_put_price(self, option: OptionContract) -> float:
        """
        Price American put options using the Barone-Adesi & Whaley approximation.
        """
        if option.T <= 0:
            return max(0, option.K - option.S)
        
        european_put = self._black_scholes_standard(option)
        a = 2 * (option.r - option.q) / option.sigma**2
        
        def find_critical_price(S_star):
            d1 = (np.log(S_star / option.K) + (option.r - option.q + 
                  0.5 * option.sigma**2) * option.T) / (option.sigma * np.sqrt(option.T))
            A2 = -S_star / (option.sigma**2) * (1 -
                  np.exp((option.r - option.q) * option.T) * stats.norm.cdf(d1))
            return (option.K - S_star) - european_put - A2 * S_star * (1 - (S_star / option.K) ** (-a))
        
        S_star_initial = option.K * 0.5
        res = minimize(lambda x: abs(find_critical_price(x[0])),
                       [S_star_initial],
                       bounds=[(0.01, option.K)])
        S_critical = res.x[0]
        
        if option.S > S_critical:
            return european_put
        else:
            d1 = (np.log(S_critical / option.K) + (option.r - option.q + 
                  0.5 * option.sigma**2) * option.T) / (option.sigma * np.sqrt(option.T))
            A2 = -(S_critical / (option.sigma**2)) * (1 -
                  np.exp((option.r - option.q) * option.T) * stats.norm.cdf(d1))
            return option.K - option.S + A2 * option.S * (1 - (option.S / S_critical)**(-a))
    
    def implied_volatility(self, option: OptionContract, market_price: float,
                           precision: float = 1e-8, max_iterations: int = 100) -> float:
        """
        Calculate the implied volatility using the Newton-Raphson method.
        """
        if market_price <= 0:
            raise ValueError("Market price must be positive")
        
        sigma = 0.3  # initial guess
        for i in range(max_iterations):
            option.sigma = sigma
            price = self.price(option)
            vega_val = self.vega(option)
            if abs(vega_val) < 1e-10:
                vega_val = 1e-10
            diff = market_price - price
            if abs(diff) < precision:
                return sigma
            sigma = sigma + diff / vega_val
            sigma = max(0.001, min(sigma, 5))
        warnings.warn(
            f"Implied volatility calculation did not converge after {max_iterations} iterations"
        )
        return sigma
    
    def delta(self, option: OptionContract) -> float:
        if option.T <= 0:
            if option.option_type == "call":
                return 1.0 if option.S > option.K else 0.0
            else:
                return -1.0 if option.S < option.K else 0.0
        d1 = (np.log(option.S / option.K) +
              (option.r - option.q + 0.5 * option.sigma**2) * option.T) \
             / (option.sigma * np.sqrt(option.T))
        if option.option_type == "call":
            return np.exp(-option.q * option.T) * stats.norm.cdf(d1)
        else:
            return np.exp(-option.q * option.T) * (stats.norm.cdf(d1) - 1)
    
    def gamma(self, option: OptionContract) -> float:
        if option.T <= 0:
            return 0.0
        d1 = (np.log(option.S / option.K) +
              (option.r - option.q + 0.5 * option.sigma**2) * option.T) \
             / (option.sigma * np.sqrt(option.T))
        return np.exp(-option.q * option.T) * stats.norm.pdf(d1) / (
            option.S * option.sigma * np.sqrt(option.T)
        )
    
    def vega(self, option: OptionContract) -> float:
        if option.T <= 0:
            return 0.0
        d1 = (np.log(option.S / option.K) +
              (option.r - option.q + 0.5 * option.sigma**2) * option.T) \
             / (option.sigma * np.sqrt(option.T))
        return option.S * np.exp(-option.q * option.T) * stats.norm.pdf(d1) * np.sqrt(option.T) / 100
    
    def theta(self, option: OptionContract) -> float:
        if option.T <= 0:
            return 0.0
        d1 = (np.log(option.S / option.K) +
              (option.r - option.q + 0.5 * option.sigma**2) * option.T) \
             / (option.sigma * np.sqrt(option.T))
        d2 = d1 - option.sigma * np.sqrt(option.T)
        if option.option_type == "call":
            theta = (-option.S * np.exp(-option.q * option.T) * stats.norm.pdf(d1) *
                     option.sigma/(2 * np.sqrt(option.T))
                     - option.r * option.K * np.exp(-option.r * option.T) *
                     stats.norm.cdf(d2)
                     + option.q * option.S * np.exp(-option.q * option.T) *
                     stats.norm.cdf(d1))
        else:
            theta = (-option.S * np.exp(-option.q * option.T) * stats.norm.pdf(d1) *
                     option.sigma/(2 * np.sqrt(option.T))
                     + option.r * option.K * np.exp(-option.r * option.T) *
                     stats.norm.cdf(-d2)
                     - option.q * option.S * np.exp(-option.q * option.T) *
                     stats.norm.cdf(-d1))
        return theta / 365  # daily theta
    
    def rho(self, option: OptionContract) -> float:
        if option.T <= 0:
            return 0.0
        d1 = (np.log(option.S / option.K) +
              (option.r - option.q + 0.5 * option.sigma**2) * option.T) \
             / (option.sigma * np.sqrt(option.T))
        d2 = d1 - option.sigma * np.sqrt(option.T)
        if option.option_type == "call":
            return option.K * option.T * np.exp(-option.r * option.T) * stats.norm.cdf(d2) / 100
        else:
            return -option.K * option.T * np.exp(-option.r * option.T) * stats.norm.cdf(-d2) / 100
    
    def calculate_all_greeks(self, option: OptionContract) -> Dict[str, float]:
        return {
            "delta": self.delta(option),
            "gamma": self.gamma(option),
            "vega": self.vega(option),
            "theta": self.theta(option),
            "rho": self.rho(option)
        }


# ---- Heston Model with Monte Carlo Pricing and Calibration ----
class HestonModel:
    """
    Heston stochastic volatility model for option pricing using Monte Carlo
    simulation and a simple calibration routine.
    """
    
    def __init__(self, num_paths: int = 5000, num_steps: int = 252):
        self.num_paths = num_paths
        self.num_steps = num_steps
    
    def price(
        self,
        option: OptionContract,
        kappa: float,
        theta: float,
        sigma_v: float,
        rho: float,
        v0: float,
    ) -> float:
        """
        Price an option using the Heston model via Monte Carlo simulation.
        
        Parameters:
            kappa: Mean reversion rate of variance.
            theta: Long-term variance level.
            sigma_v: Volatility of variance.
            rho: Correlation between asset returns and variance.
            v0: Initial variance.
        """
        dt = option.T / self.num_steps
        sqrt_dt = np.sqrt(dt)
        # Initialize paths
        S = np.full(self.num_paths, option.S)
        v = np.full(self.num_paths, v0)
        for i in range(self.num_steps):
            Z1 = np.random.normal(0, 1, self.num_paths)
            Z2 = rho * Z1 + np.sqrt(1 - rho**2) * np.random.normal(0, 1, self.num_paths)
            S = S * np.exp(
                (option.r - option.q - 0.5 * v) * dt + np.sqrt(v) * sqrt_dt * Z1
            )
            v = v + kappa * (theta - v) * dt + sigma_v * np.sqrt(v) * sqrt_dt * Z2
            v = np.maximum(v, 0)
        if option.option_type == "call":
            payoff = np.maximum(S - option.K, 0)
        else:
            payoff = np.maximum(option.K - S, 0)
        price = np.exp(-option.r * option.T) * np.mean(payoff)
        return price
    
    def calibrate(
        self,
        market_data: pd.DataFrame,
        option_template: OptionContract,
        initial_guess: Optional[list] = None,
    ) -> Dict[str, float]:
        """
        Calibrate the Heston model parameters to market data using least
        squares optimization.
        
        Parameters:
            market_data: DataFrame with columns: 'K', 'T', 'MarketPrice', and
                optionally 'option_type'.
            option_template: OptionContract with fixed S, r, q (T and K will be replaced).
            initial_guess: [kappa, theta, sigma_v, rho, v0]. Defaults to
                [1.0, 0.04, 0.3, -0.5, 0.04].
                
        Returns:
            A dictionary with calibrated parameter values.
        """
        if initial_guess is None:
            initial_guess = [1.0, 0.04, 0.3, -0.5, 0.04]
        
        def objective(params):
            kappa, theta, sigma_v, rho, v0 = params
            error = 0.0
            # Loop over each market option
            for idx, row in market_data.iterrows():
                opt = OptionContract(
                    S=option_template.S,
                    K=row["K"],
                    T=row["T"],
                    r=option_template.r,
                    q=option_template.q,
                    option_type=row.get("option_type", option_template.option_type),
                    exercise_style=option_template.exercise_style,
                )
                model_price = self.price(opt, kappa, theta, sigma_v, rho, v0)
                error += (model_price - row["MarketPrice"]) ** 2
            return error
        
        bounds = [
            (0.01, 10),     # kappa
            (0.0001, 1),    # theta
            (0.0001, 2),    # sigma_v
            (-0.999, 0.999),# rho
            (0.0001, 1)     # v0
        ]
        
        result = minimize(objective, initial_guess, bounds=bounds, method="L-BFGS-B")
        calibrated_params = {
            "kappa": result.x[0],
            "theta": result.x[1],
            "sigma_v": result.x[2],
            "rho": result.x[3],
            "v0": result.x[4],
        }
        return calibrated_params


# ---- Demonstration ----
if __name__ == "__main__":
    print("=== Black–Scholes Model ===")
    bs_model = BlackScholesModel(use_numba=False)  # Using standard routines here
    option_bs = OptionContract(
        S=100, K=105, T=0.5, r=0.05, q=0.02, sigma=0.1,
        option_type="call", exercise_style="european"
    )
    bs_price = bs_model.price(option_bs)
    print("Black–Scholes Call Price:", bs_price)
    greeks = bs_model.calculate_all_greeks(option_bs)
    print("Black–Scholes Greeks:", greeks)
    
    # Implied volatility example
    market_price = 5.0  # example market price
    implied_vol = bs_model.implied_volatility(option_bs, market_price)
    print("Implied Volatility:", implied_vol)
    
    print("\n=== Heston Model ===")
    heston_model = HestonModel(num_paths=3000, num_steps=252)
    option_heston = OptionContract(
        S=100, K=105, T=0.5, r=0.05, q=0.02,
        option_type="call", exercise_style="european"
    )
    # Price using Heston model with example parameters:
    # (kappa, theta, sigma_v, rho, v0)
    heston_price = heston_model.price(option_heston, kappa=1.0, theta=0.04,
                                      sigma_v=0.3, rho=-0.5, v0=0.04)
    print("Heston Model Call Price:", heston_price)
    
    # ---- Heston Calibration Example ----
    # Generate synthetic market data (10 options with varying strikes and maturities)
    market_options = []
    strikes = np.linspace(90, 110, 10)
    maturities = np.linspace(0.25, 1.0, 10)
    true_params = {"kappa": 1.5, "theta": 0.04, "sigma_v": 0.3, "rho": -0.7, "v0": 0.04}
    
    for K, T in zip(strikes, maturities):
        opt = OptionContract(
            S=100, K=K, T=T, r=0.05, q=0.02,
            option_type="call", exercise_style="european"
        )
        price = heston_model.price(opt, **true_params)
        market_options.append({
            "K": K, "T": T, "MarketPrice": price, "option_type": "call"
        })
    
    market_data_df = pd.DataFrame(market_options)
    print("\nSynthetic Market Data for Heston Calibration:")
    print(market_data_df)
    
    # Provide an option template (with fixed S, r, q) for calibration.
    option_template = OptionContract(
        S=100, K=105, T=0.5, r=0.05, q=0.02,
        option_type="call", exercise_style="european"
    )
    calibrated = heston_model.calibrate(market_data_df, option_template)
    print("\nCalibrated Heston Parameters:")
    print(calibrated)
