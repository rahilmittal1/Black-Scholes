1. Black scholes model:
Numba-accelerated core (with a custom norm_cdf) for performance.
Includes standard pricing, American put approximation, Greeks (delta, gamma, vega, theta, rho), and a Newton–Raphson scheme for implied volatility.

2. Heston Model:
Monte Carlo simulation to price options under stochastic volatility.
The calibrate() method loops over synthetic market data (or real data) and minimizes the squared error between the model prices and market prices to “calibrate” the parameters.


3. Demonstration Block:
Shows how to compute prices and Greeks for Black–Scholes.
Prices an option using the Heston model.
Generates synthetic market data and calibrates the Heston model parameters against it.
