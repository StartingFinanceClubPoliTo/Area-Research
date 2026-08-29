
import numpy as np
import math as m_
from scipy.stats import norm


class BnS:
    """
    Black-Scholes European Call Option pricing and implied volatility calculator.
    
    Provides static methods for computing option prices and implied volatility using
    the standard Black-Scholes model with bisection method for IV convergence.
    """
    
    def __init__(self):
        pass

    @staticmethod
    def norm_cdf(x):
        """Cumulative distribution function for the standard normal distribution."""
        return 0.5 * (1.0 + m_.erf(x / np.sqrt(2.0)))

    @staticmethod
    def bs_call_price(S, K, T, r, sigma, q=0.0):
        """Calculate the Black-Scholes price of a European Call option.

        ### Parameters:    
        S: Spot price of the underlying asset <br>
        K: Strike price of the option <br>
        T: Time to maturity (in years) <br>
        r: Risk-free interest rate (annualized) <br>
        sigma: Volatility of the underlying asset (annualized) <br>
        q: Continuous dividend yield (annualized, default=0.0)
        
        ### Returns:
        Theoretical price of the European Call option as a float.
        """

        if T <= 0: return max(S - K, 0.0)
        if sigma <= 0: return max(S * np.exp(-q * T) - K * np.exp(-r * T), 0.0)
        
        sqrtT = np.sqrt(T)
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrtT)
        d2 = d1 - sigma * sqrtT
        return S * np.exp(-q * T) * BnS.norm_cdf(d1) - K * np.exp(-r * T) * BnS.norm_cdf(d2)

    @staticmethod
    def implied_vol_call(price, S, K, T, r, q=0.0, vol_low=1e-4, vol_high=5.0, tol=1e-8, max_iters=200):
        """Calculate Implied Volatility using the Bisection method.
        
        ### Parameters:
        price: Observed market price of the call option <br>
        S: Spot price of the underlying asset <br>
        K: Strike price of the option <br>
        T: Time to maturity (in years) <br>
        r: Risk-free interest rate (annualized) <br>
        q: Continuous dividend yield (annualized, default=0.0) <br>
        vol_low: Initial lower bound for volatility (default=1e-4) <br>
        vol_high: Initial upper bound for volatility (default=5.0) <br>
        tol: Tolerance for convergence (default=1e-8) <br>
        max_iters: Maximum iterations for the bisection method (default=200) <br>

        ### Returns:
        Implied volatility as a float if successful, or np.nan if no valid solution is found.  
        """

        if S <= 0 or K <= 0 or T <= 0 or price <= 0: return np.nan
        
        discounted_spot = S * np.exp(-q * T)
        discounted_strike = K * np.exp(-r * T)
        lower_bound = max(discounted_spot - discounted_strike, 0.0)
        
        # Check for arbitrage violations
        if price < lower_bound - 1e-10 or price > discounted_spot + 1e-10:
            return np.nan
            
        f_low = BnS.bs_call_price(S, K, T, r, vol_low, q) - price
        f_high = BnS.bs_call_price(S, K, T, r, vol_high, q) - price
        
        # Dynamically expand the upper bound if the root is not bracketed
        current_high = vol_high
        for _ in range(12):
            if f_low * f_high <= 0: break
            current_high *= 2.0
            f_high = BnS.bs_call_price(S, K, T, r, current_high, q) - price
        else:
            return np.nan
            
        a, b = vol_low, current_high
        fa, fb = f_low, f_high
        
        for _ in range(max_iters):
            m = 0.5 * (a + b)
            fm = BnS.bs_call_price(S, K, T, r, m, q) - price
            if abs(fm) < tol or (b - a) < tol: return m
            if fa * fm <= 0:
                b, fb = m, fm
            else:
                a, fa = m, fm
        return 0.5 * (a + b)
    
    @staticmethod
    def calculate_bs_vega(S, K, T, r, q, sigma):
        """
        Calculates the Black-Scholes Vega.
        S: Spot Price
        K: Strike Price
        T: Time to Maturity (Years)
        r: Risk-free rate
        q: Dividend yield
        sigma: Market Implied Volatility
        """
        # Prevent division by zero for expired options or zero vol
        if T <= 0.0 or sigma <= 0.0:
            return 1e-4 
            
        d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
        
        # norm.pdf is the standard normal probability density function N'(d1)
        vega = S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T)
        
        # Return Vega (floored at a small number to prevent division by zero downstream)
        return max(vega, 1e-4)


    

    
