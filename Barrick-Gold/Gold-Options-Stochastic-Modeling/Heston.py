import math as m_
from scipy.stats import norm
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, differential_evolution


class Heston:
    """
    Heston model implementation for option pricing and calibration.
    This class provides methods to compute the characteristic function and option prices 
    via numerical integration. It features a robust calibration routine utilizing an 
    Inverse-Vega Weighted Least Squares (WLS) approach to effectively fit the volatility surface.
    """
    
    def __init__(self):
        pass

    
    # --- A. UNCONDITIONALLY STABLE CHARACTERISTIC FUNCTION ---
    @staticmethod
    def heston_charfunc(u, S0, v0, kappa, theta, sigma, rho, T, r, q=0.0):
        """
        Computes the Heston characteristic function.
        Implements the 'Little Heston Trap' correction (Albrecher et al., 2007) using 
        the e^(-dT) formulation to avoid complex plane branch cut discontinuities and overflows.
        """
        alpha = -u**2 / 2 - 1j * u / 2
        beta = kappa - rho * sigma * 1j * u
        gamma = sigma**2 / 2
        
        # Principal square root ensures real part of d is always > 0
        d = np.sqrt(beta**2 - 4 * alpha * gamma)
        
        r_plus = (beta + d) / (2 * gamma)
        r_minus = (beta - d) / (2 * gamma)
        g = r_minus / r_plus
        
        # e^(-dT) safely decays to 0 for large inputs. It will NEVER overflow.
        exp_dT = np.exp(-d * T)
        
        C = kappa * (r_minus * T - (2 / sigma**2) * np.log((1 - g * exp_dT) / (1 - g)))
        D = r_minus * (1 - exp_dT) / (1 - g * exp_dT)
        
        # Return the characteristic function for ln(S_T)
        return np.exp(1j * u * np.log(S0) + C * theta + D * v0 + 1j * u * (r - q) * T)

    # --- B. FASTER SINGLE-INTEGRAL PRICING ---

    def _integrand_fast(u, S0, K, v0, kappa, theta, sigma, rho, T, r, q=0.0):
        """
        Single-integral formulation (Carr-Madan / Lewis approach) for the option pricing integrand.
        Reduces the standard two-probability (P1, P2) Heston integrals into a single evaluation 
        for a ~2x computational speedup.
        """
        args = (S0, v0, kappa, theta, sigma, rho, T, r, q)
        
        # Evaluates the CF at (u - i) and (u)
        cf_shifted = Heston.heston_charfunc(u - 1j, *args)
        cf_standard = Heston.heston_charfunc(u, *args)
        
        num = np.exp(-r * T) * (cf_shifted - K * cf_standard)
        den = 1j * u * K**(1j * u)
        return num / den

    def heston_price_fast(S0, K, T, v0, kappa, theta, sigma, rho, r, q=0.0):
        """
        Prices a European call option using Fourier inversion of the characteristic function.
        """
        real_val, err = quad(lambda u: np.real(Heston._integrand_fast(u, S0, K, v0, kappa, theta, sigma, rho, T, r, q)),
                            0.0, 100.0, limit=200, epsabs=1e-4, epsrel=1e-4)
        
        price = np.real((S0 * np.exp(-q*T) - K * np.exp(-r*T)) / 2 + real_val / np.pi)
        return float(max(price, 0.0))

    # --- C. VEGA-WEIGHTED OBJECTIVE FUNCTION ---

    @staticmethod
    def heston_objective(params, df_market, S0, q=0.0):
        """
        Objective function for the Heston model calibration.
        
        Calculates the error between market prices and model prices. Crucially, it uses 
        a Vega-weighted sum of squared errors rather than raw price differences.
        
        Why Vega-weighting?
        The use of Vega-weighting addresses the inherent bias of minimizing raw price
        errors, which heavily favors deep In-The-Money options due to their large dollar
        values while neglecting Out-Of-The-Money options. Although minimizing Implied
        Volatility errors directly is the preferred approach to accurately capture the
        Volatility Smile, recalculating the Implied Volatility for every option at each
        optimizer step is computationally prohibitive. Instead, we leverage a first-order
        Taylor expansion, which shows that the change in volatility is approximately equal
        to the change in price divided by Vega. By scaling the dollar price error using
        the Black-Scholes Vega, we obtain a fast and highly effective approximation of the
        Implied Volatility error. This technique ensures that both inexpensive OTM and
        expensive ITM options contribute appropriately to the calibration, allowing the
        model to robustly fit the entire volatility surface.
        """
        v0, kappa, theta, sigma, rho = params
        
        # Hard bounds to prevent negative physical parameters.
        if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0 or not (-0.999 <= rho <= 0.999):
            return 1e6
        if 2.0 * kappa * theta - sigma ** 2 < 0.0:
            return 1e6

        error = 0.0
        for _, row in df_market.iterrows():
            K = row['K']
            T = row['T']
            r = row['rate']
            market_price = row['price']
            market_vega = row['vega']
            
            model_price = Heston.heston_price_fast(S0, K, T, v0, kappa, theta, sigma, rho, r, q)
            
            # Vega-weighting to mimic Implied Volatility
            safe_vega = max(market_vega, 1e-4)
            error += ((model_price - market_price) / safe_vega)**2

        return error / len(df_market)

    # --- D. CALIBRATION WRAPPER ---

    @staticmethod
    def calibrate_heston(df_market, S0, q=0.0):
        """
        Calibrates the Heston model parameters using a two-step optimization process:
        1. Global search via ***Differential Evolution***.
        2. Local refinement via ***SLSQP***.

        This method first runs a global optimization (Differential Evolution) to 
        find a robust starting point, avoiding local minima. It then refines 
        the result using a local optimizer (SLSQP).
        
        The calibration aims to find the Heston parameters (v0, kappa, theta, sigma, rho)
        that minimize the Vega-weighted pricing error (see `heston_objective`).
        This approach efficiently approximates Implied Volatility fitting, 
        ensuring accurate capture of the Volatility Smile across strikes.
        """

        bounds = [
            (1e-4, 1.0),   # v0 
            (0.1, 10.0),   # kappa 
            (1e-4, 1.0),   # theta 
            (0.01, 15.0),  # sigma (Raised safely since our new math won't crash)
            (-0.99, 0.99)  # rho 
        ]

        print("[INFO] Starting Global Calibration (Differential Evolution)...")
        result_global = differential_evolution(
            Heston.heston_objective,
            bounds=bounds,
            args=(df_market, S0, q),
            maxiter=150,      
            popsize=10,      
            tol=1e-3,
            disp=True        
        )
        
        print("\n[INFO] Global minimum found. Starting Local Refinement (SLSQP)...")
        result_local = minimize(
            Heston.heston_objective,
            x0=result_global.x,
            args=(df_market, S0, q),
            method='SLSQP',
            bounds=bounds,
            options={'ftol': 1e-6, 'maxiter': 100}
        )

        print("\n=============================================")
        print("  OPTIMAL HESTON PARAMETERS (CALIBRATED)")
        print("=============================================")
        print(f"v0 (Initial Variance)      : {result_local.x[0]:.5f}")
        print(f"kappa (Mean Reversion)     : {result_local.x[1]:.5f}")
        print(f"theta (Long-Term Variance) : {result_local.x[2]:.5f}")
        print(f"sigma (Vol of Volatility)  : {result_local.x[3]:.5f}")
        print(f"rho (Correlation)          : {result_local.x[4]:.5f}")
        print("=============================================")
        
        return result_local.x
