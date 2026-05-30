import math as m_
from scipy.stats import norm
import numpy as np
from scipy.integrate import quad
from scipy.optimize import minimize, differential_evolution
from Heston import Heston


class Bates(Heston):
    """
    Bates Model (1996) implementation for option pricing and calibration.
    Inherits from the Heston class to utilize its stabilized characteristic function.
    """
    
    def __init__(self):
        pass

    # --- A. BATES CHARACTERISTIC FUNCTION ---
    @staticmethod
    def bates_charfunc(u, S0, v0, kappa, theta, sigma, rho, lambd, mu_J, sigma_J, T, r, q=0.0):
        """
        Computes the Bates characteristic function by multiplying the Heston CF 
        with the Merton Jump CF, including the risk-neutral drift compensator.
        """
        # 1. Base Heston CF (inherited from the parent class)
        cf_heston = Heston.heston_charfunc(u, S0, v0, kappa, theta, sigma, rho, T, r, q)
        
        # 2. Risk-Neutral Drift Correction (Compensator)
        # Prevents arbitrage by offsetting the expected mean of the jumps
        omega = -lambd * (np.exp(mu_J + 0.5 * sigma_J**2) - 1.0)
        
        # 3. Poisson Jump Component (Log-Normal)
        jump_term = lambd * (np.exp(1j * u * mu_J - 0.5 * u**2 * sigma_J**2) - 1.0)
        
        # 4. Jump Characteristic Function
        cf_jump = np.exp((1j * u * omega + jump_term) * T)
        
        # 5. Total CF = Heston CF * Jump CF
        return cf_heston * cf_jump

    # --- B. FASTER SINGLE-INTEGRAL PRICING FOR BATES ---
    @staticmethod
    def _integrand_fast_bates(u, S0, K, v0, kappa, theta, sigma, rho, lambd, mu_J, sigma_J, T, r, q=0.0):
        """
        Single-integral formulation (Carr-Madan / Lewis) adapted for the Bates model.
        """
        args = (S0, v0, kappa, theta, sigma, rho, lambd, mu_J, sigma_J, T, r, q)
        
        # Evaluates the CF at (u - i) and (u)
        cf_shifted = Bates.bates_charfunc(u - 1j, *args)
        cf_standard = Bates.bates_charfunc(u, *args)
        
        # Applying the correct negative discount factor e^(-rT) to the entire difference
        num = np.exp(-r * T) * (cf_shifted - K * cf_standard)
        den = 1j * u * K**(1j * u)
        
        return num / den

    @staticmethod
    def bates_price_fast(S0, K, T, v0, kappa, theta, sigma, rho, lambd, mu_J, sigma_J, r, q=0.0):
        """
        Prices a European call option under the Bates model using Fourier inversion.
        """
        # Numerical integration using scipy.integrate.quad
        real_val, err = quad(
            lambda u: np.real(Bates._integrand_fast_bates(u, S0, K, v0, kappa, theta, sigma, rho, lambd, mu_J, sigma_J, T, r, q)),
            0.0, 100.0, limit=200, epsabs=1e-4, epsrel=1e-4
        )
        
        # Final Call Price formula
        price = np.real((S0 * np.exp(-q*T) - K * np.exp(-r*T)) / 2 + real_val / np.pi)
        return float(max(price, 0.0))

    # --- C. VEGA-WEIGHTED OBJECTIVE FUNCTION ---
    @staticmethod
    def bates_objective(params, df_market, S0, q=0.0):
        """
        Vega-weighted objective function for the Bates model.
        params array: [v0, kappa, theta, sigma, rho, lambd, mu_J, sigma_J]
        """
        v0, kappa, theta, sigma, rho, lambd, mu_J, sigma_J = params
        
        # Hard bounds to prevent negative physical parameters and invalid correlations
        if v0 <= 0 or kappa <= 0 or theta <= 0 or sigma <= 0 or not (-0.999 <= rho <= 0.999):
            return 1e6
        if lambd < 0 or sigma_J <= 0: # Jump intensity and volatility must be strictly positive
            return 1e6

        error = 0.0
        for _, row in df_market.iterrows():
            K = row['K']
            T = row['T']
            r = row['rate']
            market_price = row['price']
            market_vega = row['vega']
            
            # Use the fast Bates pricing method
            model_price = Bates.bates_price_fast(S0, K, T, v0, kappa, theta, sigma, rho, lambd, mu_J, sigma_J, r, q)
            
            # Vega-weighting to mimic Implied Volatility fitting
            safe_vega = max(market_vega, 1e-4)
            error += ((model_price - market_price) / safe_vega)**2

        return error / len(df_market)

    # --- D. CALIBRATION WRAPPER ---
    @staticmethod
    def calibrate_bates(df_market, S0, q=0.0):
        """
        Calibrates the 8 Bates model parameters using Differential Evolution + SLSQP.
        """
        bounds = [
            (1e-4, 1.0),   # 0: v0 (Initial Variance)
            (0.1, 10.0),   # 1: kappa (Mean Reversion)
            (1e-4, 1.0),   # 2: theta (Long-Term Variance)
            (0.01, 15.0),  # 3: sigma (Vol of Volatility)
            (-0.99, 0.99), # 4: rho (Correlation)
            (0.0, 5.0),    # 5: lambd (Jump Intensity: 0 to 5 jumps per year)
            (-0.5, 0.5),   # 6: mu_J (Mean jump size: -50% to +50%)
            (1e-4, 0.5)    # 7: sigma_J (Jump volatility: up to 50%)
        ]

        print("[INFO] Starting Global Calibration for Bates (Differential Evolution)...")
        
        result_global = differential_evolution(
            Bates.bates_objective,
            bounds=bounds,
            args=(df_market, S0, q),
            maxiter=150,      
            popsize=12,    # Slightly larger population needed for 8 parameters
            tol=1e-3,
            disp=True        
        )
        
        print("\n[INFO] Global minimum found. Starting Local Refinement (SLSQP)...")
        result_local = minimize(
            Bates.bates_objective,
            x0=result_global.x,
            args=(df_market, S0, q),
            method='SLSQP',
            bounds=bounds,
            options={'ftol': 1e-6, 'maxiter': 100}
        )

        print("\n=============================================")
        print("  OPTIMAL BATES PARAMETERS (CALIBRATED)")
        print("=============================================")
        print("--- Continuous Diffusion (Heston) ---")
        print(f"v0 (Initial Variance)      : {result_local.x[0]:.5f}")
        print(f"kappa (Mean Reversion)     : {result_local.x[1]:.5f}")
        print(f"theta (Long-Term Variance) : {result_local.x[2]:.5f}")
        print(f"sigma (Vol of Volatility)  : {result_local.x[3]:.5f}")
        print(f"rho (Correlation)          : {result_local.x[4]:.5f}")
        print("--- Jump Process (Merton) ---")
        print(f"lambda (Jump Intensity)    : {result_local.x[5]:.5f}")
        print(f"mu_J (Mean Jump Size)      : {result_local.x[6]:.5f}")
        print(f"sigma_J (Jump Volatility)  : {result_local.x[7]:.5f}")
        print("=============================================")
        
        return result_local.x