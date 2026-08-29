
import pandas as pd
import numpy as np
from scipy.interpolate import Rbf

class Sampling:
    """Provides methods for sampling option data points from a full dataset based on different strategies:
    - Uniform Grid Sampling: Selects points closest to a uniform grid in (T, K) space.
    - Chebyshev Nodes Sampling: Selects points closest to Chebyshev nodes, which are denser near the boundaries of the (T, K) space.
    
    Each method ensures that the sampled points are actual market data points, not theoretical grid points, by finding the nearest neighbors in the original dataset.

    Provides also Error Evaluation: 
    - Reconstructs the implied volatility surface from the sampled points and evaluates the error against the full dataset using metrics like MAE, RMSE, and Max Error.   


    """
    
    def __init__(self):
        pass

    @staticmethod
    def get_nearest_market_points(df, target_T_array, target_K_array):
        """Find the closest real market options given a theoretical (T, K) grid."""
        df_sampled = pd.DataFrame()
        T_min, T_max = df['T'].min(), df['T'].max()
        K_min, K_max = df['K'].min(), df['K'].max()
        selected_indices = set()
        
        for t_target in target_T_array:
            for k_target in target_K_array:
                # Normalized squared Euclidean distance
                dist_sq = (((df['T'] - t_target) / (T_max - T_min))**2 + 
                        ((df['K'] - k_target) / (K_max - K_min))**2)
                nearest_idx = dist_sq.sort_values().index
                for idx in nearest_idx:
                    if idx not in selected_indices:
                        selected_indices.add(idx)
                        df_sampled = pd.concat([df_sampled, df.loc[[idx]]])
                        break
        return df_sampled.sort_values(['T', 'K']).reset_index(drop=True)

    @staticmethod
    def sample_uniform(df, n_T=8, n_K=8):
        """Extract options based on a uniformly spaced (T, K) grid."""
        target_T = np.linspace(df['T'].min(), df['T'].max(), n_T)
        target_K = np.linspace(df['K'].min(), df['K'].max(), n_K)
        return Sampling.get_nearest_market_points(df, target_T, target_K)

    @staticmethod
    def chebyshev_roots(n, a, b):
        """Calculate N Chebyshev nodes rescaled to the interval [a, b]."""
        k = np.arange(1, n + 1)
        roots_standard = np.sort(np.cos((2 * k - 1) * np.pi / (2 * n)))
        return 0.5 * (a + b) + 0.5 * (b - a) * roots_standard

    @staticmethod
    def sample_chebyshev(df, n_T=8, n_K=8):
        """Extract options based on Chebyshev nodes (denser at boundaries)."""
        target_T = Sampling.chebyshev_roots(n_T, df['T'].min(), df['T'].max())
        target_K = Sampling.chebyshev_roots(n_K, df['K'].min(), df['K'].max())
        return Sampling.get_nearest_market_points(df, target_T, target_K)
    
    @staticmethod
    def evaluate_sampling_error(df_full, df_sampled, method_name=""):
        """
        Reconstructs the surface using the sampled points and calculates 
        the error against the complete dataset.
        """
        # 1. Normalize T and K to prevent the price scale (K > 300) 
        # from dominating the time scale (T < 3) during spatial interpolation.
        T_min, T_max = df_full['T'].min(), df_full['T'].max()
        K_min, K_max = df_full['K'].min(), df_full['K'].max()
        
        T_sample_norm = (df_sampled['T'] - T_min) / (T_max - T_min)
        K_sample_norm = (df_sampled['K'] - K_min) / (K_max - K_min)
        IV_sample = df_sampled['implied_vol']
        
        # 2. Create the interpolation model (Radial Basis Function)
        rbf_interpolator = Rbf(T_sample_norm, K_sample_norm, IV_sample, function='thin_plate')
        
        # 3. Make the model predict IVs for ALL points in the full dataset
        T_full_norm = (df_full['T'] - T_min) / (T_max - T_min)
        K_full_norm = (df_full['K'] - K_min) / (K_max - K_min)
        IV_true = df_full['implied_vol']
        
        IV_pred = rbf_interpolator(T_full_norm, K_full_norm)
        
        # 4. Calculate error metrics (in percentage)
        error_pct = np.abs(IV_true - IV_pred) * 100
        mae = np.mean(error_pct)
        rmse = np.sqrt(np.mean(error_pct**2))
        max_err = np.max(error_pct)
        
        print(f"--- Error Metrics: {method_name} ---")
        print(f"MAE (Mean Absolute Error) : {mae:.3f}%")
        print(f"RMSE (Root Mean Sq. Error)  : {rmse:.3f}%")
        print(f"Max Error                   : {max_err:.3f}%")
        print("-" * 45)
        
        return error_pct