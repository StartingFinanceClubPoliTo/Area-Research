import numpy as np


class Hawkes:
    """
    Exponential Hawkes process utilities for self-exciting jump arrivals.

    The process has conditional intensity

        lambda_t = lambda0 + sum_{tau_i < t} alpha exp(-beta (t - tau_i)).

    These helpers are intentionally separated from the Bates pricing class:
    Hawkes is a point-process layer, while Bates controls the option-pricing
    characteristic-function approximation used elsewhere in the project.
    """

    @staticmethod
    def branching_ratio(alpha, beta):
        """Return the exponential-kernel branching ratio alpha / beta."""
        if beta <= 0:
            raise ValueError("beta must be positive.")
        return alpha / beta

    @staticmethod
    def stationary_mean_intensity(lambda0, alpha, beta):
        """Return E[lambda_infinity] = lambda0 / (1 - alpha / beta)."""
        ratio = Hawkes.branching_ratio(alpha, beta)
        if not 0 <= ratio < 1:
            raise ValueError("stationarity requires 0 <= alpha / beta < 1.")
        return lambda0 / (1.0 - ratio)

    @staticmethod
    def intensity_on_grid(time_grid, event_times, lambda0, alpha, beta):
        """Evaluate the conditional intensity on a fixed time grid."""
        grid = np.asarray(time_grid, dtype=float)
        events = np.asarray(event_times, dtype=float)
        intensity = np.full_like(grid, fill_value=float(lambda0), dtype=float)

        for event_time in events:
            mask = grid >= event_time
            intensity[mask] += alpha * np.exp(-beta * (grid[mask] - event_time))

        return intensity

    @staticmethod
    def simulate_poisson(lambda_const, horizon, seed=None):
        """Simulate homogeneous Poisson event times on [0, horizon]."""
        if lambda_const < 0 or horizon <= 0:
            raise ValueError("lambda_const must be non-negative and horizon positive.")

        rng = np.random.default_rng(seed)
        if lambda_const == 0:
            return np.array([], dtype=float)

        event_times = []
        current_time = 0.0
        while current_time < horizon:
            current_time += rng.exponential(1.0 / lambda_const)
            if current_time < horizon:
                event_times.append(current_time)

        return np.array(event_times, dtype=float)

    @staticmethod
    def simulate_exponential(lambda0, alpha, beta, horizon, seed=None, max_events=10000):
        """
        Simulate an exponential Hawkes process with Ogata thinning.

        The implementation is compact and publication-facing. It is designed for
        diagnostics and reproducible examples, not ultra-high-frequency production
        calibration.
        """
        if lambda0 <= 0 or alpha < 0 or beta <= 0 or horizon <= 0:
            raise ValueError("lambda0, beta, and horizon must be positive; alpha must be non-negative.")
        if alpha >= beta:
            raise ValueError("stationarity requires alpha < beta for the exponential kernel.")

        rng = np.random.default_rng(seed)
        event_times = []
        current_time = 0.0
        current_intensity = float(lambda0)

        while current_time < horizon and len(event_times) < max_events:
            upper_bound = max(current_intensity, lambda0)
            previous_time = current_time
            current_time += rng.exponential(1.0 / upper_bound)
            if current_time >= horizon:
                break

            decayed_intensity = lambda0 + (current_intensity - lambda0) * np.exp(
                -beta * (current_time - previous_time)
            )

            if rng.uniform() <= decayed_intensity / upper_bound:
                event_times.append(current_time)
                current_intensity = decayed_intensity + alpha
            else:
                current_intensity = decayed_intensity

        return np.array(event_times, dtype=float)
