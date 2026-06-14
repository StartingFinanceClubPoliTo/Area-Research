from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import differential_evolution, minimize
from scipy.special import expit


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
    def intensity_on_grid(time_grid, event_times, lambda_bar, alpha, beta,
                          initial_intensity=None):
        """Evaluate the conditional intensity on a fixed time grid.

        ``lambda_bar`` is the mean-reversion baseline. ``initial_intensity``
        defaults to that baseline for backward compatibility.
        """
        grid = np.asarray(time_grid, dtype=float)
        events = np.asarray(event_times, dtype=float)
        lambda_initial = lambda_bar if initial_intensity is None else initial_intensity
        intensity = lambda_bar + (lambda_initial - lambda_bar) * np.exp(-beta * grid)

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
    def simulate_exponential(lambda_bar, alpha, beta, horizon, seed=None,
                             max_events=10000, initial_intensity=None):
        """Simulate an exponential Hawkes process with Ogata thinning."""
        lambda_initial = lambda_bar if initial_intensity is None else initial_intensity
        if lambda_bar <= 0 or lambda_initial <= 0 or alpha < 0 or beta <= 0 or horizon <= 0:
            raise ValueError(
                "lambda_bar, initial_intensity, beta, and horizon must be positive; "
                "alpha must be non-negative."
            )
        if alpha >= beta:
            raise ValueError("stationarity requires alpha < beta for the exponential kernel.")

        rng = np.random.default_rng(seed)
        event_times = []
        current_time = 0.0
        current_intensity = float(lambda_initial)

        while current_time < horizon and len(event_times) < max_events:
            upper_bound = max(current_intensity, lambda_bar)
            previous_time = current_time
            current_time += rng.exponential(1.0 / upper_bound)
            if current_time >= horizon:
                break

            decayed_intensity = lambda_bar + (current_intensity - lambda_bar) * np.exp(
                -beta * (current_time - previous_time)
            )

            if rng.uniform() <= decayed_intensity / upper_bound:
                event_times.append(current_time)
                current_intensity = decayed_intensity + alpha
            else:
                current_intensity = decayed_intensity

        return np.array(event_times, dtype=float)


@dataclass(frozen=True)
class HawkesCalibrationResult:
    """Compact, serializable result returned by both Hawkes calibrators."""

    model: str
    params: dict
    log_likelihood: float
    aic: float
    bic: float
    success: bool
    message: str
    n_events: int
    horizon: float

    def as_dict(self):
        """Return the result as a plain dictionary suitable for JSON export."""
        return asdict(self)


class _BaseHawkesCalibration:
    """Validation and result helpers shared by univariate Hawkes models."""

    @staticmethod
    def _prepare_events(event_times, horizon):
        events = np.asarray(event_times, dtype=float)
        if events.ndim != 1:
            raise ValueError("event_times must be a one-dimensional sequence.")
        if events.size == 0:
            raise ValueError("at least one event is required for calibration.")
        if not np.all(np.isfinite(events)):
            raise ValueError("event_times must contain only finite values.")

        events = np.sort(events)
        if events[0] < 0.0:
            raise ValueError("event_times must be non-negative.")
        if np.any(np.diff(events) <= 0.0):
            raise ValueError("event_times must be distinct for a simple point process.")

        if horizon is None:
            horizon = float(events[-1])
        horizon = float(horizon)
        if not np.isfinite(horizon) or horizon <= 0.0:
            raise ValueError("horizon must be a finite positive number.")
        if events[-1] > horizon:
            raise ValueError("all event_times must lie inside the observation horizon.")
        return events, horizon

    @staticmethod
    def _result(model, params, log_likelihood, optimizer_result, events, horizon):
        n_params = len([name for name in params if name != "branching_ratio"])
        n_events = int(events.size)
        return HawkesCalibrationResult(
            model=model,
            params={name: float(value) for name, value in params.items()},
            log_likelihood=float(log_likelihood),
            aic=float(2.0 * n_params - 2.0 * log_likelihood),
            bic=float(np.log(max(n_events, 1)) * n_params - 2.0 * log_likelihood),
            success=bool(optimizer_result.success),
            message=str(optimizer_result.message),
            n_events=n_events,
            horizon=float(horizon),
        )

    @staticmethod
    def _logit(value):
        value = float(np.clip(value, 1e-8, 1.0 - 1e-8))
        return np.log(value / (1.0 - value))


class ExponentialHawkesCalibration(_BaseHawkesCalibration):
    """Maximum-likelihood calibration for a univariate exponential Hawkes process.

    The conditional intensity is

        lambda(t) = lambda0 + sum(alpha * exp(-beta * (t - t_i))).

    The optimizer works with the branching ratio ``alpha / beta`` directly,
    which enforces the stationary region throughout the calibration.
    """

    model_name = "exponential"

    @staticmethod
    def branching_ratio(alpha, beta):
        if alpha < 0.0 or beta <= 0.0:
            raise ValueError("alpha must be non-negative and beta must be positive.")
        return float(alpha / beta)

    @staticmethod
    def kernel(lags, alpha, beta):
        lags = np.asarray(lags, dtype=float)
        values = np.where(lags >= 0.0, alpha * np.exp(-beta * lags), 0.0)
        return float(values) if values.ndim == 0 else values

    @staticmethod
    def _log_likelihood_prepared(events, horizon, lambda0, alpha, beta):
        if lambda0 <= 0.0 or alpha < 0.0 or beta <= 0.0 or alpha >= beta:
            return -np.inf

        excitation = 0.0
        log_intensity = np.log(lambda0)
        for index in range(1, events.size):
            delta = events[index] - events[index - 1]
            excitation = np.exp(-beta * delta) * (1.0 + excitation)
            intensity = lambda0 + alpha * excitation
            if intensity <= 0.0 or not np.isfinite(intensity):
                return -np.inf
            log_intensity += np.log(intensity)

        remaining = horizon - events
        compensator = lambda0 * horizon + (alpha / beta) * np.sum(
            1.0 - np.exp(-beta * remaining)
        )
        return float(log_intensity - compensator)

    @classmethod
    def log_likelihood(cls, event_times, horizon, lambda0, alpha, beta):
        events, horizon = cls._prepare_events(event_times, horizon)
        return cls._log_likelihood_prepared(events, horizon, lambda0, alpha, beta)

    @classmethod
    def fit(cls, event_times, horizon=None, initial_params=None, optimizer_options=None):
        """Fit ``lambda0``, ``alpha``, and ``beta`` by exact maximum likelihood."""
        events, horizon = cls._prepare_events(event_times, horizon)
        empirical_rate = events.size / horizon
        mean_gap = horizon / (events.size + 1.0)

        if initial_params is None:
            lambda0 = max(0.5 * empirical_rate, 1e-6)
            beta = max(1.0 / mean_gap, 1e-6)
            branching = 0.35
        else:
            lambda0 = float(initial_params["lambda0"])
            alpha = float(initial_params["alpha"])
            beta = float(initial_params["beta"])
            branching = cls.branching_ratio(alpha, beta)
            if not 0.0 < branching < 1.0:
                raise ValueError("initial alpha / beta must lie strictly between zero and one.")

        branching_scale = 0.999
        x0 = np.array(
            [np.log(lambda0), np.log(beta), cls._logit(branching / branching_scale)],
            dtype=float,
        )
        rate_floor = max(empirical_rate * 1e-5, 1e-10)
        rate_cap = max(empirical_rate * 100.0, 1.0 / horizon)
        beta_floor = max(1e-5 / horizon, 1e-10)
        beta_cap = max(1e5 / horizon, 1.0 / mean_gap)
        bounds = [
            (np.log(rate_floor), np.log(rate_cap)),
            (np.log(beta_floor), np.log(beta_cap)),
            (-12.0, 12.0),
        ]

        def unpack(vector):
            fitted_lambda0 = np.exp(vector[0])
            fitted_beta = np.exp(vector[1])
            fitted_branching = branching_scale * expit(vector[2])
            fitted_alpha = fitted_branching * fitted_beta
            return fitted_lambda0, fitted_alpha, fitted_beta

        def objective(vector):
            params = unpack(vector)
            value = cls._log_likelihood_prepared(events, horizon, *params)
            return 1e100 if not np.isfinite(value) else -value

        options = {"maxiter": 1000, "ftol": 1e-10}
        if optimizer_options:
            options.update(optimizer_options)
        optimized = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options=options)
        fitted_lambda0, fitted_alpha, fitted_beta = unpack(optimized.x)
        log_likelihood = cls._log_likelihood_prepared(
            events, horizon, fitted_lambda0, fitted_alpha, fitted_beta
        )
        params = {
            "lambda0": fitted_lambda0,
            "alpha": fitted_alpha,
            "beta": fitted_beta,
            "branching_ratio": fitted_alpha / fitted_beta,
        }
        return cls._result(cls.model_name, params, log_likelihood, optimized, events, horizon)

    calibrate = fit

    @staticmethod
    def intensity_on_grid(time_grid, event_times, lambda0, alpha, beta):
        return Hawkes.intensity_on_grid(time_grid, event_times, lambda0, alpha, beta)

    @classmethod
    def cumulative_compensator(cls, evaluation_times, event_times, lambda0, alpha, beta):
        evaluation_times = np.asarray(evaluation_times, dtype=float)
        events = np.asarray(event_times, dtype=float)
        values = np.empty_like(evaluation_times, dtype=float)
        for index, time in np.ndenumerate(evaluation_times):
            history = events[events < time]
            values[index] = lambda0 * time
            if history.size:
                values[index] += (alpha / beta) * np.sum(
                    1.0 - np.exp(-beta * (time - history))
                )
        return values

    @classmethod
    def time_rescaling_residuals(cls, event_times, lambda0, alpha, beta):
        events = np.sort(np.asarray(event_times, dtype=float))
        transformed = cls.cumulative_compensator(events, events, lambda0, alpha, beta)
        return np.diff(np.concatenate(([0.0], transformed)))

    @staticmethod
    def simulate(lambda0, alpha, beta, horizon, seed=None, max_events=10000):
        return Hawkes.simulate_exponential(
            lambda0, alpha, beta, horizon, seed=seed, max_events=max_events
        )


class RoughHawkesCalibration(_BaseHawkesCalibration):
    """Maximum-likelihood calibration for a regularized power-law Hawkes process.

    The rough kernel is parameterized as

        phi(t) = alpha / (t + cutoff) ** (1 + tail_index),

    where ``0 < tail_index < 1``. Its branching ratio is
    ``alpha / (tail_index * cutoff ** tail_index)``. The positive cutoff keeps
    the kernel finite at the origin while preserving its slow power-law tail.
    """

    model_name = "rough_power_law"

    @staticmethod
    def branching_ratio(alpha, tail_index, cutoff):
        if alpha < 0.0 or not 0.0 < tail_index < 1.0 or cutoff <= 0.0:
            raise ValueError(
                "alpha must be non-negative, cutoff positive, and tail_index in (0, 1)."
            )
        return float(alpha / (tail_index * cutoff ** tail_index))

    @staticmethod
    def kernel(lags, alpha, tail_index, cutoff):
        lags = np.asarray(lags, dtype=float)
        safe_lags = np.maximum(lags, 0.0)
        values = np.where(
            lags >= 0.0,
            alpha / np.power(safe_lags + cutoff, 1.0 + tail_index),
            0.0,
        )
        return float(values) if values.ndim == 0 else values

    @classmethod
    def _log_likelihood_prepared(
        cls, events, horizon, lambda0, alpha, tail_index, cutoff
    ):
        if lambda0 <= 0.0 or alpha < 0.0 or cutoff <= 0.0:
            return -np.inf
        if not 0.0 < tail_index < 1.0:
            return -np.inf
        branching = alpha / (tail_index * cutoff ** tail_index)
        if branching >= 1.0:
            return -np.inf

        log_intensity = np.log(lambda0)
        for index in range(1, events.size):
            lags = events[index] - events[:index]
            intensity = lambda0 + np.sum(cls.kernel(lags, alpha, tail_index, cutoff))
            if intensity <= 0.0 or not np.isfinite(intensity):
                return -np.inf
            log_intensity += np.log(intensity)

        remaining = horizon - events
        compensator = lambda0 * horizon + branching * np.sum(
            1.0 - np.power(cutoff / (cutoff + remaining), tail_index)
        )
        return float(log_intensity - compensator)

    @classmethod
    def log_likelihood(
        cls, event_times, horizon, lambda0, alpha, tail_index, cutoff
    ):
        events, horizon = cls._prepare_events(event_times, horizon)
        return cls._log_likelihood_prepared(
            events, horizon, lambda0, alpha, tail_index, cutoff
        )

    @classmethod
    def fit(cls, event_times, horizon=None, initial_params=None, optimizer_options=None):
        """Fit the baseline and rough-kernel parameters by exact likelihood."""
        events, horizon = cls._prepare_events(event_times, horizon)
        empirical_rate = events.size / horizon
        mean_gap = horizon / (events.size + 1.0)

        if initial_params is None:
            lambda0 = max(0.5 * empirical_rate, 1e-6)
            cutoff = max(0.1 * mean_gap, 1e-8)
            tail_index = 0.5
            branching = 0.35
        else:
            lambda0 = float(initial_params["lambda0"])
            alpha = float(initial_params["alpha"])
            tail_index = float(initial_params["tail_index"])
            cutoff = float(initial_params["cutoff"])
            branching = cls.branching_ratio(alpha, tail_index, cutoff)
            if not 0.0 < branching < 1.0:
                raise ValueError("initial rough-kernel branching ratio must be in (0, 1).")

        branching_scale = 0.999
        tail_floor = 0.02
        tail_width = 0.96
        x0 = np.array(
            [
                np.log(lambda0),
                np.log(cutoff),
                cls._logit(branching / branching_scale),
                cls._logit((tail_index - tail_floor) / tail_width),
            ],
            dtype=float,
        )
        rate_floor = max(empirical_rate * 1e-5, 1e-10)
        rate_cap = max(empirical_rate * 100.0, 1.0 / horizon)
        cutoff_floor = max(horizon * 1e-8, 1e-12)
        cutoff_cap = max(10.0 * horizon, mean_gap)
        bounds = [
            (np.log(rate_floor), np.log(rate_cap)),
            (np.log(cutoff_floor), np.log(cutoff_cap)),
            (-12.0, 12.0),
            (-8.0, 8.0),
        ]

        def unpack(vector):
            fitted_lambda0 = np.exp(vector[0])
            fitted_cutoff = np.exp(vector[1])
            fitted_branching = branching_scale * expit(vector[2])
            fitted_tail = tail_floor + tail_width * expit(vector[3])
            fitted_alpha = fitted_branching * fitted_tail * fitted_cutoff ** fitted_tail
            return fitted_lambda0, fitted_alpha, fitted_tail, fitted_cutoff

        def objective(vector):
            params = unpack(vector)
            value = cls._log_likelihood_prepared(events, horizon, *params)
            return 1e100 if not np.isfinite(value) else -value

        options = {"maxiter": 1500, "ftol": 1e-10}
        if optimizer_options:
            options.update(optimizer_options)
        optimized = minimize(objective, x0, method="L-BFGS-B", bounds=bounds, options=options)
        fitted_lambda0, fitted_alpha, fitted_tail, fitted_cutoff = unpack(optimized.x)
        log_likelihood = cls._log_likelihood_prepared(
            events,
            horizon,
            fitted_lambda0,
            fitted_alpha,
            fitted_tail,
            fitted_cutoff,
        )
        params = {
            "lambda0": fitted_lambda0,
            "alpha": fitted_alpha,
            "tail_index": fitted_tail,
            "cutoff": fitted_cutoff,
            "branching_ratio": cls.branching_ratio(
                fitted_alpha, fitted_tail, fitted_cutoff
            ),
        }
        return cls._result(cls.model_name, params, log_likelihood, optimized, events, horizon)

    calibrate = fit

    @classmethod
    def intensity_on_grid(
        cls, time_grid, event_times, lambda0, alpha, tail_index, cutoff
    ):
        grid = np.asarray(time_grid, dtype=float)
        events = np.asarray(event_times, dtype=float)
        intensity = np.full_like(grid, float(lambda0), dtype=float)
        for event_time in events:
            mask = grid >= event_time
            intensity[mask] += cls.kernel(
                grid[mask] - event_time, alpha, tail_index, cutoff
            )
        return intensity

    @classmethod
    def cumulative_compensator(
        cls, evaluation_times, event_times, lambda0, alpha, tail_index, cutoff
    ):
        evaluation_times = np.asarray(evaluation_times, dtype=float)
        events = np.asarray(event_times, dtype=float)
        branching = cls.branching_ratio(alpha, tail_index, cutoff)
        values = np.empty_like(evaluation_times, dtype=float)
        for index, time in np.ndenumerate(evaluation_times):
            history = events[events < time]
            values[index] = lambda0 * time
            if history.size:
                remaining = time - history
                values[index] += branching * np.sum(
                    1.0 - np.power(cutoff / (cutoff + remaining), tail_index)
                )
        return values

    @classmethod
    def time_rescaling_residuals(
        cls, event_times, lambda0, alpha, tail_index, cutoff
    ):
        events = np.sort(np.asarray(event_times, dtype=float))
        transformed = cls.cumulative_compensator(
            events, events, lambda0, alpha, tail_index, cutoff
        )
        return np.diff(np.concatenate(([0.0], transformed)))

    @classmethod
    def simulate(
        cls,
        lambda0,
        alpha,
        tail_index,
        cutoff,
        horizon,
        seed=None,
        max_events=10000,
    ):
        """Simulate the monotone rough kernel with Ogata thinning."""
        if lambda0 <= 0.0 or horizon <= 0.0:
            raise ValueError("lambda0 and horizon must be positive.")
        branching = cls.branching_ratio(alpha, tail_index, cutoff)
        if branching >= 1.0:
            raise ValueError("stationarity requires a rough-kernel branching ratio below one.")

        rng = np.random.default_rng(seed)
        events = []
        current_time = 0.0
        upper_bound = float(lambda0)
        jump_at_zero = cls.kernel(0.0, alpha, tail_index, cutoff)

        while current_time < horizon and len(events) < max_events:
            current_time += rng.exponential(1.0 / upper_bound)
            if current_time >= horizon:
                break

            if events:
                lags = current_time - np.asarray(events)
                candidate_intensity = lambda0 + np.sum(
                    cls.kernel(lags, alpha, tail_index, cutoff)
                )
            else:
                candidate_intensity = float(lambda0)

            if rng.uniform() <= candidate_intensity / upper_bound:
                events.append(current_time)
                upper_bound = candidate_intensity + jump_at_zero
            else:
                upper_bound = candidate_intensity

        return np.asarray(events, dtype=float)


class ExactHawkesCalibration:
    """Option-surface calibration for the exact affine Bates-Hawkes pricer.

    The pricing engine remains in ``BatesHawkesExact.py``. This class owns the
    calibration objectives, parameter mappings, constraints, and optimizers so
    notebooks can use one calibration API from ``Hawkes.py``.
    """

    MAX_BRANCHING = 0.98

    @staticmethod
    def _pricer():
        from BatesHawkesExact import BatesHawkesExact

        return BatesHawkesExact

    @classmethod
    def objective_constvol(
        cls, params, df_market, S0, q=0.0, n_steps=None, cos_N=256
    ):
        """Vega-weighted objective for exact Hawkes with constant volatility."""
        sigma, lambda_bar, alpha, beta, mu_j, sigma_j = params
        if sigma <= 0 or lambda_bar <= 0 or beta <= 0 or sigma_j <= 0:
            return 1e8
        if alpha < 0 or alpha >= beta or alpha / beta >= cls.MAX_BRANCHING:
            return 1e8

        pricer = cls._pricer()
        error = 0.0
        count = 0
        for maturity, group in df_market.groupby("T"):
            strikes = group["K"].to_numpy(dtype=float)
            rate = float(group["rate"].iloc[0])
            model_prices = pricer.hawkes_price_constvol_cos(
                S0,
                strikes,
                float(maturity),
                sigma,
                lambda_bar,
                lambda_bar,
                alpha,
                beta,
                mu_j,
                sigma_j,
                rate,
                q,
                N=cos_N,
                n_steps=n_steps,
            )
            market_prices = group["price"].to_numpy(dtype=float)
            safe_vega = np.maximum(group["vega"].to_numpy(dtype=float), 1e-4)
            error += float(np.sum(((model_prices - market_prices) / safe_vega) ** 2))
            count += len(group)

        branching = alpha / beta
        penalty = 0.01 * branching ** 2 / max(1.0 - branching, 1e-4)
        return error / max(count, 1) + penalty

    @classmethod
    def calibrate_constvol(
        cls,
        df_market,
        S0,
        q=0.0,
        maxiter=30,
        popsize=8,
        n_steps=None,
        seed=None,
    ):
        """Calibrate the constant-volatility exact Hawkes option model."""
        bounds = [
            (1e-2, 2.0),
            (1e-2, 5.0),
            (0.0, 5.0),
            (1e-2, 8.0),
            (-0.5, 0.5),
            (1e-3, 0.6),
        ]
        constraints = (
            {"type": "ineq", "fun": lambda x: x[3] - x[2] - 1e-4},
        )
        result_global = differential_evolution(
            cls.objective_constvol,
            bounds=bounds,
            args=(df_market, S0, q, n_steps),
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-3,
            polish=False,
            seed=seed,
        )
        result_local = minimize(
            cls.objective_constvol,
            x0=result_global.x,
            args=(df_market, S0, q, n_steps),
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"ftol": 1e-6, "maxiter": 60},
        )
        return result_local

    @staticmethod
    def unpack_heston_params(params):
        """Map the eleven optimizer variables to named model parameters."""
        (
            v0,
            kappa,
            theta,
            xi,
            rho,
            lambda0,
            lambda_bar,
            branching,
            beta,
            mu_j,
            sigma_j,
        ) = (float(value) for value in params)
        alpha = branching * beta
        return {
            "v0": v0,
            "kappa": kappa,
            "theta": theta,
            "xi": xi,
            "rho": rho,
            "lambda0": lambda0,
            "lambda_bar": lambda_bar,
            "stationary_mean_intensity": lambda_bar / max(1.0 - branching, 1e-12),
            "branching_ratio": branching,
            "alpha": alpha,
            "beta": beta,
            "mu_J": mu_j,
            "sigma_J": sigma_j,
        }

    @classmethod
    def objective_heston(
        cls, params, df_market, S0, q=0.0, n_steps=None, cos_N=192
    ):
        """Vega-weighted objective for the full affine Heston-Hawkes model."""
        try:
            p = cls.unpack_heston_params(params)
        except (TypeError, ValueError):
            return 1e8

        if p["v0"] <= 0 or p["kappa"] <= 0 or p["theta"] <= 0 or p["xi"] <= 0:
            return 1e8
        if not -0.999 < p["rho"] < 0.999:
            return 1e8
        if p["lambda0"] <= 0 or p["lambda_bar"] <= 0 or p["beta"] <= 0:
            return 1e8
        if p["sigma_J"] <= 0:
            return 1e8
        if not 0.0 <= p["branching_ratio"] < cls.MAX_BRANCHING:
            return 1e8
        if p["alpha"] >= p["beta"]:
            return 1e8

        pricer = cls._pricer()
        error = 0.0
        count = 0
        try:
            for maturity, group in df_market.groupby("T"):
                strikes = group["K"].to_numpy(dtype=float)
                rate = float(group["rate"].iloc[0])
                model_prices = pricer.hawkes_price_cos(
                    S0,
                    strikes,
                    float(maturity),
                    p["v0"],
                    p["kappa"],
                    p["theta"],
                    p["xi"],
                    p["rho"],
                    p["lambda0"],
                    p["lambda_bar"],
                    p["alpha"],
                    p["beta"],
                    p["mu_J"],
                    p["sigma_J"],
                    rate,
                    q,
                    N=cos_N,
                    n_steps=n_steps,
                )
                if not np.all(np.isfinite(model_prices)):
                    return 1e8
                market_prices = group["price"].to_numpy(dtype=float)
                safe_vega = np.maximum(group["vega"].to_numpy(dtype=float), 1e-4)
                error += float(np.sum(((model_prices - market_prices) / safe_vega) ** 2))
                count += len(group)
        except (FloatingPointError, OverflowError, ValueError):
            return 1e8
        return error / max(count, 1)

    @classmethod
    def calibrate_heston(
        cls,
        df_market,
        S0,
        q=0.0,
        bates_seed=None,
        maxiter=25,
        popsize=6,
        n_steps=None,
        seed=None,
        global_cos_N=128,
        local_cos_N=192,
        min_branching=0.02,
    ):
        """Calibrate the full affine Heston-Hawkes option model in two stages."""
        if bates_seed is None:
            bates_seed = (
                0.07997,
                1.71643,
                0.04410,
                0.71955,
                0.22059,
                0.89639,
                -0.19071,
                0.10083,
            )
        bates_seed = np.asarray(bates_seed, dtype=float)
        if bates_seed.shape != (8,):
            raise ValueError("bates_seed must contain 8 Bates parameters.")
        if not 0.0 <= min_branching < 0.95:
            raise ValueError("min_branching must lie in [0, 0.95).")

        diffusion_seed = bates_seed[:5]
        jump_seed = bates_seed[5:]
        hawkes_bounds = [
            (1e-3, 5.0),
            (1e-3, 5.0),
            (min_branching, 0.95),
            (0.1, 12.0),
            (-0.5, 0.5),
            (1e-3, 0.6),
        ]

        def conditional_objective(hawkes_params):
            full = np.concatenate([diffusion_seed, hawkes_params])
            return cls.objective_heston(
                full, df_market, S0, q, n_steps, global_cos_N
            )

        result_global = differential_evolution(
            conditional_objective,
            bounds=hawkes_bounds,
            maxiter=maxiter,
            popsize=popsize,
            tol=1e-3,
            polish=False,
            seed=seed,
        )
        full_bounds = [
            (1e-4, 1.0),
            (0.1, 10.0),
            (1e-4, 1.0),
            (0.01, 8.0),
            (-0.99, 0.99),
            *hawkes_bounds,
        ]
        x0 = np.concatenate([diffusion_seed, result_global.x])
        bates_like = np.concatenate(
            [
                diffusion_seed,
                [
                    jump_seed[0],
                    jump_seed[0] * (1.0 - min_branching),
                    min_branching,
                    max(result_global.x[3], 1.0),
                    jump_seed[1],
                    jump_seed[2],
                ],
            ]
        )
        candidates = []
        for start in (x0, bates_like):
            candidates.append(
                minimize(
                    cls.objective_heston,
                    x0=start,
                    args=(df_market, S0, q, n_steps, local_cos_N),
                    method="SLSQP",
                    bounds=full_bounds,
                    options={"ftol": 1e-8, "maxiter": 80, "disp": False},
                )
            )
        return min(candidates, key=lambda result: float(result.fun))
