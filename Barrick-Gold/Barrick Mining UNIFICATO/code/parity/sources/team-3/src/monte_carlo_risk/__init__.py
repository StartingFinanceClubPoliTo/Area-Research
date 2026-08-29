"""Reusable Monte Carlo utilities for the Starting Finance Club PoliTo article."""

from .option_pricing import (
    black_scholes_call,
    black_scholes_put,
    monte_carlo_european_call,
    price_asian_call_mc,
    price_asian_call_qmc,
    price_up_and_out_call,
)
from .random_generators import (
    box_muller_from_uniforms,
    chi_square_uniform_test,
    empirical_correlation,
    inverse_exponential,
    lcg_uniform,
    qq_data_standard_normal,
    sample_mvnormal_cholesky,
    sample_mvnormal_pca,
)
from .returns import (
    conditional_value_at_risk,
    empirical_inverse_cdf,
    simulate_gaussian_mixture_returns,
    simulate_terminal_wealth,
    value_at_risk,
)
from .stochastic_processes import (
    risk_neutral_jump_drift,
    simulate_gbm_exact,
    simulate_gbm_multivariate,
    simulate_jump_diffusion_grid,
    simulate_jump_diffusion_jump_times,
)
from .variance_reduction import (
    antithetic_variates,
    control_variate_single,
    control_variates_multi,
    importance_sampling_normal_shift,
    stratified_sampling_uniform,
)

__all__ = [
    "antithetic_variates",
    "black_scholes_call",
    "black_scholes_put",
    "box_muller_from_uniforms",
    "chi_square_uniform_test",
    "conditional_value_at_risk",
    "control_variate_single",
    "control_variates_multi",
    "empirical_correlation",
    "empirical_inverse_cdf",
    "importance_sampling_normal_shift",
    "inverse_exponential",
    "lcg_uniform",
    "monte_carlo_european_call",
    "price_asian_call_mc",
    "price_asian_call_qmc",
    "price_up_and_out_call",
    "qq_data_standard_normal",
    "risk_neutral_jump_drift",
    "sample_mvnormal_cholesky",
    "sample_mvnormal_pca",
    "simulate_gaussian_mixture_returns",
    "simulate_gbm_exact",
    "simulate_gbm_multivariate",
    "simulate_jump_diffusion_grid",
    "simulate_jump_diffusion_jump_times",
    "simulate_terminal_wealth",
    "stratified_sampling_uniform",
    "value_at_risk",
]
