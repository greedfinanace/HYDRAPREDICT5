"""Module 8 stress-testing interfaces."""

from .stress_test_engine import (
    FeeSensitivityResult,
    GapScenarioResult,
    MonteCarloStressConfig,
    MonteCarloStressResult,
    max_drawdown_from_returns,
    regime_shift_adaptation_latency,
    run_fee_sensitivity,
    simulate_gap_scenarios,
    simulate_monte_carlo_max_drawdown,
)

__all__ = [
    "FeeSensitivityResult",
    "GapScenarioResult",
    "MonteCarloStressConfig",
    "MonteCarloStressResult",
    "max_drawdown_from_returns",
    "regime_shift_adaptation_latency",
    "run_fee_sensitivity",
    "simulate_gap_scenarios",
    "simulate_monte_carlo_max_drawdown",
]
