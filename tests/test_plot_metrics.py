"""Property-based tests for the plot_metrics post-processing script.

Uses Hypothesis to verify correctness properties defined in the design document. The ``ema_smooth`` function lives in ``tools/plot_metrics.py``. That module imports matplotlib, but it selects the non-interactive Agg backend, so it imports cleanly in a headless test environment.
"""
import math
import os
import sys

# Make the tools/ directory importable so we can import plot_metrics.
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "tools"
    ),
)

from hypothesis import given, settings
from hypothesis import strategies as st

from plot_metrics import ema_smooth


# =============================================================================
# EMA Smoothing Invariants
# =============================================================================
# For any non-empty time series of floats and smoothing factor `alpha` in (0, 1):
#   (a) the smoothed output SHALL have the same length as the input,
#   (b) the first element SHALL equal the first input element, and
#   (c) each subsequent element SHALL equal `alpha * previous_smoothed + (1 - alpha) * current_input`.
#
# =============================================================================

# Non-empty lists of finite, bounded floats.
finite_floats = st.floats(
    allow_nan=False, allow_infinity=False, min_value=-1e6, max_value=1e6
)
value_series = st.lists(finite_floats, min_size=1)

# Smoothing factor strictly inside (0, 1).
alpha_strategy = st.floats(min_value=0.001, max_value=0.999, allow_nan=False)


@given(values=value_series, alpha=alpha_strategy)
@settings(max_examples=200)
def test_ema_smoothing_invariants(values, alpha):
    """EMA Smoothing Invariants.

    For a non-empty series and alpha in (0, 1):
    (a) output length matches input length,
    (b) first element is unchanged,
    (c) the EMA recurrence holds for every subsequent element.

    """
    smoothed = ema_smooth(values, alpha)

    # (a) Same length as input.
    assert len(smoothed) == len(values)

    # (b) First element unchanged.
    assert smoothed[0] == values[0]

    # (c) Recurrence holds for all subsequent elements.
    for i in range(1, len(values)):
        expected = alpha * smoothed[i - 1] + (1 - alpha) * values[i]
        assert math.isclose(smoothed[i], expected, rel_tol=1e-9, abs_tol=1e-12)
