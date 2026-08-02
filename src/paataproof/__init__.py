"""Reusable exact computations for extremal Seidel quadratic forms."""

from .exact import (
    exact_minimum_m,
    iter_switching_normalized_seidel_matrices,
    normalized_seidel_count,
)
from .seidel import SeidelMatrix, evaluate_m

__all__ = [
    "SeidelMatrix",
    "evaluate_m",
    "exact_minimum_m",
    "iter_switching_normalized_seidel_matrices",
    "normalized_seidel_count",
]
