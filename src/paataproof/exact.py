"""Exact exhaustive enumeration for small Seidel-matrix orders.

The normalization used here fixes every off-diagonal entry in the first row
and first column to ``+1``.  This chooses one representative from every
switching class when vertex labels are fixed.  It is not canonicalization
under simultaneous row/column permutations: permutation-equivalent
representatives can both be generated.
"""

from __future__ import annotations

from itertools import product
from numbers import Integral
from typing import Iterator

from .seidel import SeidelMatrix, evaluate_m


def _positive_order(order: object) -> int:
    """Validate and return a positive, non-Boolean integer order."""
    if isinstance(order, bool) or not isinstance(order, Integral):
        raise TypeError(f"order must be an integer, got {order!r}")
    checked = int(order)
    if checked < 1:
        raise ValueError(f"order must be positive, got {checked}")
    return checked


def normalized_seidel_count(order: int) -> int:
    """Return the number of labeled switching-normalized matrices of ``order``."""
    checked = _positive_order(order)
    free_entries = (checked - 1) * (checked - 2) // 2
    return 1 << free_entries


def iter_switching_normalized_seidel_matrices(
    order: int,
) -> Iterator[SeidelMatrix]:
    """Generate every first-row/column-off-diagonal-``+1`` matrix exactly once.

    With fixed vertex labels, switching by a diagonal sign matrix sends every
    Seidel matrix to a unique matrix in this normalization.  The remaining
    ``(n - 1)(n - 2)/2`` upper-triangular entries are enumerated independently,
    so this iterator has ``2**((n - 1)(n - 2)/2)`` elements.

    This is deliberately not permutation-class canonicalization.  Matrices
    related by a relabeling of vertices may occur as distinct elements.
    Validation happens when this function is called, before iteration begins.
    """
    checked = _positive_order(order)
    free_positions = tuple(
        (row, column)
        for row in range(1, checked)
        for column in range(row + 1, checked)
    )

    def generate() -> Iterator[SeidelMatrix]:
        for free_values in product((-1, 1), repeat=len(free_positions)):
            rows = [[1] * checked for _ in range(checked)]
            for index in range(checked):
                rows[index][index] = 0
            for (row, column), value in zip(
                free_positions, free_values, strict=True
            ):
                rows[row][column] = value
                rows[column][row] = value
            yield SeidelMatrix(rows)

    return generate()


def exact_minimum_m(order: int) -> int:
    r"""Return ``min_A M(A)`` exactly by exhaustive normalized enumeration.

    Switching leaves ``M(A)`` invariant, so the minimum over the normalized
    matrices is the minimum over all labeled Seidel matrices.  This small-order
    routine examines ``2**((n - 1)(n - 2)/2)`` matrices and, for each matrix,
    ``evaluate_m`` examines ``2**(n - 1)`` sign vectors.  Thus it performs
    ``2**(n(n - 1)/2)`` quadratic-form evaluations, each taking quadratic time
    in ``n``; the cost becomes prohibitive quickly.
    """
    checked = _positive_order(order)
    return min(
        evaluate_m(matrix)
        for matrix in iter_switching_normalized_seidel_matrices(checked)
    )
