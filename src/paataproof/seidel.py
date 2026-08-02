"""Representation and exact small-order evaluation of Seidel matrices."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from numbers import Integral
from typing import Iterable, Iterator, Sequence, overload


def _integer(value: object, *, location: str) -> int:
    """Return an ordinary integer, rejecting non-integral and Boolean values."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise TypeError(f"{location} must be an integer, got {value!r}")
    return int(value)


@dataclass(frozen=True, slots=True, init=False)
class SeidelMatrix(Sequence[tuple[int, ...]]):
    """An immutable symmetric matrix with zero diagonal and ±1 off diagonal."""

    rows: tuple[tuple[int, ...], ...]

    def __init__(self, rows: Iterable[Iterable[int]]) -> None:
        try:
            supplied_rows = tuple(rows)
        except TypeError as error:
            raise TypeError("rows must be an iterable of row iterables") from error

        if not supplied_rows:
            raise ValueError("a Seidel matrix must have positive order")

        normalized: list[tuple[int, ...]] = []
        for row_index, row in enumerate(supplied_rows):
            try:
                supplied_values = tuple(row)
            except TypeError as error:
                raise TypeError(f"row {row_index} must be iterable") from error
            normalized.append(
                tuple(
                    _integer(value, location=f"entry ({row_index}, {column_index})")
                    for column_index, value in enumerate(supplied_values)
                )
            )

        order = len(normalized)
        for row_index, row in enumerate(normalized):
            if len(row) != order:
                raise ValueError(
                    "a Seidel matrix must be square: "
                    f"row {row_index} has length {len(row)}, expected {order}"
                )

        for row_index in range(order):
            if normalized[row_index][row_index] != 0:
                raise ValueError(
                    "a Seidel matrix must have zero diagonal: "
                    f"entry ({row_index}, {row_index}) is "
                    f"{normalized[row_index][row_index]}"
                )
            for column_index in range(row_index + 1, order):
                upper = normalized[row_index][column_index]
                lower = normalized[column_index][row_index]
                if upper not in (-1, 1):
                    raise ValueError(
                        "off-diagonal Seidel entries must be -1 or 1: "
                        f"entry ({row_index}, {column_index}) is {upper}"
                    )
                if lower not in (-1, 1):
                    raise ValueError(
                        "off-diagonal Seidel entries must be -1 or 1: "
                        f"entry ({column_index}, {row_index}) is {lower}"
                    )
                if upper != lower:
                    raise ValueError(
                        "a Seidel matrix must be symmetric: "
                        f"entries ({row_index}, {column_index}) and "
                        f"({column_index}, {row_index}) differ"
                    )

        object.__setattr__(self, "rows", tuple(normalized))

    @property
    def order(self) -> int:
        """The number of rows and columns."""
        return len(self.rows)

    def __len__(self) -> int:
        return self.order

    @overload
    def __getitem__(self, index: int) -> tuple[int, ...]: ...

    @overload
    def __getitem__(self, index: slice) -> tuple[tuple[int, ...], ...]: ...

    def __getitem__(
        self, index: int | slice
    ) -> tuple[int, ...] | tuple[tuple[int, ...], ...]:
        return self.rows[index]

    def __iter__(self) -> Iterator[tuple[int, ...]]:
        return iter(self.rows)

    def quadratic_form(self, signs: Iterable[int]) -> int:
        """Evaluate ``xᵀAx`` exactly for a sign vector ``x``."""
        try:
            supplied_signs = tuple(signs)
        except TypeError as error:
            raise TypeError("signs must be an iterable") from error
        vector = tuple(
            _integer(value, location=f"sign {index}")
            for index, value in enumerate(supplied_signs)
        )

        if len(vector) != self.order:
            raise ValueError(
                f"sign vector has length {len(vector)}, expected {self.order}"
            )
        for index, value in enumerate(vector):
            if value not in (-1, 1):
                raise ValueError(f"sign {index} must be -1 or 1, got {value}")

        # Symmetry and the zero diagonal let us sum the strict upper triangle.
        return 2 * sum(
            self.rows[row][column] * vector[row] * vector[column]
            for row in range(self.order)
            for column in range(row + 1, self.order)
        )


def evaluate_m(matrix: SeidelMatrix) -> int:
    r"""Compute ``M(A) = max_x |xᵀAx|`` by exhaustive sign enumeration.

    This routine is intended for small matrix orders: it evaluates exactly
    ``2**(n - 1)`` vectors.  Fixing the first sign to +1 loses no values because
    a quadratic form has the same value at ``x`` and ``-x``.
    """
    if not isinstance(matrix, SeidelMatrix):
        raise TypeError("matrix must be a SeidelMatrix")

    best = 0
    for tail in product((-1, 1), repeat=matrix.order - 1):
        value = abs(matrix.quadratic_form((1, *tail)))
        if value > best:
            best = value
    return best
