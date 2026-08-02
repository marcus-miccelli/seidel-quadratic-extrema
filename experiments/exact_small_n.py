"""Regenerate compact exact minima for small orders.

From a source checkout on PowerShell:

    $env:PYTHONPATH='src'; python experiments/exact_small_n.py --max-order 6
    $env:PYTHONPATH='src'; python experiments/exact_small_n.py --max-order 6 --output results/exact_small_n.json

With no output path, the same compact JSON is written to standard output.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

from paataproof import exact_minimum_m, normalized_seidel_count


def exact_small_order_results(max_order: int) -> dict[str, Any]:
    """Compute a deterministic JSON-compatible exact-results record."""
    if isinstance(max_order, bool) or not isinstance(max_order, int):
        raise TypeError(f"max_order must be an integer, got {max_order!r}")
    if max_order < 1:
        raise ValueError(f"max_order must be positive, got {max_order}")

    orders = []
    for order in range(1, max_order + 1):
        minimum_m = exact_minimum_m(order)
        orders.append(
            {
                "minimum_m": minimum_m,
                "minimum_pair_sum_objective": minimum_m // 2,
                "normalized_matrix_count": normalized_seidel_count(order),
                "order": order,
            }
        )
    return {
        "normalization": {
            "first_row_and_column_off_diagonal_entries": 1,
            "permutation_class_canonicalization": False,
        },
        "orders": orders,
        "schema_version": 1,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Exhaustively compute exact small-order Seidel minima."
    )
    parser.add_argument("--max-order", type=int, default=6)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args(argv)
    if arguments.max_order < 1:
        parser.error("--max-order must be positive")

    payload = exact_small_order_results(arguments.max_order)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
