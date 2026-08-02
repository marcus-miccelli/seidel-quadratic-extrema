"""Audit scalar defects and selected exact two-block compositions.

The scalar part reads the evidence-labelled values in
``results/reported_exact_values.json``.  The bridge part is independent and
exhausts all rectangular sign bridges for four fixed pairs of children.
Fixing the first bridge entry to +1 is lossless because replacing a bridge by
its negative does not change its cap.

From a source checkout on PowerShell:

    python experiments/composition_small_orders.py
    python experiments/composition_small_orders.py --output results/composition_small_orders.json
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Sequence


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_VALUES = ROOT / "results" / "reported_exact_values.json"


def spins(order: int) -> list[tuple[int, ...]]:
    """Return one representative of each antipodal spin pair."""
    return [(1, *tail) for tail in itertools.product((-1, 1), repeat=order - 1)]


def signing(order: int, key: tuple[int, ...]) -> tuple[tuple[int, ...], ...]:
    """Build a first-row-positive Seidel matrix from its remaining entries."""
    expected = (order - 1) * (order - 2) // 2
    if len(key) != expected:
        raise ValueError(f"order {order} requires a key of length {expected}")
    rows = [[0 if row == column else 1 for column in range(order)]
            for row in range(order)]
    values = iter(key)
    for row in range(1, order):
        for column in range(row + 1, order):
            rows[row][column] = rows[column][row] = next(values)
    return tuple(tuple(row) for row in rows)


def energies(matrix: tuple[tuple[int, ...], ...]) -> list[int]:
    """Evaluate the pair-sum quadratic form on the antipodal half-cube."""
    return [
        sum(
            matrix[row][column] * spin[row] * spin[column]
            for row in range(len(matrix))
            for column in range(row + 1, len(matrix))
        )
        for spin in spins(len(matrix))
    ]


def h_relation(parent: int, left: int, right: int) -> tuple[str, int]:
    """Certify the sign of parent^(2/3)-left^(2/3)-right^(2/3).

    If z=parent^2, a=left^2, and b=right^2, equality after cubing is
    (z-a-b)^3=27abz.  Checking z-a-b first makes the polynomial comparison
    equivalent to the original comparison over nonnegative inputs.
    """
    z, a, b = parent * parent, left * left, right * right
    shifted = z - a - b
    polynomial = shifted**3 - 27 * a * b * z
    if shifted <= 0:
        return "less", polynomial
    if polynomial > 0:
        return "greater", polynomial
    if polynomial < 0:
        return "less", polynomial
    return "equal", polynomial


def bridge_optimum(
    left: tuple[tuple[int, ...], ...],
    right: tuple[tuple[int, ...], ...],
) -> tuple[int, int, int]:
    """Return the exact (+right cap, -right cap, rectangular optimum)."""
    m, n = len(left), len(right)
    xs, ys = spins(m), spins(n)
    left_energies, right_energies = energies(left), energies(right)
    states = [
        (
            abs(left_energies[i] + right_energies[j]),
            abs(left_energies[i] - right_energies[j]),
            tuple(x[row] * y[column] for row in range(m) for column in range(n)),
        )
        for i, x in enumerate(xs)
        for j, y in enumerate(ys)
    ]
    initial = m * n + max(map(abs, left_energies)) + max(map(abs, right_energies))
    best_plus = best_minus = initial
    best_rectangular = m * n
    for tail in itertools.product((-1, 1), repeat=m * n - 1):
        bridge = (1, *tail)
        cap_plus = cap_minus = cap_rectangular = 0
        for internal_plus, internal_minus, state in states:
            cross = abs(sum(entry * value for entry, value in zip(bridge, state)))
            cap_plus = max(cap_plus, internal_plus + cross)
            cap_minus = max(cap_minus, internal_minus + cross)
            cap_rectangular = max(cap_rectangular, cross)
            if (
                cap_plus >= best_plus
                and cap_minus >= best_minus
                and cap_rectangular >= best_rectangular
            ):
                break
        best_plus = min(best_plus, cap_plus)
        best_minus = min(best_minus, cap_minus)
        best_rectangular = min(best_rectangular, cap_rectangular)
    return best_plus, best_minus, best_rectangular


def load_reported_values(path: Path = DEFAULT_VALUES) -> tuple[dict[int, int], dict[int, str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    values = {row["order"]: row["F"] for row in payload["orders"]}
    statuses = {row["order"]: row["status"] for row in payload["orders"]}
    return values, statuses


def scalar_record(m: int, n: int, values: dict[int, int]) -> dict[str, Any]:
    parent, left, right = values[m + n], values[m], values[n]
    relation, polynomial = h_relation(parent, left, right)
    defect = parent ** (2.0 / 3.0) - left ** (2.0 / 3.0) - right ** (2.0 / 3.0)
    return {
        "comparable": n <= 2 * m,
        "defect_decimal": f"{defect:.12f}",
        "integer_polynomial": polynomial,
        "left_order": m,
        "parent_order": m + n,
        "relation": relation,
        "right_order": n,
    }


def composition_results(values_path: Path = DEFAULT_VALUES) -> dict[str, Any]:
    """Return the deterministic JSON-compatible composition audit."""
    values, statuses = load_reported_values(values_path)
    pairs = sorted(
        (
            scalar_record(m, n, values)
            for m in range(2, 16)
            for n in range(m, 16)
            if m + n <= 15
        ),
        key=lambda row: (row["parent_order"], row["left_order"], row["right_order"]),
    )
    positive = [row for row in pairs if row["relation"] == "greater"]
    comparable = [row for row in pairs if row["comparable"]]
    positive_comparable = [row for row in comparable if row["relation"] == "greater"]
    with_large_children = [
        row for row in positive_comparable
        if row["left_order"] >= 3 and row["right_order"] >= 3
    ]
    over_one = [row for row in positive_comparable if float(row["defect_decimal"]) > 1]
    largest = max(positive, key=lambda row: float(row["defect_decimal"]))

    representative_keys = {
        3: (-1,),
        4: (-1, -1, 1),
        5: (-1, -1, 1, 1, -1, 1),
    }
    representatives = {
        order: signing(order, key) for order, key in representative_keys.items()
    }
    bridge_cases = []
    for m, n in ((3, 3), (3, 4), (3, 5), (4, 4)):
        left, right = representatives[m], representatives[n]
        left_cap = max(map(abs, energies(left)))
        right_cap = max(map(abs, energies(right)))
        plus, minus, rectangular = bridge_optimum(left, right)
        free = min(plus, minus)
        relation, polynomial = h_relation(free, left_cap, right_cap)
        bridge_cases.append(
            {
                "J_minus": minus,
                "J_orientation_free": free,
                "J_plus": plus,
                "bridge_count": 1 << (m * n - 1),
                "h_integer_polynomial": polynomial,
                "h_relation": relation,
                "left_cap": left_cap,
                "left_key": representative_keys[m],
                "left_order": m,
                "rectangular_optimum": rectangular,
                "reported_parent_F": values[m + n],
                "right_cap": right_cap,
                "right_key": representative_keys[n],
                "right_order": n,
            }
        )

    return {
        "bridge_audit": {
            "cases": bridge_cases,
            "method": "all 2^(mn-1) sign bridges with the first entry fixed positive",
            "scope": "four fixed pairs of first-row-positive optimal children",
        },
        "caveats": [
            "Scalar conclusions inherit the evidence labels of the reported F table.",
            "Bridge enumeration is exact only for the four fixed child pairs.",
            "No asymptotic exponent is inferred from these finite orders.",
        ],
        "input": {
            "path": "results/reported_exact_values.json",
            "statuses_by_order": {str(order): statuses[order] for order in sorted(statuses)},
        },
        "normalization": "F and all caps use Q_A(x)=sum_{i<j} a_ij x_i x_j",
        "scalar_audit": {
            "comparable_pair_count": len(comparable),
            "first_positive": positive[0],
            "first_positive_with_both_children_at_least_3": with_large_children[0],
            "first_positive_exceeding_one": over_one[0],
            "largest_positive": largest,
            "pair_count": len(pairs),
            "positive_comparable_pair_count": len(positive_comparable),
            "positive_pair_count": len(positive),
            "scope": "2 <= m <= n and m+n <= 15",
        },
        "schema_version": 1,
    }


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--values", type=Path, default=DEFAULT_VALUES)
    arguments = parser.parse_args(argv)
    payload = composition_results(arguments.values)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"
    if arguments.output is None:
        print(encoded, end="")
    else:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded, encoding="utf-8", newline="\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
