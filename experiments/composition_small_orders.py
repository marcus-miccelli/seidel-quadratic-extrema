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
from collections import Counter
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


def bridge_from_index(index: int, m: int, n: int) -> tuple[int, ...]:
    """Recover the indexed bridge in the experiment's lexicographic order."""
    free = m * n - 1
    if index < 0 or index >= 1 << free:
        raise ValueError(f"bridge index {index} is outside [0, {1 << free})")
    tail = tuple(
        1 if (index >> (free - position - 1)) & 1 else -1
        for position in range(free)
    )
    return (1, *tail)


def walsh_hadamard(values: list[int]) -> list[int]:
    """Return the unnormalized Walsh--Hadamard transform."""
    transformed = list(values)
    width = 1
    while width < len(transformed):
        for start in range(0, len(transformed), 2 * width):
            for offset in range(width):
                left = transformed[start + offset]
                right = transformed[start + width + offset]
                transformed[start + offset] = left + right
                transformed[start + width + offset] = left - right
        width *= 2
    return transformed


def violation_counts(eta: list[int], zeta: list[int], gain: int) -> list[int]:
    """Return the bad-pair addition-fiber sizes for every switching."""
    if len(eta) != len(zeta):
        raise ValueError("eta and zeta must have the same length")
    size = len(eta)
    if size == 0 or size & (size - 1):
        raise ValueError("eta and zeta must be indexed by a nonempty two-group")
    return [
        sum(eta[state] + zeta[state ^ shift] < gain for state in range(size))
        for shift in range(size)
    ]


def two_replica_occupancy_record(counts: list[int]) -> dict[str, int | bool]:
    """Audit the sharp occupancy bounds for the two-replica criterion."""
    size = len(counts)
    if size == 0 or any(value < 0 for value in counts):
        raise ValueError("fiber counts must be a nonempty nonnegative list")
    total = sum(counts)
    if total < size:
        raise ValueError("the two-replica occupancy audit requires total >= size")
    zero_count = counts.count(0)
    positive_fibers = size - zero_count
    maximum = max(counts)
    excess = total - size
    collisions = sum(value * (value - 1) // 2 for value in counts)
    threshold = excess * (excess + 1) // 2

    balanced_level, balanced_remainder = divmod(total, positive_fibers)
    minimum_collisions = (
        positive_fibers * balanced_level * (balanced_level - 1) // 2
        + balanced_remainder * balanced_level
    )

    if maximum == 1:
        maximum_collisions = 0
    else:
        extra_entries = total - positive_fibers
        full_fibers, final_extra = divmod(extra_entries, maximum - 1)
        maximum_collisions = (
            full_fibers * maximum * (maximum - 1) // 2
            + final_extra * (final_extra + 1) // 2
        )

    assert minimum_collisions <= collisions <= maximum_collisions
    return {
        "bounded_fiber_upper_bound_obstructs": maximum_collisions <= threshold,
        "collision_count": collisions,
        "collision_threshold": threshold,
        "excess_violation_count": excess,
        "maximum_fiber_size": maximum,
        "maximum_occupancy_collision_count": maximum_collisions,
        "minimum_occupancy_collision_count": minimum_collisions,
        "occupancy_lower_bound_certifies": minimum_collisions > threshold,
        "total_violation_count": total,
        "two_replica_certifies": collisions > threshold,
        "zero_fiber_count": zero_count,
    }


def joint_overlap_type_count(eta: list[int], zeta: list[int]) -> int:
    """Count distinct simultaneous shell-autocorrelation signatures."""
    if len(eta) != len(zeta):
        raise ValueError("eta and zeta must have the same length")
    size = len(eta)
    if size == 0 or size & (size - 1):
        raise ValueError("eta and zeta must be indexed by a nonempty two-group")
    signatures = set()
    for difference in range(size):
        eta_overlap = Counter(
            (eta[state], eta[state ^ difference]) for state in range(size)
        )
        zeta_overlap = Counter(
            (zeta[state], zeta[state ^ difference]) for state in range(size)
        )
        signatures.add(
            (tuple(sorted(eta_overlap.items())), tuple(sorted(zeta_overlap.items())))
        )
    return len(signatures)


def quotient_shell_record(
    left: tuple[tuple[int, ...], ...],
    right: tuple[tuple[int, ...], ...],
    bridge_index: int,
    orientation: int,
) -> dict[str, Any]:
    """Audit histogram, one-character, and exact switching gains."""
    m, n = len(left), len(right)
    bridge = bridge_from_index(bridge_index, m, n)
    left_states, right_states = spins(m), spins(n)
    left_energies, right_energies = energies(left), energies(right)
    left_cap = max(map(abs, left_energies))
    right_cap = max(map(abs, right_energies))
    state_count = 1 << (m + n - 2)
    eta = [0] * state_count
    zeta_raw = [0] * state_count

    def state_code(spin: tuple[int, ...]) -> int:
        return sum(
            (spin[position] == -1) << (position - 1)
            for position in range(1, len(spin))
        )

    for i, x in enumerate(left_states):
        for j, y in enumerate(right_states):
            code = state_code(x) | (state_code(y) << (m - 1))
            eta[code] = (
                left_cap
                + right_cap
                - abs(left_energies[i] + orientation * right_energies[j])
            )
            zeta_raw[code] = abs(
                sum(
                    bridge[n * row + column] * x[row] * y[column]
                    for row in range(m)
                    for column in range(n)
                )
            )
    bridge_cap = max(zeta_raw)
    zeta = [bridge_cap - value for value in zeta_raw]

    histogram_gain = one_character_gain = exact_gain = 0
    one_character_mask = 0
    one_character_second_moment_certifies = False
    one_character_counts: list[int] | None = None
    exact_counts: list[int] | None = None
    for gain in range(left_cap + right_cap + bridge_cap + 2):
        counts = violation_counts(eta, zeta, gain)
        total = sum(counts)
        if total < state_count:
            histogram_gain = gain
        transform = walsh_hadamard(counts)
        if len(transform) > 1:
            mask = max(range(1, state_count), key=lambda item: abs(transform[item]))
            if total - abs(transform[mask]) < state_count:
                one_character_gain = gain
                one_character_mask = mask
                one_character_second_moment_certifies = (
                    total < state_count
                    or state_count * sum(value * value for value in counts) - total * total
                    > (state_count - 1) * (total - state_count) ** 2
                )
                one_character_counts = counts
        if min(counts) == 0:
            exact_gain = gain
            exact_counts = counts

    assert exact_counts is not None
    assert one_character_counts is not None
    one_character_occupancy = (
        two_replica_occupancy_record(one_character_counts)
        if sum(one_character_counts) >= state_count
        else None
    )
    zero_count = sum(value == 0 for value in exact_counts)
    positive_values = [value for value in exact_counts if value > 0]
    minimum_positive = min(positive_values) if positive_values else 0
    dimension = m + n - 2
    quotient_dimension_lower_bound = 0
    maximum_certifying_coset_size: int | None = None
    if minimum_positive > 1:
        maximum_certifying_coset_size = 0
        for coset_size in (1 << power for power in range(dimension + 1)):
            if (minimum_positive - 1) * coset_size < minimum_positive * zero_count:
                maximum_certifying_coset_size = coset_size
        quotient_dimension_lower_bound = dimension - (
            maximum_certifying_coset_size.bit_length() - 1
        )

    baseline = left_cap + right_cap + bridge_cap
    return {
        "ambient_switching_dimension": dimension,
        "baseline_parent_cap": baseline,
        "bridge": [
            list(bridge[row * n : (row + 1) * n]) for row in range(m)
        ],
        "bridge_cap": bridge_cap,
        "bridge_index": bridge_index,
        "exact_gain": exact_gain,
        "exact_parent_cap": baseline - exact_gain,
        "exact_target_minimum_positive_violation_count": minimum_positive,
        "exact_target_quotient_dimension_lower_bound": quotient_dimension_lower_bound,
        "exact_target_zero_switching_count": zero_count,
        "histogram_gain": histogram_gain,
        "histogram_parent_cap": baseline - histogram_gain,
        "counting_bound_maximum_coset_size": maximum_certifying_coset_size,
        "joint_overlap_type_count": joint_overlap_type_count(eta, zeta),
        "one_character_gain": one_character_gain,
        "one_character_mask": one_character_mask,
        "one_character_parent_cap": baseline - one_character_gain,
        "one_character_second_moment_certifies": one_character_second_moment_certifies,
        "one_character_target_occupancy": one_character_occupancy,
        "orientation": orientation,
        "orders": [m, n],
        "state_count": state_count,
    }


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

    quotient_cases = [
        quotient_shell_record(representatives[m], representatives[n], index, orientation)
        for m, n, index, orientation in (
            (3, 3, 1, -1),
            (3, 4, 3, 1),
            (3, 5, 3, 1),
            (4, 4, 18, -1),
        )
    ]

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
        "quotient_shell_audit": {
            "cases": quotient_cases,
            "classification": "exact finite computation",
            "criterion": (
                "for violation counts V(s), the histogram test is sum_s V(s)<|G|; "
                "one character succeeds when sum_s V(s)-max_chi|sum_s chi(s)V(s)|<|G|"
            ),
            "collision_normal_form": (
                "when T=sum_s V(s)=|G|+q, the global two-replica test is "
                "equivalent to sum_s binom(V(s),2)>binom(q+1,2)"
            ),
            "overlap_type_definition": (
                "a joint type is one distinct pair of eta- and zeta-shell "
                "autocorrelation tables indexed by a switching difference"
            ),
            "scope": "four fixed child, orientation, and bridge witnesses",
            "two_replica_criterion": (
                "for T=sum_s V(s)>=S=|G| and U=sum_s V(s)^2, "
                "S*U-T^2>(S-1)(T-S)^2 forces a successful one-character quotient"
            ),
        },
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
