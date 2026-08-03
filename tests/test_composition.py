from __future__ import annotations

import importlib.util
import itertools
from pathlib import Path
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "experiments" / "composition_small_orders.py"
SPEC = importlib.util.spec_from_file_location("composition_small_orders", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
composition = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(composition)


class ScalarCompositionTests(unittest.TestCase):
    def test_exact_counterexample_certificates(self) -> None:
        self.assertEqual(composition.h_relation(4, 1, 1), ("greater", 2312))
        self.assertEqual(composition.h_relation(10, 3, 4), ("greater", 33075))

    def test_reported_value_audit_summary(self) -> None:
        summary = composition.composition_results()["scalar_audit"]
        self.assertEqual(summary["pair_count"], 42)
        self.assertEqual(summary["comparable_pair_count"], 24)
        self.assertEqual(summary["positive_pair_count"], 21)
        self.assertEqual(summary["positive_comparable_pair_count"], 15)
        self.assertEqual(
            (summary["largest_positive"]["left_order"],
             summary["largest_positive"]["right_order"]),
            (5, 6),
        )


class ExactBridgeTests(unittest.TestCase):
    def test_relative_orientation_matters_at_three_plus_three(self) -> None:
        child = composition.signing(3, (-1,))
        self.assertEqual(composition.bridge_optimum(child, child), (7, 5, 5))

    def test_block_identity_against_direct_parent_evaluation(self) -> None:
        left = composition.signing(3, (-1,))
        right = composition.signing(3, (-1,))
        bridges = (
            (1, -1, 1, 1, -1, -1, 1, -1, 1),
            (-1,) * 9,
        )
        projective_xs, projective_ys = composition.spins(3), composition.spins(3)
        full_ys = list(itertools.product((-1, 1), repeat=3))

        def pair_energy(matrix, spin):
            return sum(
                matrix[row][column] * spin[row] * spin[column]
                for row in range(3)
                for column in range(row + 1, 3)
            )

        def cross_energy(bridge, x, y):
            return sum(
                bridge[3 * row + column] * x[row] * y[column]
                for row in range(3)
                for column in range(3)
            )

        for bridge in bridges:
            for orientation in (-1, 1):
                with self.subTest(bridge=bridge, orientation=orientation):
                    direct = max(
                        abs(
                            pair_energy(left, x)
                            + orientation * pair_energy(right, y)
                            + cross_energy(bridge, x, y)
                        )
                        for x in projective_xs
                        for y in full_ys
                    )
                    separated = max(
                        abs(
                            pair_energy(left, x)
                            + orientation * pair_energy(right, y)
                        )
                        + abs(cross_energy(bridge, x, y))
                        for x in projective_xs
                        for y in projective_ys
                    )
                    self.assertEqual(direct, separated)

    def test_quotient_shell_hierarchy_and_counting_obstruction(self) -> None:
        child = composition.signing(3, (-1,))
        record = composition.quotient_shell_record(child, child, 1, -1)

        self.assertEqual(record["histogram_gain"], 2)
        self.assertEqual(record["one_character_gain"], 4)
        self.assertTrue(record["one_character_second_moment_certifies"])
        self.assertEqual(record["joint_overlap_type_count"], 4)
        occupancy = record["one_character_target_occupancy"]
        self.assertEqual(occupancy["collision_count"], 6)
        self.assertEqual(occupancy["collision_threshold"], 3)
        self.assertTrue(occupancy["occupancy_lower_bound_certifies"])
        self.assertEqual(record["exact_gain"], 6)
        self.assertEqual(record["exact_parent_cap"], 5)
        self.assertEqual(record["exact_target_zero_switching_count"], 1)
        self.assertEqual(
            record["exact_target_minimum_positive_violation_count"], 4
        )
        self.assertEqual(
            record["exact_target_quotient_dimension_lower_bound"], 4
        )

    def test_one_character_can_certify_the_exact_three_plus_four_cap(self) -> None:
        left = composition.signing(3, (-1,))
        right = composition.signing(4, (-1, -1, 1))
        record = composition.quotient_shell_record(left, right, 3, 1)

        self.assertEqual(record["histogram_parent_cap"], 11)
        self.assertEqual(record["one_character_parent_cap"], 9)
        self.assertFalse(record["one_character_second_moment_certifies"])
        self.assertEqual(record["joint_overlap_type_count"], 14)
        occupancy = record["one_character_target_occupancy"]
        self.assertEqual(occupancy["collision_count"], 35)
        self.assertEqual(occupancy["collision_threshold"], 153)
        self.assertEqual(occupancy["maximum_occupancy_collision_count"], 49)
        self.assertTrue(occupancy["bounded_fiber_upper_bound_obstructs"])
        self.assertEqual(record["exact_parent_cap"], 9)

    def test_two_replica_collision_normal_form(self) -> None:
        eta = [0, 0, 2, 4]
        zeta = [0, 2, 0, 2]
        counts = composition.violation_counts(eta, zeta, 3)
        occupancy = composition.two_replica_occupancy_record(counts)
        size = len(counts)
        total = sum(counts)
        direct_parseval_gap = (
            size * sum(value * value for value in counts) - total * total
            - (size - 1) * (total - size) ** 2
        )

        self.assertEqual(
            direct_parseval_gap > 0,
            occupancy["collision_count"] > occupancy["collision_threshold"],
        )


if __name__ == "__main__":
    unittest.main()
