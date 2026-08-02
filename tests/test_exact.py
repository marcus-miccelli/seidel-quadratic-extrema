from __future__ import annotations

import itertools
import unittest

from paataproof import (
    exact_minimum_m,
    iter_switching_normalized_seidel_matrices,
    normalized_seidel_count,
)


def all_labeled_seidel_rows(order: int):
    """Independently enumerate the full labeled Seidel space for tests."""
    positions = tuple(itertools.combinations(range(order), 2))
    for values in itertools.product((-1, 1), repeat=len(positions)):
        rows = [[0] * order for _ in range(order)]
        for (row, column), value in zip(positions, values, strict=True):
            rows[row][column] = value
            rows[column][row] = value
        yield tuple(tuple(row) for row in rows)


def independently_switch_normalize(rows: tuple[tuple[int, ...], ...]):
    """Normalize by the defining DAD formula, independently of package code."""
    switches = (1, *(rows[0][column] for column in range(1, len(rows))))
    return tuple(
        tuple(switches[row] * rows[row][column] * switches[column]
              for column in range(len(rows)))
        for row in range(len(rows))
    )


class NormalizedEnumerationTests(unittest.TestCase):
    def test_counts_are_complete_and_rows_have_no_duplicates(self) -> None:
        for order in range(1, 6):
            with self.subTest(order=order):
                matrices = list(iter_switching_normalized_seidel_matrices(order))
                rows = {matrix.rows for matrix in matrices}
                expected = 2 ** ((order - 1) * (order - 2) // 2)

                self.assertEqual(normalized_seidel_count(order), expected)
                self.assertEqual(len(matrices), expected)
                self.assertEqual(len(rows), expected)
                for matrix in matrices:
                    self.assertTrue(
                        all(matrix[0][index] == 1 for index in range(1, order))
                    )
                    self.assertTrue(
                        all(matrix[index][0] == 1 for index in range(1, order))
                    )

    def test_independent_switching_normalization_covers_generated_space(self) -> None:
        order = 4
        generated = {
            matrix.rows
            for matrix in iter_switching_normalized_seidel_matrices(order)
        }
        normalized_full_space = {
            independently_switch_normalize(rows)
            for rows in all_labeled_seidel_rows(order)
        }
        self.assertEqual(normalized_full_space, generated)

    def test_order_validation_is_eager_and_rejects_booleans(self) -> None:
        for invalid in (True, 2.0, "2", None):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(TypeError, "order must be an integer"):
                    iter_switching_normalized_seidel_matrices(invalid)  # type: ignore[arg-type]
                with self.assertRaisesRegex(TypeError, "order must be an integer"):
                    exact_minimum_m(invalid)  # type: ignore[arg-type]
                with self.assertRaisesRegex(TypeError, "order must be an integer"):
                    normalized_seidel_count(invalid)  # type: ignore[arg-type]

        for invalid in (0, -1):
            with self.subTest(invalid=invalid):
                with self.assertRaisesRegex(ValueError, "order must be positive"):
                    iter_switching_normalized_seidel_matrices(invalid)
                with self.assertRaisesRegex(ValueError, "order must be positive"):
                    exact_minimum_m(invalid)
                with self.assertRaisesRegex(ValueError, "order must be positive"):
                    normalized_seidel_count(invalid)


class ExactMinimumTests(unittest.TestCase):
    def test_exact_minima_through_order_six(self) -> None:
        expected = (0, 2, 6, 8, 8, 10)
        self.assertEqual(
            tuple(exact_minimum_m(order) for order in range(1, 7)), expected
        )


if __name__ == "__main__":
    unittest.main()
