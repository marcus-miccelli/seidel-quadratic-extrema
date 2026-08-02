from __future__ import annotations

import itertools
import unittest

from paataproof import SeidelMatrix, evaluate_m


def full_cube_objective(rows: list[list[int]]) -> int:
    """Independent definition used to check the symmetry-reduced evaluator."""
    order = len(rows)
    return max(
        abs(
            sum(
                rows[row][column] * signs[row] * signs[column]
                for row in range(order)
                for column in range(order)
            )
        )
        for signs in itertools.product((-1, 1), repeat=order)
    )


class SeidelMatrixTests(unittest.TestCase):
    def test_constructs_an_immutable_matrix(self) -> None:
        source = [[0, 1, -1], [1, 0, 1], [-1, 1, 0]]
        matrix = SeidelMatrix(source)
        source[0][1] = -1

        self.assertEqual(matrix.order, 3)
        self.assertEqual(matrix.rows, ((0, 1, -1), (1, 0, 1), (-1, 1, 0)))
        self.assertEqual(tuple(matrix), matrix.rows)
        self.assertEqual(matrix[1], (1, 0, 1))
        self.assertEqual(matrix[:2], matrix.rows[:2])

    def test_rejects_non_square_or_empty_input(self) -> None:
        with self.assertRaisesRegex(ValueError, "positive order"):
            SeidelMatrix([])
        with self.assertRaisesRegex(ValueError, "must be square"):
            SeidelMatrix([[0, 1]])
        with self.assertRaisesRegex(ValueError, "must be square"):
            SeidelMatrix([[0, 1], [1]])

    def test_rejects_invalid_entries(self) -> None:
        invalid = (
            ([[1]], ValueError, "zero diagonal"),
            ([[0, 0], [0, 0]], ValueError, "off-diagonal"),
            ([[0, 1], [-1, 0]], ValueError, "symmetric"),
            ([[0, 1.0], [1, 0]], TypeError, "must be an integer"),
            ([[0, True], [True, 0]], TypeError, "must be an integer"),
        )
        for rows, error_type, message in invalid:
            with self.subTest(rows=rows):
                with self.assertRaisesRegex(error_type, message):
                    SeidelMatrix(rows)

    def test_quadratic_form_requires_a_sign_vector_of_matching_order(self) -> None:
        matrix = SeidelMatrix([[0, -1], [-1, 0]])

        self.assertEqual(matrix.quadratic_form([1, 1]), -2)
        self.assertEqual(matrix.quadratic_form([1, -1]), 2)
        with self.assertRaisesRegex(ValueError, "length 1"):
            matrix.quadratic_form([1])
        with self.assertRaisesRegex(ValueError, "must be -1 or 1"):
            matrix.quadratic_form([1, 0])
        with self.assertRaisesRegex(TypeError, "must be an integer"):
            matrix.quadratic_form([1, 1.0])


class ExhaustiveObjectiveTests(unittest.TestCase):
    def test_order_one_has_zero_objective(self) -> None:
        self.assertEqual(evaluate_m(SeidelMatrix([[0]])), 0)

    def test_known_three_by_three_objective(self) -> None:
        matrix = SeidelMatrix([[0, 1, -1], [1, 0, 1], [-1, 1, 0]])
        self.assertEqual(evaluate_m(matrix), 6)

    def test_matches_the_full_cube_definition(self) -> None:
        examples = [
            [[0, 1], [1, 0]],
            [
                [0, 1, 1, -1],
                [1, 0, -1, 1],
                [1, -1, 0, 1],
                [-1, 1, 1, 0],
            ],
            [
                [0, 1, 1, 1, -1],
                [1, 0, -1, 1, 1],
                [1, -1, 0, -1, 1],
                [1, 1, -1, 0, -1],
                [-1, 1, 1, -1, 0],
            ],
        ]
        for rows in examples:
            with self.subTest(order=len(rows)):
                self.assertEqual(
                    evaluate_m(SeidelMatrix(rows)), full_cube_objective(rows)
                )

    def test_requires_a_validated_matrix(self) -> None:
        with self.assertRaisesRegex(TypeError, "must be a SeidelMatrix"):
            evaluate_m([[0]])  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
