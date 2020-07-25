import unittest

from tests.base import Base
from sparkquantum import constants
from sparkquantum.math import util


class TestUtil(Base):
    def setUp(self):
        super().setUp()

    def tearDown(self):
        super().tearDown()

    def test_is_scalar(self):
        self.assertTrue(util.is_scalar(1))
        self.assertTrue(util.is_scalar(1.0))
        self.assertTrue(util.is_scalar(1.0j))
        self.assertFalse(util.is_scalar(True))
        self.assertFalse(util.is_scalar(""))

    def test_is_shape(self):
        self.assertTrue(util.is_shape((1, 1)))
        self.assertTrue(util.is_shape([1, 1]))
        self.assertFalse(util.is_shape([1]))
        self.assertFalse(util.is_shape([1, 1, 1]))
        self.assertFalse(util.is_shape([0, 0]))

    def test_change_coordinate(self):
        sc = self.spark_context

        items_default = [(1, 2, 3), (4, 5, 6)]
        items_multiplier = [(2, (1, 3)), (5, (4, 6))]
        items_multiplicand = [(1, (2, 3)), (4, (5, 6))]
        items_indexed = [((1, 2), 3), ((4, 5), 6)]

        # MatrixCoordinateDefault

        items = util.change_coordinate(
            sc.parallelize(items_default),
            constants.MatrixCoordinateDefault,
            constants.MatrixCoordinateDefault
        ).collect()

        self.assertListEqual(items, items_default)

        items = util.change_coordinate(
            sc.parallelize(items_default),
            constants.MatrixCoordinateDefault,
            constants.MatrixCoordinateMultiplier
        ).collect()

        self.assertListEqual(items, items_multiplier)

        items = util.change_coordinate(
            sc.parallelize(items_default),
            constants.MatrixCoordinateDefault,
            constants.MatrixCoordinateMultiplicand
        ).collect()

        self.assertListEqual(items, items_multiplicand)

        items = util.change_coordinate(
            sc.parallelize(items_default),
            constants.MatrixCoordinateDefault,
            constants.MatrixCoordinateIndexed
        ).collect()

        self.assertListEqual(items, items_indexed)

        # MatrixCoordinateMultiplier

        items = util.change_coordinate(
            sc.parallelize(items_multiplier),
            constants.MatrixCoordinateMultiplier,
            constants.MatrixCoordinateDefault
        ).collect()

        self.assertListEqual(items, items_default)

        items = util.change_coordinate(
            sc.parallelize(items_multiplier),
            constants.MatrixCoordinateMultiplier,
            constants.MatrixCoordinateMultiplier
        ).collect()

        self.assertListEqual(items, items_multiplier)

        items = util.change_coordinate(
            sc.parallelize(items_multiplier),
            constants.MatrixCoordinateMultiplier,
            constants.MatrixCoordinateMultiplicand
        ).collect()

        self.assertListEqual(items, items_multiplicand)

        items = util.change_coordinate(
            sc.parallelize(items_multiplier),
            constants.MatrixCoordinateMultiplier,
            constants.MatrixCoordinateIndexed
        ).collect()

        self.assertListEqual(items, items_indexed)

        # MatrixCoordinateMultiplicand

        items = util.change_coordinate(
            sc.parallelize(items_multiplicand),
            constants.MatrixCoordinateMultiplicand,
            constants.MatrixCoordinateDefault
        ).collect()

        self.assertListEqual(items, items_default)

        items = util.change_coordinate(
            sc.parallelize(items_multiplicand),
            constants.MatrixCoordinateMultiplicand,
            constants.MatrixCoordinateMultiplier
        ).collect()

        self.assertListEqual(items, items_multiplier)

        items = util.change_coordinate(
            sc.parallelize(items_multiplicand),
            constants.MatrixCoordinateMultiplicand,
            constants.MatrixCoordinateMultiplicand
        ).collect()

        self.assertListEqual(items, items_multiplicand)

        items = util.change_coordinate(
            sc.parallelize(items_multiplicand),
            constants.MatrixCoordinateMultiplicand,
            constants.MatrixCoordinateIndexed
        ).collect()

        self.assertListEqual(items, items_indexed)

        # MatrixCoordinateIndexed

        items = util.change_coordinate(
            sc.parallelize(items_indexed),
            constants.MatrixCoordinateIndexed,
            constants.MatrixCoordinateDefault
        ).collect()

        self.assertListEqual(items, items_default)

        items = util.change_coordinate(
            sc.parallelize(items_indexed),
            constants.MatrixCoordinateIndexed,
            constants.MatrixCoordinateMultiplier
        ).collect()

        self.assertListEqual(items, items_multiplier)

        items = util.change_coordinate(
            sc.parallelize(items_indexed),
            constants.MatrixCoordinateIndexed,
            constants.MatrixCoordinateMultiplicand
        ).collect()

        self.assertListEqual(items, items_multiplicand)

        items = util.change_coordinate(
            sc.parallelize(items_indexed),
            constants.MatrixCoordinateIndexed,
            constants.MatrixCoordinateIndexed
        ).collect()

        self.assertListEqual(items, items_indexed)

    def test_remove_zeros(self):
        sc = self.spark_context

        items_initial = [(1, 2, 0), (4, 5, 6)]
        items_removed = [(4, 5, 6)]

        items = util.remove_zeros(
            sc.parallelize(items_initial),
            int,
            constants.MatrixCoordinateDefault
        ).collect()

        self.assertEqual(items, items_removed)


if __name__ == '__main__':
    unittest.main()
