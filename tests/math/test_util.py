import unittest

from tests.base import Base
from sparkquantum import constants
from sparkquantum.math import util


class TestUtil(Base):
    def setUp(self):
        super().setUp()

    def test_is_scalar(self):
        self.assertTrue(util.is_scalar(1))
        self.assertTrue(util.is_scalar(1.0))
        self.assertTrue(util.is_scalar(1.0j))
        self.assertFalse(util.is_scalar(True))
        self.assertFalse(util.is_scalar(""))

    def test_is_shape(self):
        self.assertTrue(util.is_shape((1, 1)))
        self.assertFalse(util.is_shape([1, 1]))
        self.assertFalse(util.is_shape(tuple()))
        self.assertFalse(util.is_shape((1, 1, 1), ndim=2))
        self.assertFalse(util.is_shape((0, 2)))


if __name__ == '__main__':
    unittest.main()
