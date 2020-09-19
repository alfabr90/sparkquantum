import unittest

from tests.base import Base
from sparkquantum.dtqw.mesh.percolation.percolation import is_percolation
from sparkquantum.dtqw.mesh.percolation.random import Random

# TODO: implement test_generate


class TestRandom(Base):
    def setUp(self):
        super().setUp()
        self.prob = 0.1
        self.percolation = Random(self.prob)

    def test_is_percolation(self):
        self.assertTrue(is_percolation(self.percolation))

    def test_str(self):
        self.assertEqual(str(self.percolation),
                         'Random percolations generator with probability value of {}'.format(self.prob))

    def test_probability(self):
        self.assertEqual(self.percolation.probability, self.prob)


if __name__ == '__main__':
    unittest.main()
