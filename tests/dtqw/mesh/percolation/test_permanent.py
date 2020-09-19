import unittest

from tests.base import Base
from sparkquantum.dtqw.mesh.percolation.percolation import is_percolation
from sparkquantum.dtqw.mesh.percolation.permanent import Permanent, is_permanent

# TODO: implement test_generate


class TestPermanent(Base):
    def setUp(self):
        super().setUp()
        self.edges = [2, 4]
        self.percolation = Permanent(self.edges)

    def test_is_percolation(self):
        self.assertTrue(is_percolation(self.percolation))

    def test_str(self):
        self.assertEqual(str(self.percolation),
                         'Permanent percolations generator with {} percolations'.format(len(self.edges)))

    def test_is_permanent(self):
        self.assertTrue(is_permanent(self.percolation))

    def test_edges(self):
        self.assertTupleEqual(self.percolation.edges, tuple(self.edges))


if __name__ == '__main__':
    unittest.main()
