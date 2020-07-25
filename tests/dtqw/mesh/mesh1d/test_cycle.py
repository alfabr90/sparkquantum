import unittest

from tests.base import Base
from sparkquantum.dtqw.mesh.broken_links.broken_links import is_broken_links
from sparkquantum.dtqw.mesh.broken_links.permanent_broken_links import PermanentBrokenLinks
from sparkquantum.dtqw.mesh.mesh import is_mesh
from sparkquantum.dtqw.mesh.mesh1d.cycle import Cycle

# TODO: implement test_create_operator


class TestCycle(Base):
    def setUp(self):
        super().setUp()
        self.size = 10
        self.steps = 10
        self.mesh = Cycle(self.size)
        self.mesh_bl = Cycle(self.size,
                             broken_links=PermanentBrokenLinks([2, 4]))

    def test_is_mesh(self):
        self.assertTrue(is_mesh(self.mesh))
        self.assertTrue(is_mesh(self.mesh_bl))

    def test_size(self):
        self.assertEqual(self.mesh.size, self.size)

    def test_num_edges(self):
        self.assertEqual(self.mesh.num_edges, self.mesh.size)

    def test_coin_size(self):
        self.assertEqual(self.mesh.coin_size, 2)

    def test_dimension(self):
        self.assertEqual(self.mesh.dimension, 1)

    def test_broken_links(self):
        self.assertTrue(is_broken_links(self.mesh_bl.broken_links))

    def test_center(self):
        self.assertEqual(self.mesh.center(), int((self.mesh.size - 1) / 2))

    def test_center_coordinates(self):
        self.assertEqual(self.mesh.center_coordinates(),
                         (int((self.mesh.size - 1) / 2), ))

    def test_axis(self):
        self.assertEqual(self.mesh.axis(), range(self.mesh.size))

    def test_check_steps(self):
        self.assertFalse(self.mesh.check_steps(-1))
        self.assertTrue(self.mesh.check_steps(self.steps))

    def test_str(self):
        self.assertEqual(str(self.mesh),
                         'Cycle with dimension {}'.format(self.mesh.size))
        self.assertEqual(str(self.mesh_bl),
                         'Cycle with dimension {} and {}'.format(self.mesh_bl.size,
                                                                 str(self.mesh_bl.broken_links)))


if __name__ == '__main__':
    unittest.main()
