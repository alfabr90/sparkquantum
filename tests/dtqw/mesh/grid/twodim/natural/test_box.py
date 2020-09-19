import unittest

from tests.base import Base
from sparkquantum.dtqw.mesh.percolation.percolation import is_percolation
from sparkquantum.dtqw.mesh.percolation.permanent import Permanent
from sparkquantum.dtqw.mesh.mesh import is_mesh
from sparkquantum.dtqw.mesh.grid.twodim.natural.box import Box

# TODO: implement test_create_operator


class TestBox(Base):
    def setUp(self):
        super().setUp()
        self.shape = (20, 10)
        self.mesh = Box(self.shape)
        self.mesh_perc = Box(self.shape,
                             percolation=Permanent([2, 4]))

    def test_is_mesh(self):
        self.assertTrue(is_mesh(self.mesh))

    def test_shape(self):
        self.assertEqual(self.mesh.shape,
                         (self.shape[0], self.shape[1]))

    def test_sites(self):
        self.assertEqual(self.mesh.sites,
                         self.mesh.shape[0] * self.mesh.shape[1])

    def test_edges(self):
        self.assertEqual(self.mesh.edges,
                         2 * self.mesh.shape[0] * self.mesh.shape[1])

    def test_ndim(self):
        self.assertEqual(self.mesh.ndim, 2)

    def test_percolation(self):
        self.assertTrue(is_percolation(self.mesh_perc.percolation))

    def test_center(self):
        coord_x = int((self.mesh.shape[0] - 1) / 2)
        coord_y = int((self.mesh.shape[1] - 1) / 2)

        self.assertEqual(self.mesh.center(),
                         coord_y * self.mesh.shape[0] + coord_x)
        self.assertEqual(self.mesh.center(coord=True), (coord_x, coord_y))

    def test_has_site(self):
        sites = self.mesh.shape[0] * self.mesh.shape[1]
        self.assertTrue(self.mesh.has_site(0))
        self.assertTrue(self.mesh.has_site(sites - 1))
        self.assertFalse(self.mesh.has_site(sites))

    def test_has_coordinate(self):
        size_x, size_y = self.mesh.shape
        self.assertTrue(self.mesh.has_coordinate((0, 0)))
        self.assertTrue(self.mesh.has_coordinate((size_x - 1, size_y - 1)))
        self.assertFalse(self.mesh.has_coordinate((size_x - 1, size_y)))
        self.assertFalse(self.mesh.has_coordinate((size_x + 1, size_y - 1)))
        self.assertFalse(self.mesh.has_coordinate((size_x, size_y)))

    def test_to_site(self):
        coord_x, coord_y = self.mesh.shape
        self.assertEqual(self.mesh.to_site((0, 0)), 0)
        self.assertEqual(
            self.mesh.to_site(
                (0, coord_y - 1)), coord_x * coord_y - coord_x)
        self.assertEqual(
            self.mesh.to_site((coord_x - 1, 0)), coord_x - 1)
        self.assertEqual(
            self.mesh.to_site(
                (coord_x - 1, coord_y - 1)), self.mesh.shape[0] * self.mesh.shape[1] - 1)

    def test_to_coordinate(self):
        size = self.mesh.shape[0] * self.mesh.shape[1]
        self.assertEqual(self.mesh.to_coordinate(0), (0, 0))
        self.assertEqual(
            self.mesh.to_coordinate(
                self.mesh.shape[1] * self.mesh.shape[0] - self.mesh.shape[0]), (0, self.mesh.shape[1] - 1))
        self.assertEqual(
            self.mesh.to_coordinate(self.mesh.shape[0] - 1), (self.mesh.shape[0] - 1, 0))
        self.assertEqual(
            self.mesh.to_coordinate(size - 1), (self.mesh.shape[0] - 1, self.mesh.shape[1] - 1))

    def test_axis(self):
        self.assertTupleEqual(self.mesh.axis(),
                              (range(self.shape[0]), range(self.shape[1])))

    def test_str(self):
        self.assertEqual(str(self.mesh),
                         'Natural box with shape {}'.format(self.mesh.shape))
        self.assertEqual(str(self.mesh_perc),
                         'Natural box with shape {} and {}'.format(self.mesh_perc.shape,
                                                                   str(self.mesh_perc.percolation)))


if __name__ == '__main__':
    unittest.main()
