import unittest

from tests.base import Base
from sparkquantum.dtqw.mesh.percolation.percolation import is_percolation
from sparkquantum.dtqw.mesh.percolation.permanent import Permanent
from sparkquantum.dtqw.mesh.mesh import is_mesh
from sparkquantum.dtqw.mesh.grid.onedim.segment import Segment

# TODO: implement test_create_operator


class TestSegment(Base):
    def setUp(self):
        super().setUp()
        self.shape = (10, )
        self.mesh = Segment(self.shape)
        self.mesh_perc = Segment(self.shape,
                                 percolation=Permanent([2, 4]))

    def test_is_mesh(self):
        self.assertTrue(is_mesh(self.mesh))
        self.assertTrue(is_mesh(self.mesh_perc))

    def test_shape(self):
        self.assertTupleEqual(self.mesh.shape, self.shape)

    def test_sites(self):
        self.assertEqual(self.mesh.sites, self.shape[0])

    def test_edges(self):
        self.assertEqual(self.mesh.edges, self.mesh.shape[0])

    def test_ndim(self):
        self.assertEqual(self.mesh.ndim, 1)

    def test_percolation(self):
        self.assertTrue(is_percolation(self.mesh_perc.percolation))

    def test_center(self):
        self.assertEqual(self.mesh.center(), int((self.mesh.shape[0] - 1) / 2))
        self.assertEqual(self.mesh.center(coord=True),
                         (int((self.mesh.shape[0] - 1) / 2), ))

    def test_has_site(self):
        self.assertTrue(self.mesh.has_site(0))
        self.assertTrue(self.mesh.has_site(self.mesh.shape[0] - 1))
        self.assertFalse(self.mesh.has_site(self.mesh.shape[0]))

    def test_has_coordinate(self):
        self.assertTrue(self.mesh.has_coordinate((0, )))
        self.assertTrue(self.mesh.has_coordinate((self.mesh.shape[0] - 1, )))
        self.assertFalse(self.mesh.has_coordinate((self.mesh.shape[0], )))

    def test_to_site(self):
        self.assertEqual(self.mesh.to_site((0, )), 0)
        self.assertEqual(
            self.mesh.to_site(
                (self.mesh.shape[0] - 1, )), self.mesh.shape[0] - 1)

    def test_to_coordinate(self):
        self.assertEqual(self.mesh.to_coordinate(0), (0, ))
        self.assertEqual(
            self.mesh.to_coordinate(
                self.mesh.shape[0] - 1), (self.mesh.shape[0] - 1, ))

    def test_axis(self):
        self.assertTupleEqual(self.mesh.axis(), (range(self.mesh.shape[0]), ))

    def test_str(self):
        self.assertEqual(str(self.mesh),
                         'Segment grid with shape {}'.format(self.mesh.shape))
        self.assertEqual(str(self.mesh_perc),
                         'Segment grid with shape {} and {}'.format(self.mesh_perc.shape,
                                                                    str(self.mesh_perc.percolation)))


if __name__ == '__main__':
    unittest.main()
