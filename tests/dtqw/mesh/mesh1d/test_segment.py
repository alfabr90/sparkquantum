import unittest

from tests.base import Base
from sparkquantum.dtqw.mesh.broken_links.broken_links import is_broken_links
from sparkquantum.dtqw.mesh.broken_links.permanent import PermanentBrokenLinks
from sparkquantum.dtqw.mesh.mesh import is_mesh
from sparkquantum.dtqw.mesh.mesh1d.segment import Segment

# TODO: implement test_create_operator


class TestSegment(Base):
    def setUp(self):
        super().setUp()
        self.size = (10, )
        self.steps = 10
        self.mesh = Segment(self.size)
        self.mesh_bl = Segment(self.size,
                               broken_links=PermanentBrokenLinks([2, 4]))

    def test_is_mesh(self):
        self.assertTrue(is_mesh(self.mesh))
        self.assertTrue(is_mesh(self.mesh_bl))

    def test_size(self):
        self.assertEqual(self.mesh.size, self.size)

    def test_num_edges(self):
        self.assertEqual(self.mesh.num_edges, self.mesh.size[0])

    def test_coin_size(self):
        self.assertEqual(self.mesh.coin_size, 2)

    def test_dimension(self):
        self.assertEqual(self.mesh.dimension, 1)

    def test_broken_links(self):
        self.assertTrue(is_broken_links(self.mesh_bl.broken_links))

    def test_center(self):
        self.assertEqual(self.mesh.center(), int((self.mesh.size[0] - 1) / 2))

    def test_center_coordinates(self):
        self.assertEqual(self.mesh.center_coordinates(),
                         (int((self.mesh.size[0] - 1) / 2), ))

    def test_axis(self):
        self.assertEqual(self.mesh.axis(), range(self.mesh.size[0]))

    def test_has_site(self):
        self.assertTrue(self.mesh.has_site(0))
        self.assertTrue(self.mesh.has_site(self.mesh.size[0] - 1))
        self.assertFalse(self.mesh.has_site(self.mesh.size[0]))

    def test_has_coordinates(self):
        self.assertTrue(self.mesh.has_coordinates((0, )))
        self.assertTrue(self.mesh.has_coordinates((self.mesh.size[0] - 1, )))
        self.assertFalse(self.mesh.has_coordinates((self.mesh.size[0], )))

    def test_to_site(self):
        self.assertEqual(self.mesh.to_site((0, )), 0)
        self.assertEqual(
            self.mesh.to_site(
                (self.mesh.size[0] - 1, )), self.mesh.size[0] - 1)

    def test_to_coordinates(self):
        self.assertEqual(self.mesh.to_coordinates(0), (0, ))
        self.assertEqual(
            self.mesh.to_coordinates(
                self.mesh.size[0] - 1), (self.mesh.size[0] - 1, ))

    def test_check_steps(self):
        self.assertFalse(self.mesh.check_steps(-1))
        self.assertTrue(self.mesh.check_steps(self.steps))

    def test_str(self):
        self.assertEqual(str(self.mesh),
                         'Segment with dimension {}'.format(self.mesh.size))
        self.assertEqual(str(self.mesh_bl),
                         'Segment with dimension {} and {}'.format(self.mesh_bl.size,
                                                                   str(self.mesh_bl.broken_links)))


if __name__ == '__main__':
    unittest.main()
