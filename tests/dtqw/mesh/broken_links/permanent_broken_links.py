import unittest

from tests.base import Base
from sparkquantum.dtqw.mesh.broken_links.broken_links import is_broken_links
from sparkquantum.dtqw.mesh.broken_links.permanent_broken_links import PermanentBrokenLinks

# TODO: implement test_generate


class TestPermanentBrokenLinks(Base):
    def setUp(self):
        super().setUp()
        self.edges = [2, 4]
        self.bl = PermanentBrokenLinks(self.edges)

    def test_is_broken_links(self):
        self.assertTrue(is_broken_links(self.bl))

    def test_str(self):
        self.assertEqual(str(self.bl),
                         'Permanent Broken Links Generator with {} broken links'.format(len(self.edges)))

    def test_is_constant(self):
        self.assertTrue(self.bl.is_constant())

    def test_edges(self):
        self.assertEqual(self.bl.edges, self.edges)


if __name__ == '__main__':
    unittest.main()
