import unittest

from tests.base import Base
from sparkquantum.dtqw.mesh.broken_links.broken_links import is_broken_links
from sparkquantum.dtqw.mesh.broken_links.random import RandomBrokenLinks

# TODO: implement test_generate


class TestRandomBrokenLinks(Base):
    def setUp(self):
        super().setUp()
        self.prob = 0.1
        self.bl = RandomBrokenLinks(self.prob)

    def test_is_broken_links(self):
        self.assertTrue(is_broken_links(self.bl))

    def test_str(self):
        self.assertEqual(str(self.bl),
                         'Random Broken Links Generator with probability value of {}'.format(self.prob))

    def test_is_constant(self):
        self.assertFalse(self.bl.is_constant())

    def test_probability(self):
        self.assertEqual(self.bl.probability, self.prob)


if __name__ == '__main__':
    unittest.main()
