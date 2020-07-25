import unittest

from tests.base import Base
from sparkquantum import util


class TestUtil(Base):
    def setUp(self):
        super().setUp()

    def test_broadcast(self):
        sc = self.spark_context
        self.assertEqual(util.broadcast(sc, 1).value, 1)
        self.assertEqual(util.broadcast(sc, 1.0).value, 1.0)
        self.assertEqual(util.broadcast(sc, True).value, True)
        self.assertEqual(util.broadcast(sc, "broadcast").value, "broadcast")
        self.assertEqual(util.broadcast(sc, (1, 2)).value, (1, 2))
        self.assertListEqual(util.broadcast(sc, [1, 2]).value, [1, 2])

    def test_get_precedent_type(self):
        self.assertEqual(util.get_precedent_type(complex, complex), complex)
        self.assertEqual(util.get_precedent_type(complex, float), complex)
        self.assertEqual(util.get_precedent_type(float, complex), complex)
        self.assertEqual(util.get_precedent_type(float, float), float)
        self.assertEqual(util.get_precedent_type(complex, int), complex)
        self.assertEqual(util.get_precedent_type(int, complex), complex)
        self.assertEqual(util.get_precedent_type(int, int), int)
        self.assertEqual(util.get_precedent_type(float, int), float)
        self.assertEqual(util.get_precedent_type(int, float), float)


if __name__ == '__main__':
    unittest.main()
