import unittest

from tests.base import Base
from sparkquantum.dtqw.coin.coin import is_coin
from sparkquantum.dtqw.coin.hadamard import Hadamard

# TODO: implement test_create_operator


class TestHadamard(Base):
    def setUp(self):
        super().setUp()
        self.m = 2
        self.coin = Hadamard(self.m)

    def test_is_coin(self):
        self.assertTrue(is_coin(self.coin))

    def test_str(self):
        self.assertEqual(str(self.coin), 'Hadamard coin')

    def test_data(self):
        value = (complex() + 1) / 2.0
        data = self.coin.data

        self.assertEqual(data[0][0], value)
        self.assertEqual(data[0][1], value)
        self.assertEqual(data[0][2], value)
        self.assertEqual(data[0][3], value)
        self.assertEqual(data[1][0], value)
        self.assertEqual(data[1][1], -value)
        self.assertEqual(data[1][2], value)
        self.assertEqual(data[1][3], -value)
        self.assertEqual(data[2][0], value)
        self.assertEqual(data[2][1], value)
        self.assertEqual(data[2][2], -value)
        self.assertEqual(data[2][3], -value)
        self.assertEqual(data[3][0], value)
        self.assertEqual(data[3][1], -value)
        self.assertEqual(data[3][2], -value)
        self.assertEqual(data[3][3], value)


if __name__ == '__main__':
    unittest.main()
