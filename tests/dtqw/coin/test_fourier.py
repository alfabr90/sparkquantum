import unittest

from tests.base import Base
from sparkquantum.dtqw.coin.coin import is_coin
from sparkquantum.dtqw.coin.fourier import Fourier

# TODO: implement test_create_operator


class TestFourier(Base):
    def setUp(self):
        super().setUp()
        self.ndim = 2
        self.coin = Fourier(self.ndim)

    def test_is_coin(self):
        self.assertTrue(is_coin(self.coin))

    def test_str(self):
        self.assertEqual(str(self.coin), '{}d fourier coin'.format(self.ndim))

    def test_ndim(self):
        self.assertEqual(self.coin.ndim, self.ndim)

    def test_data(self):
        value_r = (complex() + 1) / 2
        value_i = (complex() + 1j) / 2
        data = self.coin.data

        self.assertEqual(data[0][0], value_r)
        self.assertEqual(data[0][1], value_r)
        self.assertEqual(data[0][2], value_r)
        self.assertEqual(data[0][3], value_r)
        self.assertEqual(data[1][0], value_r)
        self.assertEqual(data[1][1], value_i)
        self.assertEqual(data[1][2], -value_r)
        self.assertEqual(data[1][3], -value_i)
        self.assertEqual(data[2][0], value_r)
        self.assertEqual(data[2][1], -value_r)
        self.assertEqual(data[2][2], value_r)
        self.assertEqual(data[2][3], -value_r)
        self.assertEqual(data[3][0], value_r)
        self.assertEqual(data[3][1], -value_i)
        self.assertEqual(data[3][2], -value_r)
        self.assertEqual(data[3][3], value_i)


if __name__ == '__main__':
    unittest.main()
