import numpy as np

from sparkquantum.dtqw.coin.coin2d.coin2d import Coin2D

__all__ = ['Fourier']


class Fourier(Coin2D):
    """Class that represents the two-dimensional Fourier coin."""

    def __init__(self):
        """Build a two-dimensional Fourier coin object."""
        super().__init__()

        self._data = np.array(
            [[1, 1, 1, 1],
             [1, 1.0j, -1, -1.0j],
             [1, -1, 1, -1],
             [1, -1.0j, -1, 1.0j]], dtype=complex
        ) / 2.0

    def __str__(self):
        return 'Two-dimensional Fourier Coin'
