import numpy as np

from sparkquantum.dtqw.coin.coin2d.coin2d import Coin2D

__all__ = ['Hadamard']


class Hadamard(Coin2D):
    """Class that represents the two-dimensional Hadamard coin."""

    def __init__(self):
        """Build a two-dimensional Hadamard coin object."""
        super().__init__()

        self._data = np.array(
            [[1, 1, 1, 1],
             [1, -1, 1, -1],
             [1, 1, -1, -1],
             [1, -1, -1, 1]], dtype=complex
        ) / 2.0

    def __str__(self):
        return 'Two-dimensional Hadamard Coin'