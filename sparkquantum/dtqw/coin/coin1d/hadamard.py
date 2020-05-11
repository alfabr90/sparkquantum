import math
import numpy as np

from sparkquantum.dtqw.coin.coin1d.coin1d import Coin1D

__all__ = ['Hadamard']


class Hadamard(Coin1D):
    """Class that represents the one-dimensional Hadamard coin."""

    def __init__(self):
        """Build a one-dimensional Hadamard coin object."""
        super().__init__()

        self._data = np.array(
            [[1, 1],
             [1, -1]], dtype=complex
        ) / math.sqrt(2)

    def __str__(self):
        return 'One-dimensional Hadamard Coin'
