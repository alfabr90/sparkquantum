import math
import numpy as np

from sparkquantum.dtqw.coin.coin1d.coin1d import Coin1D

__all__ = ['Hadamard1D']


class Hadamard1D(Coin1D):
    """Class that represents the 1-dimensional Hadamard coin."""

    def __init__(self):
        """Build a 1-dimensional Hadamard `Coin` object."""
        super().__init__()

        self._data = np.array(
            [[1, 1],
             [1, -1]], dtype=complex
        ) / math.sqrt(2)
