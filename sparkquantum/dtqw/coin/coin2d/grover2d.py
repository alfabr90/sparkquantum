import numpy as np

from sparkquantum.dtqw.coin.coin2d.coin2d import Coin2D

__all__ = ['Grover2D']


class Grover2D(Coin2D):
    """Class that represents the 2-dimensional Grover coin."""

    def __init__(self):
        """Build a 2-dimensional Grover `Coin` object."""
        super().__init__()

        self._data = np.array(
            [[-1, 1, 1, 1],
             [1, -1, 1, 1],
             [1, 1, -1, 1],
             [1, 1, 1, -1]], dtype=complex
        ) / 2.0
