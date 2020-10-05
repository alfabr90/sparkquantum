import math

from sparkquantum.dtqw.coin.coin import Coin

__all__ = ['Fourier']


class Fourier(Coin):
    """Class that represents the Fourier coin."""

    def __init__(self, n, phase=1j):
        """Build a Fourier coin object.

        Parameters
        ----------
        n : int
            The number of points of the DFT to build the coin. Must be greater than 1.
        phase : complex
            The phase of the fourier transform.

        """
        if n < 2:
            raise ValueError("invalid number of points")

        super().__init__()

        size = 2 ** n
        sq = math.sqrt(size)

        self._data = tuple(
            [tuple([phase ** (i * j) / sq for i in range(size)]) for j in range(size)])

    def __str__(self):
        return 'Fourier coin'
