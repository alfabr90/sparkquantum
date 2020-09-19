import math

from sparkquantum.dtqw.coin.coin import Coin

__all__ = ['Fourier']


class Fourier(Coin):
    """Class that represents the Fourier coin."""

    def __init__(self, ndim, phase=1j):
        """Build a Fourier coin object.

        Parameters
        ----------
        ndim : int
            The number of dimensions for the coin. Must be greater than 1.
        phase : complex
            The phase of the fourier transform.

        """
        if ndim < 2:
            raise ValueError("invalid number of dimensions")

        super().__init__()

        self._ndim = ndim

        n = 2 ** ndim
        sq = math.sqrt(n)

        self._data = tuple(
            [tuple([phase ** (i * j) / sq for i in range(n)]) for j in range(n)])

    def __str__(self):
        return '{}d fourier coin'.format(self._ndim)
