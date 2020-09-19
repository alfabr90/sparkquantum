import math

from sparkquantum.dtqw.coin.coin import Coin

__all__ = ['Hadamard']


class Hadamard(Coin):
    """Class that represents the Hadamard coin."""

    def __init__(self, ndim):
        """Build a Hadamard coin object.

        Parameters
        ----------
        ndim : int
            The number of dimensions for the coin. Must be positive.

        """
        if ndim < 1:
            raise ValueError("invalid number of dimensions")

        super().__init__()

        self._ndim = ndim

        n = 2 ** ndim
        sq = math.sqrt(n)

        self._data = tuple(
            [tuple([(-1) ** (format(i & j, 'b').count("1")) / sq for i in range(n)]) for j in range(n)])

    def __str__(self):
        return '{}d hadamard coin'.format(self._ndim)
