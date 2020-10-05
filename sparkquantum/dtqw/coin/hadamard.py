import math

from sparkquantum.dtqw.coin.coin import Coin

__all__ = ['Hadamard']


class Hadamard(Coin):
    """Class that represents the Hadamard coin."""

    def __init__(self, m):
        """Build a Hadamard coin object.

        Parameters
        ----------
        m : int
            The number of recursions to build the coin. Must be positive.

        """
        if m < 1:
            raise ValueError("invalid number of recursions")

        super().__init__()

        size = 2 ** m
        sq = math.sqrt(size)

        self._data = tuple(
            [tuple([(-1) ** (format(i & j, 'b').count("1")) / sq for i in range(size)]) for j in range(size)])

    def __str__(self):
        return 'Hadamard coin'
