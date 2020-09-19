from sparkquantum.dtqw.coin.coin import Coin

__all__ = ['Grover']


class Grover(Coin):
    """Class that represents the Grover coin."""

    def __init__(self, ndim):
        """Build a Grover coin object.

        Parameters
        ----------
        ndim : int
            The number of dimensions for the coin. Must be greater than 1.

        """
        if ndim < 2:
            raise ValueError("invalid number of dimensions")

        super().__init__()

        self._ndim = ndim

        n = 2 ** ndim
        v = 2 / n

        self._data = tuple([tuple([v - (1 if i == j else 0)
                                   for i in range(n)]) for j in range(n)])

    def __str__(self):
        return '{}d grover coin'.format(self._ndim)
