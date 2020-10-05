from sparkquantum.dtqw.coin.coin import Coin

__all__ = ['Grover']


class Grover(Coin):
    """Class that represents the Grover coin."""

    def __init__(self, size):
        """Build a Grover coin object.

        Parameters
        ----------
        size : int
            The size of the quantum state to build the coin. Must be greater than 1.

        """
        if size < 2:
            raise ValueError("invalid size")

        super().__init__()

        v = 2 / size

        self._data = tuple([tuple([v - (1 if i == j else 0)
                                   for i in range(size)]) for j in range(size)])

    def __str__(self):
        return 'Grover coin'
