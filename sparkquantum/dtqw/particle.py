# from sparkquantum.dtqw.coin.coin import is_coin

__all__ = ['Particle', 'is_particle']


class Particle:
    """Class that represents a particle in a quantum walk."""

    def __init__(self, coin, identifier=None):
        """Build a particle object.

        Parameters
        ----------
        coin : :py:class:`sparkquantum.dtqw.coin.coin.Coin`
            The coin whose operator will be applied on this particle.
        identifier : int or str
            The identifier of the particle. Can be a number, code or name.

        """
        # if not is_coin(coin):
        #     raise TypeError(
        #         "'Coin' instance expected, not '{}'".format(type(self._coin)))

        if identifier is not None and len(str(identifier)) == 0:
            raise ValueError("invalid identifier")

        self._coin = coin
        self._identifier = identifier

    @property
    def coin(self):
        """:py:class:`sparkquantum.dtqw.coin.coin.Coin`"""
        return self._coin

    @property
    def identifier(self):
        """int or str"""
        return self._identifier


def is_particle(obj):
    """Check whether argument is a :py:class:`sparkquantum.dtqw.particle.Particle` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a :py:class:`sparkquantum.dtqw.particle.Particle` object, False otherwise.

    """
    return isinstance(obj, Particle)
