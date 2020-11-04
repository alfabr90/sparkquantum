# from sparkquantum.dtqw.coin.coin import is_coin

__all__ = ['Particle', 'is_particle']


class Particle:
    """Class that represents a particle in a quantum walk."""

    def __init__(self, coin, name=None):
        """Build a particle object.

        Parameters
        ----------
        coin : :py:class:`sparkquantum.dtqw.coin.coin.Coin`
            The coin whose operator will be applied on this particle.
        name : int or str
            The name of the particle. Can be a number, code or name.

        """
        # if not is_coin(coin):
        #     raise TypeError(
        #         "'Coin' instance expected, not '{}'".format(type(self._coin)))

        if name is not None and len(str(name)) == 0:
            raise ValueError("invalid name")

        self._coin = coin
        self._name = name

    @property
    def coin(self):
        """:py:class:`sparkquantum.dtqw.coin.coin.Coin`"""
        return self._coin

    @property
    def name(self):
        """int or str"""
        return self._name


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
