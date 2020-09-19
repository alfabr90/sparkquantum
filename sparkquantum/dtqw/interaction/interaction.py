from pyspark import SparkContext

from sparkquantum import constants, util

__all__ = ['Interaction', 'is_interaction']


class Interaction:
    """Top-level class for interaction between particles."""

    def __init__(self):
        """Build a top-level interaction object."""
        self._sc = SparkContext.getOrCreate()

        self._logger = util.get_logger(
            self._sc, self.__class__.__name__)

    @property
    def sc(self):
        """:py:class:`pyspark.SparkContext`"""
        return self._sc

    def __del__(self):
        # In cases where multiple simulations are performed,
        # the same Python's logger object is used for the same class name.
        # Removing all its handlers, the object is reset.
        for h in self._logger.handlers:
            self._logger.removeHandler(h)

    def __str__(self):
        """Build a string representing this interaction between particles.

        Returns
        -------
        str
            The string representation of this interaction between particles.

        """
        return self.__class__.__name__

    def create_operator(self, mesh, particles,
                        repr_format=constants.StateRepresentationFormatCoinPosition):
        """Build the interaction operator for a quantum walk.

        Raises
        -------
        NotImplementedError
            This method must not be called from this class, because the successor classes should implement it.

        """
        raise NotImplementedError


def is_interaction(obj):
    """Check whether argument is a :py:class:`sparkquantum.dtqw.interaction.interaction.Interaction` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a :py:class:`sparkquantum.dtqw.interaction.interaction.Interaction` object, False otherwise.

    """
    return isinstance(obj, Interaction)
