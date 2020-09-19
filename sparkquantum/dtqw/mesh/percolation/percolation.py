import random

from pyspark import SparkContext

from sparkquantum import constants, util

__all__ = ['Percolation', 'is_percolation']


class Percolation():
    """Top-level class for mesh percolations."""

    def __init__(self):
        """Build a top-level mesh percolations object."""
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
        """Build a string representing this percolations generator.

        Returns
        -------
        str
            The string representation of this percolation generator.

        """
        return 'Percolations generator'

    def generate(self, edges,
                 perc_mode=constants.PercolationsGenerationModeBroadcast):
        """Generate mesh percolations.

        Parameters
        ----------
        edges : int
            Number of edges of a mesh.
        perc_mode : int, optional
            Indicate how the percolations will be generated.
            Default value is :py:const:`sparkquantum.constants.PercolationsGenerationModeBroadcast`.

        Raises
        -------
        NotImplementedError
            This method must not be called from this class, because the successor classes should implement it.

        """
        raise NotImplementedError


def is_percolation(obj):
    """Check whether argument is a :py:class:`sparkquantum.dtqw.mesh.percolation.percolation.Percolation` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a :py:class:`sparkquantum.dtqw.mesh.percolation.percolation.Percolation` object, False otherwise.

    """
    return isinstance(obj, Percolation)
