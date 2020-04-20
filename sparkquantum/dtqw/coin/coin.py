from datetime import datetime

from pyspark import SparkContext, StorageLevel

from sparkquantum.utils.profiler import is_profiler
from sparkquantum.utils.utils import Utils

__all__ = ['Coin', 'is_coin']


class Coin:
    """Top-level class for coins."""

    def __init__(self):
        """Build a top-level coin object."""
        self._spark_context = SparkContext.getOrCreate()
        self._size = None
        self._data = None

        self._logger = Utils.get_logger(
            self._spark_context, self.__class__.__name__)
        self._profiler = None

    @property
    def spark_context(self):
        """:py:class:`pyspark.SparkContext`"""
        return self._spark_context

    @property
    def size(self):
        """int"""
        return self._size

    @property
    def data(self):
        """:py:class:`pyspark.RDD`"""
        return self._data

    @property
    def profiler(self):
        """:py:class:`sparkquantum.utils.profiler.Profiler`.

        To disable profiling, set it to None.

        """
        return self._profiler

    @profiler.setter
    def profiler(self, profiler):
        if is_profiler(profiler) or profiler is None:
            self._profiler = profiler
        else:
            raise TypeError(
                "'Profiler' instance expected, not '{}'".format(
                    type(profiler)))

    def __str__(self):
        """Build a string representing this coin.

        Returns
        -------
        str
            The string representation of this coin.

        """
        return self.__class__.__name__

    def is_1d(self):
        """Check if this is a coin for one-dimensional meshes.

        Returns
        -------
        Bool
            True if this coin is for one-dimensional meshes, False otherwise.

        """
        return False

    def is_2d(self):
        """Check if this is a Coin for two-dimensional meshes.

        Returns
        -------
        Bool
            True if this coin is for two-dimensional meshes, False otherwise.

        """
        return False

    def create_operator(self, mesh, coord_format=Utils.MatrixCoordinateDefault,
                        storage_level=StorageLevel.MEMORY_AND_DISK):
        """Build the coin operator.

        Raises
        -------
        NotImplementedError
            This method must not be called from this class, because the successor classes should implement it.

        """
        raise NotImplementedError


def is_coin(obj):
    """Check whether argument is a :py:class:`sparkquantum.dtqw.coin.coin.Coin` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a :py:class:`sparkquantum.dtqw.coin.coin.Coin` object, False otherwise.

    """
    return isinstance(obj, Coin)
