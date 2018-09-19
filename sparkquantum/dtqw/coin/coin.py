from datetime import datetime

from pyspark import StorageLevel

from sparkquantum.utils.logger import is_logger
from sparkquantum.utils.profiler import is_profiler
from sparkquantum.utils.utils import Utils

__all__ = ['Coin', 'is_coin']


class Coin:
    """Top-level class for Coins."""

    def __init__(self, spark_context):
        """
        Build a top-level Coin object.

        Parameters
        ----------
        spark_context : SparkContext
            The SparkContext object.

        """
        self._spark_context = spark_context
        self._size = None
        self._data = None

        self._logger = None
        self._profiler = None

    @property
    def spark_context(self):
        return self._spark_context

    @property
    def size(self):
        return self._size

    @property
    def data(self):
        return self._data

    @property
    def logger(self):
        return self._logger

    @property
    def profiler(self):
        return self._profiler

    @logger.setter
    def logger(self, logger):
        """
        Parameters
        ----------
        logger : Logger
            A Logger object or None to disable logging.

        Raises
        ------
        TypeError

        """
        if is_logger(logger) or logger is None:
            self._logger = logger
        else:
            raise TypeError('logger instance expected, not "{}"'.format(type(logger)))

    @profiler.setter
    def profiler(self, profiler):
        """
        Parameters
        ----------
        profiler : Profiler
            A Profiler object or None to disable profiling.

        Raises
        ------
        TypeError

        """
        if is_profiler(profiler) or profiler is None:
            self._profiler = profiler
        else:
            raise TypeError('profiler instance expected, not "{}"'.format(type(profiler)))

    def __str__(self):
        return self.__class__.__name__

    def _profile(self, operator, initial_time):
        if self._profiler is not None:
            app_id = self._spark_context.applicationId

            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            info = self._profiler.profile_operator(
                'coinOperator', operator, (datetime.now() - initial_time).total_seconds()
            )

            if self._logger:
                self._logger.info("coin operator was built in {}s".format(info['buildingTime']))
                self._logger.info(
                    "coin operator is consuming {} bytes in memory and {} bytes in disk".format(
                        info['memoryUsed'], info['diskUsed']
                    )
                )

    def to_string(self):
        return self.__str__()

    def is_1d(self):
        """
        Check if this is a Coin for 1-dimensional meshes.

        Raises
        -------
        NotImplementedError

        """
        raise NotImplementedError

    def is_2d(self):
        """
        Check if this is a Coin for 2-dimensional meshes.

        Raises
        -------
        NotImplementedError

        """
        raise NotImplementedError

    def create_operator(self, mesh, coord_format=Utils.CoordinateDefault, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Build the coin operator.

        Raises
        -------
        NotImplementedError

        """
        raise NotImplementedError


def is_coin(obj):
    """
    Check whether argument is a Coin object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a Coin object, False otherwise.

    """
    return isinstance(obj, Coin)
