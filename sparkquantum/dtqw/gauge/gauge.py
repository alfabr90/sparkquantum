from datetime import datetime

from pyspark import SparkContext, StorageLevel

from sparkquantum.utils.profiler import is_profiler
from sparkquantum.utils.utils import Utils

__all__ = ['Gauge', 'is_gauge']


class Gauge:
    """Top-level class for system state measurements (gauge)."""

    def __init__(self):
        """Build a top-level system state measurements (gauge) object."""
        self._spark_context = SparkContext.getOrCreate()

        self._logger = Utils.get_logger(
            self._spark_context, self.__class__.__name__)
        self._profiler = None

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
                "'Profiler' instance expected, not '{}'".format(type(profiler)))

    def measure_system(
            self, state, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Perform the measurement of the entire system state.

        Raises
        -------
        NotImplementedError
            This method must not be called from this class, because the successor classes should implement it.

        """
        raise NotImplementedError

    def measure_particle(self, state, particle,
                         storage_level=StorageLevel.MEMORY_AND_DISK):
        """Perform the partial measurement of a particle of the system state.

        Raises
        -------
        NotImplementedError
            This method must not be called from this class, because the successor classes should implement it.

        """
        raise NotImplementedError

    def measure_particles(
            self, state, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Perform the partial measurement of each particle of the system state.

        Raises
        -------
        NotImplementedError
            This method must not be called from this class, because the successor classes should implement it.

        """
        return [self.measure_particle(state, p, storage_level)
                for p in range(state.num_particles)]

    def measure(self, state, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Perform the measurement of the system state.

        Raises
        -------
        NotImplementedError
            This method must not be called from this class, because the successor classes should implement it.

        """
        raise NotImplementedError


def is_gauge(obj):
    """Check whether argument is a :py:class:`sparkquantum.dtqw.gauge.Gauge` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a :py:class:`sparkquantum.dtqw.gauge.Gauge` object, False otherwise.

    """
    return isinstance(obj, Gauge)
