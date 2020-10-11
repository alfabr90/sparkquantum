from datetime import datetime

from pyspark import SparkContext, StorageLevel

from sparkquantum import conf, util
from sparkquantum.dtqw.profiler import QuantumWalkProfiler

__all__ = ['Observable', 'is_observable']


class Observable:
    """Top-level class for observables."""

    def __init__(self):
        """Build a top-level observable object."""
        self._sc = SparkContext.getOrCreate()

        self._logger = util.get_logger(
            self._sc, self.__class__.__name__)
        self._profiler = QuantumWalkProfiler()

    @property
    def profiler(self):
        """:py:class:`sparkquantum.dtqw.profiler.Profiler`."""
        return self._profiler

    def _profile_distribution(
            self, profile_title, log_title, distribution, time):
        app_id = self._sc.applicationId

        self._profiler.profile_resources(app_id)
        self._profiler.profile_executors(app_id)

        info = self._profiler.profile_distribution(
            profile_title, distribution, time)

        if info is not None:
            self._logger.info(
                "{} was done in {}s".format(log_title, info['buildingTime']))
            self._logger.info(
                "probability distribution with {} is consuming {} bytes in memory and {} bytes in disk".format(
                    log_title, info['memoryUsed'], info['diskUsed']
                )
            )

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
                for p in range(state.particles)]

    def measure(self, state, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Perform the measurement of the entire system state.

        Raises
        -------
        NotImplementedError
            This method must not be called from this class, because the successor classes should implement it.

        """
        raise NotImplementedError


def is_observable(obj):
    """Check whether argument is a :py:class:`sparkquantum.dtqw.observable.observable.Observable` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a :py:class:`sparkquantum.dtqw.observable.observable.Observable` object, False otherwise.

    """
    return isinstance(obj, Observable)
