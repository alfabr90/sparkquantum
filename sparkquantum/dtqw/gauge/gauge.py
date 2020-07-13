from datetime import datetime

from pyspark import SparkContext, StorageLevel

from sparkquantum import util
from sparkquantum.dtqw.profiler import QuantumWalkProfiler

__all__ = ['Gauge', 'is_gauge']


class Gauge:
    """Top-level class for system state measurements (gauge)."""

    def __init__(self):
        """Build a top-level system state measurements (gauge) object."""
        self._spark_context = SparkContext.getOrCreate()

        self._logger = util.get_logger(
            self._spark_context, self.__class__.__name__)
        self._profiler = QuantumWalkProfiler()

    @property
    def profiler(self):
        """:py:class:`sparkquantum.utils.profiler.Profiler`.

        To disable profiling, set it to None.

        """
        return self._profiler

    def _profile_probability_distribution(
            self, profiler_title, log_title, probability_distribution, initial_time):
        app_id = self._spark_context.applicationId

        self._profiler.profile_resources(app_id)
        self._profiler.profile_executors(app_id)

        info = self._profiler.profile_probability_distribution(
            profiler_title,
            probability_distribution,
            (datetime.now() - initial_time).total_seconds())

        if info is not None:
            self._logger.info(
                "{} was done in {}s".format(log_title, info['buildingTime']))
            self._logger.info(
                "Probability distribution with {} is consuming {} bytes in memory and {} bytes in disk".format(
                    log_title, info['memoryUsed'], info['diskUsed']
                )
            )

        if util.get_conf(self._spark_context,
                         'sparkquantum.dtqw.profiler.logExecutors') == 'True':
            self._profiler.log_executors(app_id=app_id)

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
