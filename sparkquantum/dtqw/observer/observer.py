from datetime import datetime

from pyspark import SparkContext, StorageLevel

from sparkquantum import conf, util
from sparkquantum.dtqw.profiler import get_profiler

__all__ = ['Observer', 'is_observer']


class Observer:
    """Top-level class for observers."""

    def __init__(self):
        """Build a top-level observer object."""
        self._sc = SparkContext.getOrCreate()

        self._logger = util.get_logger(
            self._sc, self.__class__.__name__)
        self._profiler = get_profiler()

    @property
    def profiler(self):
        """:py:class:`sparkquantum.dtqw.profiler.Profiler`."""
        return self._profiler

    def _profile_distribution(self, profile_title, distribution, time):
        app_id = self._sc.applicationId

        self._profiler.profile_rdd(app_id)
        self._profiler.profile_executors(app_id)
        self._profiler.profile_distribution(profile_title, distribution, time)

    def measure(self, state, particle=None,
                storage_level=StorageLevel.MEMORY_AND_DISK):
        """Perform the measurement of the system state.

        Raises
        -------
        NotImplementedError
            This method must not be called from this class, because the successor classes should implement it.

        """
        raise NotImplementedError


def is_observer(obj):
    """Check whether argument is a :py:class:`sparkquantum.dtqw.observer.observer.Observer` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a :py:class:`sparkquantum.dtqw.observer.observer.Observer` object, False otherwise.

    """
    return isinstance(obj, Observer)
