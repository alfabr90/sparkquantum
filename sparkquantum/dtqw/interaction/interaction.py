from datetime import datetime

from pyspark import SparkContext, StorageLevel

from sparkquantum.utils.logger import is_logger
from sparkquantum.utils.profiler import is_profiler
from sparkquantum.utils.utils import Utils

__all__ = ['Interaction', 'is_interaction']


class Interaction:
    """Top-level class for interaction between particles."""

    def __init__(self, num_particles, mesh, logger=None, profiler=None):
        """Build a top-level interaction object.

        Parameters
        ----------
        num_particles : int
            The number of particles present in the walk.
        mesh : :py:class:`sparkquantum.dtqw.mesh.Mesh`
            The mesh where the particles will walk over.
        logger : py:class:`sparkquantum.utils.logger.Logger`, optional
            A logger object.
        profiler : py:class:`sparkquantum.utils.profiler.Profiler`, optional
            A profiler object.

        """
        self._spark_context = SparkContext.getOrCreate()

        self._num_particles = num_particles
        self._mesh = mesh

        self._logger = logger
        self._profiler = profiler

    @property
    def spark_context(self):
        """:py:class:`pyspark.SparkContext`"""
        return self._spark_context

    @property
    def num_particles(self):
        """int"""
        return self._num_particles

    @property
    def mesh(self):
        """:py:class:`sparkquantum.dtqw.mesh.Mesh`"""
        return self._mesh

    @property
    def logger(self):
        """:py:class:`sparkquantum.utils.logger.Logger`.

        To disable logging, set it to None.

        """
        return self._logger

    @property
    def profiler(self):
        """:py:class:`sparkquantum.utils.profiler.Profiler`.

        To disable profiling, set it to None.

        """
        return self._profiler

    @logger.setter
    def logger(self, logger):
        if is_logger(logger) or logger is None:
            self._logger = logger
        else:
            raise TypeError(
                "'Logger' instance expected, not '{}'".format(
                    type(logger)))

    @profiler.setter
    def profiler(self, profiler):
        if is_profiler(profiler) or profiler is None:
            self._profiler = profiler
        else:
            raise TypeError(
                "'Profiler' instance expected, not '{}'".format(
                    type(profiler)))

    def __str__(self):
        return self.__class__.__name__

    def _profile(self, operator, initial_time):
        if self._profiler is not None:
            app_id = self._spark_context.applicationId

            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            info = self._profiler.profile_operator(
                'interactionOperator', operator, (datetime.now(
                ) - initial_time).total_seconds()
            )

            if self._logger:
                self._logger.info(
                    "interaction operator was built in {}s".format(
                        info['buildingTime']))
                self._logger.info(
                    "interaction operator is consuming {} bytes in memory and {} bytes in disk".format(
                        info['memoryUsed'], info['diskUsed']
                    )
                )

    def to_string(self):
        """Build a string representing this interaction between particles.

        Returns
        -------
        str
            The string representation of this interaction between particles.

        """
        return self.__str__()

    def create_operator(self, coord_format=Utils.MatrixCoordinateDefault,
                        storage_level=StorageLevel.MEMORY_AND_DISK):
        """Build the interaction operator.

        Raises
        -------
        NotImplementedError
            This method must not be called from this class, because the successor classes should implement it.

        """
        raise NotImplementedError


def is_interaction(obj):
    """Check whether argument is a :py:class:`sparkquantum.dtqw.interaction.Interaction` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a :py:class:`sparkquantum.dtqw.interaction.Interaction` object, False otherwise.

    """
    return isinstance(obj, Interaction)
