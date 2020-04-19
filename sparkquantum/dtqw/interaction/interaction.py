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
        mesh : :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh`
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
        """:py:class:`sparkquantum.dtqw.mesh.mesh.Mesh`"""
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
        """Build a string representing this interaction between particles.

        Returns
        -------
        str
            The string representation of this interaction between particles.

        """
        return self.__class__.__name__

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
