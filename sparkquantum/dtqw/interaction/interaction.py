from datetime import datetime

from pyspark import SparkContext, StorageLevel

from sparkquantum.utils.profiler import is_profiler
from sparkquantum.utils.utils import Utils

__all__ = ['Interaction', 'is_interaction']


class Interaction:
    """Top-level class for interaction between particles."""

    def __init__(self, num_particles, mesh):
        """Build a top-level interaction object.

        Parameters
        ----------
        num_particles : int
            The number of particles present in the walk.
        mesh : :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh`
            The mesh where the particles will walk over.

        """
        self._spark_context = SparkContext.getOrCreate()

        self._num_particles = num_particles
        self._mesh = mesh

        self._logger = Utils.get_logger(
            self._spark_context, self.__class__.__name__)
        self._profiler = None

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
