from datetime import datetime

from pyspark import SparkContext, StorageLevel

from sparkquantum.dtqw.mesh.broken_links.broken_links import is_broken_links
from sparkquantum.utils.logger import is_logger
from sparkquantum.utils.profiler import is_profiler
from sparkquantum.utils.utils import Utils

__all__ = ['Mesh', 'is_mesh']


class Mesh:
    """Top-level class for meshes."""

    def __init__(self, size, broken_links=None):
        """Build a top-level `Mesh` object.

        Parameters
        ----------
        size : int or tuple
            Size of the mesh.
        broken_links : `BrokenLinks`, optional
            A `BrokenLinks` object.

        """
        self._spark_context = SparkContext.getOrCreate()
        self._size = self._define_size(size)
        self._num_edges = self._define_num_edges(size)
        self._coin_size = None
        self._dimension = None

        if broken_links:
            if not is_broken_links(broken_links):
                # self._logger.error("'BrokenLinks' instance expected, not '{}'".format(type(broken_links)))
                raise TypeError("'BrokenLinks' instance expected, not '{}'".format(type(broken_links)))

        self._broken_links = broken_links

        self._logger = None
        self._profiler = None

    @property
    def spark_context(self):
        """`SparkContext`"""
        return self._spark_context

    @property
    def size(self):
        """int or tuple"""
        return self._size

    @property
    def num_edges(self):
        """int"""
        return self._num_edges

    @property
    def broken_links(self):
        """`BrokenLinks`"""
        return self._broken_links

    @property
    def coin_size(self):
        """int"""
        return self._coin_size

    @property
    def dimension(self):
        """int"""
        return self._dimension

    @property
    def logger(self):
        """`Logger`.

        To disable logging, set it to `None`.

        """
        return self._logger

    @property
    def profiler(self):
        """`Profiler`.

        To disable profiling, set it to `None`.

        """
        return self._profiler

    @logger.setter
    def logger(self, logger):
        if is_logger(logger) or logger is None:
            self._logger = logger
        else:
            raise TypeError("'Logger' instance expected, not '{}'".format(type(logger)))

    @profiler.setter
    def profiler(self, profiler):
        if is_profiler(profiler) or profiler is None:
            self._profiler = profiler
        else:
            raise TypeError("'Profiler' instance expected, not '{}'".format(type(profiler)))

    def __str__(self):
        return self.__class__.__name__

    def _validate(self, size):
        raise NotImplementedError

    def _define_size(self, size):
        raise NotImplementedError

    def _define_num_edges(self, size):
        raise NotImplementedError

    def _profile(self, operator, initial_time):
        if self._profiler is not None:
            app_id = self._spark_context.applicationId

            self._profiler.profile_resources(app_id)
            self._profiler.profile_executors(app_id)

            info = self._profiler.profile_operator(
                'shiftOperator', operator, (datetime.now() - initial_time).total_seconds()
            )

            if self._logger:
                self._logger.info("shift operator was built in {}s".format(info['buildingTime']))
                self._logger.info(
                    "shift operator is consuming {} bytes in memory and {} bytes in disk".format(
                        info['memoryUsed'], info['diskUsed']
                    )
                )

    def to_string(self):
        return self.__str__()

    def title(self):
        return self.__str__()

    def filename(self):
        return self.__str__()

    def axis(self):
        raise NotImplementedError

    def is_1d(self):
        """Check if this is a 1-dimensional mesh.

        Returns
        -------
        Bool

        """
        return False

    def is_2d(self):
        """Check if this is a 2-dimensional mesh.

        Returns
        -------
        Bool

        """
        return False

    def check_steps(self, steps):
        """Check if the number of steps is valid for the size of the mesh.

        Parameters
        ----------
        steps : int

        Raises
        -------
        `NotImplementedError`

        """
        raise NotImplementedError

    def create_operator(self, coord_format=Utils.MatrixCoordinateDefault, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Build the mesh operator.

        Parameters
        ----------
        coord_format : int, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is `Utils.MatrixCoordinateDefault`.
        storage_level : `StorageLevel`, optional
            The desired storage level when materializing the RDD. Default value is `StorageLevel.MEMORY_AND_DISK`.

        Raises
        -------
        `NotImplementedError`

        """
        raise NotImplementedError


def is_mesh(obj):
    """Check whether argument is a `Mesh` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a `Mesh` object, False otherwise.

    """
    return isinstance(obj, Mesh)
