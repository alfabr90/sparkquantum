from datetime import datetime

from pyspark import SparkContext, StorageLevel

from sparkquantum.dtqw.mesh.broken_links.broken_links import is_broken_links
from sparkquantum.utils.utils import Utils

__all__ = ['Mesh', 'is_mesh']


class Mesh:
    """Top-level class for meshes."""

    def __init__(self, size, broken_links=None):
        """Build a top-level mesh object.

        Parameters
        ----------
        size : int or tuple
            Size of the mesh.
        broken_links : :py:class:`sparkquantum.dtqw.mesh.broken_links.BrokenLinks`, optional
            A :py:class:`sparkquantum.dtqw.mesh.broken_links.BrokenLinks` object.

        """
        self._spark_context = SparkContext.getOrCreate()
        self._size = self._define_size(size)
        self._num_edges = self._define_num_edges(size)
        self._coin_size = None
        self._dimension = None

        self._broken_links = broken_links

        self._logger = Utils.get_logger(
            self._spark_context, self.__class__.__name__)

        if broken_links is not None:
            if not is_broken_links(broken_links):
                self._logger.error(
                    "'BrokenLinks' instance expected, not '{}'".format(
                        type(broken_links)))
                raise TypeError(
                    "'BrokenLinks' instance expected, not '{}'".format(
                        type(broken_links)))

    @property
    def spark_context(self):
        """:py:class:`pyspark.SparkContext`"""
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
        """:py:class:`sparkquantum.dtqw.mesh.broken_links.BrokenLinks`"""
        return self._broken_links

    @property
    def coin_size(self):
        """int"""
        return self._coin_size

    @property
    def dimension(self):
        """int"""
        return self._dimension

    def __del__(self):
        # In cases where multiple simulations are performed,
        # the same Python's logger object is used for the same class name.
        # Removing all its handlers, the object is reset.
        for h in self._logger.handlers:
            self._logger.removeHandler(h)

    def __strcomp__(self):
        broken_links = ''

        if self._broken_links is not None:
            broken_links = ' and {}'.format(self._broken_links)

        return 'with dimension {}{}'.format(self._size, broken_links)

    def __str__(self):
        """Build a string representing this mesh.

        Returns
        -------
        str
            The string representation of this mesh.

        """
        return 'Mesh {}'.format(self.__strcomp__())

    def _validate(self, size):
        raise NotImplementedError

    def _define_size(self, size):
        raise NotImplementedError

    def _define_num_edges(self, size):
        raise NotImplementedError

    def center(self):
        """Return the site number of the center of this mesh.

        Raises
        -------
        NotImplementedError
            This method must not be called from this class, because the successor classes should implement it.

        """
        raise NotImplementedError

    def axis(self):
        """Build a generator (or meshgrid) with the size(s) of this mesh.

        Raises
        -------
        NotImplementedError
            This method must not be called from this class, because the successor classes should implement it.

        """
        raise NotImplementedError

    def check_steps(self, steps):
        """Check if the number of steps is valid for the size of the mesh.

        Parameters
        ----------
        steps : int
            Number of steps of the walk.

        Returns
        -------
        bool
            True if this number of steps is valid for the size of the mesh, False otherwise.

        """
        return 0 <= steps

    def create_operator(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Build the mesh operator.

        Parameters
        ----------
        storage_level : :py:class:`pyspark.StorageLevel`, optional
            The desired storage level when materializing the RDD. Default value is :py:const:`pyspark.StorageLevel.MEMORY_AND_DISK`.

        Raises
        -------
        NotImplementedError
            This method must not be called from this class, because the successor classes should implement it.

        """
        raise NotImplementedError


def is_mesh(obj):
    """Check whether argument is a :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh` object, False otherwise.

    """
    return isinstance(obj, Mesh)
