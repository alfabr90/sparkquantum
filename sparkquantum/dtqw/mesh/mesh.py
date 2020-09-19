from pyspark import SparkContext

from sparkquantum import constants, util
from sparkquantum.dtqw.mesh.percolation.percolation import is_percolation

__all__ = ['Mesh', 'is_mesh']


class Mesh:
    """Top-level class for meshes."""

    def __init__(self, percolation=None):
        """Build a top-level mesh object.

        Parameters
        ----------
        percolation : :py:class:`sparkquantum.dtqw.mesh.percolation.Percolation`, optional
            A percolation object.

        """
        if percolation is not None and not is_percolation(percolation):
            raise TypeError(
                "'Percolation' instance expected, not '{}'".format(type(percolation)))

        self._sc = SparkContext.getOrCreate()

        self._percolation = percolation

        self._logger = util.get_logger(
            self._sc, self.__class__.__name__)

    @property
    def sc(self):
        """:py:class:`pyspark.SparkContext`"""
        return self._sc

    @property
    def percolation(self):
        """:py:class:`sparkquantum.dtqw.mesh.percolation.Percolation`"""
        return self._percolation

    def __del__(self):
        # In cases where multiple simulations are performed,
        # the same Python's logger object is used for the same class name.
        # Removing all its handlers, the object is reset.
        for h in self._logger.handlers:
            self._logger.removeHandler(h)

    def __str__(self):
        """Build a string representing this mesh.

        Returns
        -------
        str
            The string representation of this mesh.

        """
        percolation = ''

        if self._percolation is not None:
            percolation = ' with {}'.format(self._percolation)

        return '{}{}'.format(self.__class__.__name__, percolation)

    def has_site(self, site):
        """Indicate whether this mesh comprises a site.

        Parameters
        ----------
        site : int
            Site number.

        Raises
        -------
        NotImplementedError
            This method must not be called from this class, because the successor classes should implement it.

        """
        raise NotImplementedError

    def create_operator(self, cspace,
                        repr_format=constants.StateRepresentationFormatCoinPosition):
        """Build the shift operator for a quantum walk.

        Parameters
        ----------
        cspace : int
            The size of the coin space.
        repr_format : int, optional
            Indicate how the quantum system is represented.
            Default value is :py:const:`sparkquantum.constants.StateRepresentationFormatCoinPosition`.

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
