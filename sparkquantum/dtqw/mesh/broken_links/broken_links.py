import random

from pyspark import SparkContext

from sparkquantum.utils.utils import Utils

__all__ = ['BrokenLinks']


class BrokenLinks():
    """Top-level class for broken links."""

    def __init__(self):
        """Build a top-level :py:class:`sparkquantum.dtqw.mesh.broken_links.BrokenLinks` object."""
        self._spark_context = SparkContext.getOrCreate()

        self._logger = Utils.get_logger(
            self._spark_context, self.__class__.__name__)

    def __str__(self):
        """Build a string representing this broken links generator.

        Returns
        -------
        str
            The string representation of this broken links generator.

        """
        return 'Broken Links Generator'

    def is_random(self):
        """Check if this is a random broken links generator.

        Returns
        -------
        bool
            True if this is a random broken links generator, False otherwise.

        """
        return False

    def is_permanent(self):
        """Check if this is a permanent broken links generator.

        Returns
        -------
        bool
            True if this is a permanent broken links generator, False otherwise.

        """
        return False

    def generate(self, num_edges):
        """
        Generate broken links for the mesh.

        Parameters
        ----------
        num_edges : int
            Number of edges of a mesh.

        Raises
        -------
        NotImplementedError
            This method must not be called from this class, because the successor classes should implement it.

        """
        raise NotImplementedError


def is_broken_links(obj):
    """Check whether argument is a :py:class:`sparkquantum.dtqw.mesh.broken_links.BrokenLinks` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a :py:class:`sparkquantum.dtqw.mesh.broken_links.BrokenLinks` object, False otherwise.

    """
    return isinstance(obj, BrokenLinks)
