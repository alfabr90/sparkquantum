import random

from pyspark import SparkContext

__all__ = ['BrokenLinks']


class BrokenLinks():
    """Top-level class for broken links."""

    def __init__(self):
        """Build a top-level `BrokenLinks` object."""
        self._spark_context = SparkContext.getOrCreate()

    @property
    def spark_context(self):
        return self._spark_context

    def is_random(self):
        """Check if this is a random broken links generator.

        Returns
        -------
        Bool

        """
        return False

    def is_permanent(self):
        """Check if this is a permanent broken links generator.

        Returns
        -------
        Bool

        """
        return False

    def generate(self, num_edges):
        """
        Generate broken links for the mesh.

        Raises
        -------
        NotImplementedError

        """
        raise NotImplementedError


def is_broken_links(obj):
    """Check whether argument is a `BrokenLinks` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a `BrokenLinks` object, False otherwise.

    """
    return isinstance(obj, BrokenLinks)
