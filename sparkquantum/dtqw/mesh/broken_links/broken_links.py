import random

__all__ = ['BrokenLinks']


class BrokenLinks():
    """Top-level class for broken links."""

    def __init__(self, spark_session):
        """Build a top-level `BrokenLinks` object.

        Parameters
        ----------
        spark_session : `SparkSession`
            The `SparkSession` object.

        """
        self._spark_session = spark_session

    @property
    def spark_session(self):
        return self._spark_session

    def generate(self, num_edges):
        """
        Generate broken links for the mesh.

        Raises
        -------
        `NotImplementedError`

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
