from sparkquantum import constants, util
from sparkquantum.dtqw.mesh.percolation.percolation import Percolation

__all__ = ['Permanent', 'is_permanent']


class Permanent(Percolation):
    """Class for generating permanent mesh percolations."""

    def __init__(self, edges):
        """Build a permanent mesh percolations object.

        Parameters
        ----------
        edges : range, tuple, list or generator of int
            Collection of the edges that are broken. Edge numbers must be positive.

        """
        if not len(edges):
            raise ValueError("there must be at least one percolation")

        if min(edges) < 0:
            raise ValueError("edge numbers must be positive")

        super().__init__()

        self._edges = edges

    @property
    def edges(self):
        """tuple of int"""
        return tuple(self._edges)

    def __str__(self):
        edges = len(self._edges)

        if edges > 1:
            percolation = '{} percolations'.format(edges)
        else:
            percolation = '{} percolation'.format(edges)

        return 'Permanent percolations generator with {}'.format(percolation)

    def generate(self, edges,
                 perc_mode=constants.PercolationsGenerationModeBroadcast):
        """Generate mesh percolations based on its probability to have a percolation.

        Parameters
        ----------
        edges : int
            Number of edges of the mesh.
        perc_mode : int, optional
            Indicate how the percolations will be generated.
            Default value is :py:const:`sparkquantum.constants.PercolationsGenerationModeBroadcast`.

        Returns
        -------
        :py:class:`pyspark.RDD` or :py:class:`pyspark.Broadcast`
            The :py:class:`pyspark.RDD` or :py:class:`pyspark.Broadcast` dict which keys are the numbered edges that are broken,
            depending on the chosen 'sparkquantum.dtqw.mesh.percolation.generationMode' configuration.

        Raises
        ------
        ValueError
            If some edge of this object is out of the bounds of the number of edges of the mesh or
            if the chosen 'sparkquantum.dtqw.mesh.percolation.generationMode' configuration is not valid.

        """
        if max(self._edges) >= edges:
            self._logger.error(
                "this mesh supports edges from {} to {}".format(0, edges - 1))
            raise ValueError(
                "this mesh supports edges from {} to {}".format(0, edges - 1))

        if isinstance(self._edges, range):
            rdd = self._sc.range(
                self._edges
            )
        else:
            rdd = self._sc.parallelize(
                self._edges
            )

        rdd = rdd.map(
            lambda m: (m, True)
        )

        if perc_mode == constants.PercolationsGenerationModeRDD:
            return rdd
        elif perc_mode == constants.PercolationsGenerationModeBroadcast:
            return util.broadcast(self._sc, rdd.collectAsMap())
        else:
            self._logger.error("invalid percolations generation mode")
            raise ValueError("invalid percolations generation mode")


def is_permanent(obj):
    """Check whether argument is a :py:class:`sparkquantum.dtqw.mesh.percolation.permanent.Permanent` object.

    Parameters
    ----------
    obj
        Any Python object.

    Returns
    -------
    bool
        True if argument is a :py:class:`sparkquantum.dtqw.mesh.percolation.permanent.Permanent` object, False otherwise.

    """
    return isinstance(obj, Permanent)
