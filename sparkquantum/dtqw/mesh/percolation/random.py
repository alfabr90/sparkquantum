import random

from sparkquantum import constants, util
from sparkquantum.dtqw.mesh.percolation.percolation import Percolation

__all__ = ['Random']


class Random(Percolation):
    """Class for generating random mesh percolations."""

    def __init__(self, prob):
        """Build a random mesh percolations object.

        Parameters
        ----------
        prob : float
            Probability of the occurences of percolations in the mesh. Must be positive.

        """
        if prob <= 0:
            raise ValueError("probability of percolations must be positive")

        super().__init__()

        self._probability = prob

    @property
    def probability(self):
        return self._probability

    def __str__(self):
        return 'Random percolations generator with probability value of {}'.format(
            self._probability)

    def generate(self, edges,
                 perc_mode=constants.PercolationsGenerationModeBroadcast):
        """Generate mesh percolations based on its probability to have a broken link.

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
            If the chosen 'sparkquantum.dtqw.mesh.percolation.generationMode' configuration is not valid.

        """
        prob = self._probability

        rdd = self._sc.range(
            edges
        ).map(
            lambda m: (m, random.random() < prob)
        ).filter(
            lambda m: m[1] is True
        )

        if perc_mode == constants.PercolationsGenerationModeRDD:
            return rdd
        elif perc_mode == constants.PercolationsGenerationModeBroadcast:
            return util.broadcast(self._sc, rdd.collectAsMap())
        else:
            self._logger.error("invalid percolations generation mode")
            raise ValueError("invalid percolations generation mode")
