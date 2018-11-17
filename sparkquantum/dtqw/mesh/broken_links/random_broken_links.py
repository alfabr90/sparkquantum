import random

from sparkquantum.dtqw.mesh.broken_links.broken_links import BrokenLinks
from sparkquantum.utils.utils import Utils

__all__ = ['RandomBrokenLinks']


class RandomBrokenLinks(BrokenLinks):
    """Class for generating random broken links for a mesh."""

    def __init__(self, spark_context, probability):
        """Build a random broken links generator object.

        Parameters
        ----------
        spark_context : `SparkContext`
            The `SparkContext` object.
        probability : float
            Probability of the occurences of broken links in the mesh.

        """
        super().__init__(spark_context)

        if probability <= 0:
            # self._logger.error("probability of broken links must be positive")
            raise ValueError("probability of broken links must be positive")

        self._probability = probability

    @property
    def probability(self):
        return self._probability

    def generate(self, num_edges):
        """Generate broken links for the mesh based on its probability to have a broken link/edge.

        Returns
        -------
        `RDD` or `Broadcast`
            The `RDD` or `Broadcast` dict which keys are the numbered edges that are broken.

        Raises
        ------
        `ValueError`

        """
        probability = self._probability
        seed = Utils.get_conf(self._spark_context, 'quantum.dtqw.mesh.randomBrokenLinks.seed')

        def __map(e):
            random.seed(seed)
            return e, random.random() < probability

        rdd = self._spark_context.range(
            num_edges
        ).map(
            __map
        ).filter(
            lambda m: m[1] is True
        )

        generation_mode = Utils.get_conf(self._spark_context, 'quantum.dtqw.mesh.brokenLinks.generationMode')

        if generation_mode == 'rdd':
            return rdd
        elif generation_mode == 'broadcast':
            return Utils.broadcast(self._spark_context, rdd.collectAsMap())
        else:
            raise ValueError("invalid broken links generation mode")
