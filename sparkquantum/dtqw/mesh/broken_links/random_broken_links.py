import random

from sparkquantum.dtqw.mesh.broken_links.broken_links import BrokenLinks
from sparkquantum.utils.utils import Utils

__all__ = ['RandomBrokenLinks']


class RandomBrokenLinks(BrokenLinks):
    """Class for generating random broken links for a mesh."""

    def __init__(self, probability):
        """Build a random broken links object.

        Parameters
        ----------
        probability : float
            Probability of the occurences of broken links in the mesh.

        """
        super().__init__()

        if probability <= 0:
            self._logger.error("probability of broken links must be positive")
            raise ValueError("probability of broken links must be positive")

        self._probability = probability

    @property
    def probability(self):
        return self._probability

    def __str__(self):
        return 'Random Broken Links Generator with probability value of {}'.format(
            self._probability)

    def is_constant(self):
        """Check if the broken links are constant, i.e., does not change according to any kind of variable.

        Returns
        -------
        bool
            True if the broken links are constant, False otherwise.

        """
        return False

    def generate(self, num_edges):
        """Generate broken links for the mesh based on its probability to have a broken link.

        Parameters
        ----------
        num_edges : int
            Number of edges of the mesh.

        Returns
        -------
        :py:class:`pyspark.RDD` or :py:class:`pyspark.Broadcast`
            The :py:class:`pyspark.RDD` or :py:class:`pyspark.Broadcast` dict which keys are the numbered edges that are broken,
            depending on the chosen 'quantum.dtqw.mesh.brokenLinks.generationMode' configuration.

        Raises
        ------
        ValueError
            If the chosen 'quantum.dtqw.mesh.brokenLinks.generationMode' configuration is not valid.

        """
        probability = self._probability

        rdd = self._spark_context.range(
            num_edges
        ).map(
            lambda m: (m, random.random() < probability)
        ).filter(
            lambda m: m[1] is True
        )

        generation_mode = Utils.get_conf(
            self._spark_context,
            'quantum.dtqw.mesh.brokenLinks.generationMode')

        if generation_mode == Utils.BrokenLinksGenerationModeRDD:
            return rdd
        elif generation_mode == Utils.BrokenLinksGenerationModeBroadcast:
            return Utils.broadcast(self._spark_context, rdd.collectAsMap())
        else:
            self._logger.error("invalid broken links generation mode")
            raise ValueError("invalid broken links generation mode")
