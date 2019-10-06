import random

from sparkquantum.dtqw.mesh.broken_links.broken_links import BrokenLinks
from sparkquantum.utils.utils import Utils

__all__ = ['PermanentBrokenLinks']


class PermanentBrokenLinks(BrokenLinks):
    """Class for permanent broken links of a mesh."""

    def __init__(self, edges):
        """Build a permanent broken links object.

        Parameters
        ----------
        edges : list, tuple or generator
            Collection of the edges that are broken.

        """
        super().__init__()

        if not (isinstance(edges, range) or isinstance(edges, (list, tuple))):
            raise ValueError("invalid edges format")

        if not len(edges):
            # self._logger.error("probability of broken links must be positive")
            raise ValueError("there must be at least one broken edge")

        self._edges = edges

    @property
    def edges(self):
        return self._edges

    def is_permanent(self):
        """Check if this is a permanent broken links generator.

        Returns
        -------
        bool
            True if this is a permanent broken links generator, False otherwise.

        """
        return True

    def generate(self, num_edges):
        """Generate broken links for the mesh based on the informed broken edges.

        Parameters
        ----------
        num_edges : int
            Number of edges of a mesh.

        Returns
        -------
        :py:class:`pyspark.RDD` or :py:class:`pyspark.Broadcast`
            The :py:class:`pyspark.RDD` or :py:class:`pyspark.Broadcast` dict which keys are the numbered edges that are broken,
            depending on the chosen 'quantum.dtqw.mesh.brokenLinks.generationMode' configuration.

        Raises
        ------
        ValueError
            If `num_edges` is out of the bounds of the number of edges of the mesh or
            if the chosen 'quantum.dtqw.mesh.brokenLinks.generationMode' configuration is not valid.

        """
        if isinstance(self._edges, range):
            if self._edges.start < 0 or self._edges.stop > num_edges:
                raise ValueError(
                    "invalid edges for broken links. This mesh supports edges from {} to {}".format(
                        0, num_edges
                    )
                )

            rdd = self._spark_context.range(
                self._edges
            )
        elif isinstance(self._edges, (list, tuple)):
            if min(self._edges) < 0 or max(self._edges) >= num_edges:
                raise ValueError(
                    "invalid edges for broken links. This mesh supports edges from {} to {}".format(
                        0, num_edges
                    )
                )

            rdd = self._spark_context.parallelize(
                self._edges
            )

        rdd = rdd.map(
            lambda m: (m, True)
        )

        generation_mode = Utils.get_conf(self._spark_context, 'quantum.dtqw.mesh.brokenLinks.generationMode')

        if generation_mode == Utils.BrokenLinksGenerationModeRDD:
            return rdd
        elif generation_mode == Utils.BrokenLinksGenerationModeBroadcast:
            return Utils.broadcast(self._spark_context, rdd.collectAsMap())
        else:
            raise ValueError("invalid broken links generation mode")
