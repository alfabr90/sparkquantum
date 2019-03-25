import random

from sparkquantum.dtqw.mesh.broken_links.broken_links import BrokenLinks
from sparkquantum.utils.utils import Utils

__all__ = ['PermanentBrokenLinks']


class PermanentBrokenLinks(BrokenLinks):
    """Class for permanent broken links of a mesh."""

    def __init__(self, spark_session, edges):
        """Build a permanent `BrokenLinks` object.

        Parameters
        ----------
        spark_session : `SparkSession`
            The `SparkSession` object.
        edges : collection
            Collection of the edges that are broken.

        """
        super().__init__(spark_session)

        if not (isinstance(edges, range) or isinstance(edges, (list, tuple))):
            raise ValueError("invalid edges format")

        if not len(edges):
            # self._logger.error("probability of broken links must be positive")
            raise ValueError("there must be at least one broken edge")

        self._edges = edges

    @property
    def edges(self):
        return self._edges

    def generate(self, num_edges):
        """Generate broken links for the mesh based on the informed broken edges.

        Returns
        -------
        `RDD` or `Broadcast`
            The `RDD` or `Broadcast` dict which keys are the numbered edges that are broken.

        Raises
        ------
        `ValueError`

        """
        if isinstance(self._edges, range):
            if self._edges.start < 0 or self._edges.stop > num_edges:
                raise ValueError(
                    "invalid edges for broken links. This mesh supports edges from {} to {}".format(
                        0, num_edges
                    )
                )

            rdd = self._spark_session.sparkContext.range(
                self._edges
            )
        elif isinstance(self._edges, (list, tuple)):
            if min(self._edges) < 0 or max(self._edges) >= num_edges:
                raise ValueError(
                    "invalid edges for broken links. This mesh supports edges from {} to {}".format(
                        0, num_edges
                    )
                )

            rdd = self._spark_session.sparkContext.parallelize(
                self._edges
            )

        rdd = rdd.map(
            lambda m: (m, True)
        )

        generation_mode = Utils.get_conf(self._spark_session, 'quantum.dtqw.mesh.brokenLinks.generationMode')

        if generation_mode == Utils.BrokenLinksGenerationModeRDD:
            return rdd
        elif generation_mode == Utils.BrokenLinksGenerationModeBroadcast:
            return Utils.broadcast(self._spark_session, rdd.collectAsMap())
        else:
            raise ValueError("invalid broken links generation mode")
