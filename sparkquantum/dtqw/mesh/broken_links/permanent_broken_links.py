import random

from sparkquantum import util
from sparkquantum.dtqw.mesh.broken_links.broken_links import BrokenLinks

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
            self._logger.error(
                "'list' or 'tuple' expected, not '{}'".format(
                    type(edges)))
            raise TypeError(
                "'list' or 'tuple' expected, not '{}'".format(
                    type(edges)))

        if not len(edges):
            self._logger.error("there must be at least one broken edge")
            raise ValueError("there must be at least one broken edge")

        self._edges = edges

    @property
    def edges(self):
        return self._edges

    def __str__(self):
        num_edges = len(self._edges)

        if num_edges > 1:
            broken_links = '{} broken links'.format(num_edges)
        else:
            broken_links = '{} broken link'.format(num_edges)

        return 'Permanent Broken Links Generator with {}'.format(broken_links)

    def is_constant(self):
        """Check if the broken links are constant, i.e., does not change according to any kind of variable.

        Returns
        -------
        bool
            True if the broken links are constant, False otherwise.

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
            depending on the chosen 'sparkquantum.dtqw.mesh.brokenLinks.generationMode' configuration.

        Raises
        ------
        ValueError
            If `num_edges` is out of the bounds of the number of edges of the mesh or
            if the chosen 'sparkquantum.dtqw.mesh.brokenLinks.generationMode' configuration is not valid.

        """
        if isinstance(self._edges, range):
            if self._edges.start < 0 or self._edges.stop > num_edges:
                self._logger.error(
                    "invalid edges for broken links. This mesh supports edges from {} to {}".format(
                        0, num_edges))
                raise ValueError(
                    "invalid edges for broken links. This mesh supports edges from {} to {}".format(
                        0, num_edges))

            rdd = self._spark_context.range(
                self._edges
            )
        elif isinstance(self._edges, (list, tuple)):
            if min(self._edges) < 0 or max(self._edges) >= num_edges:
                self._logger.error(
                    "invalid edges for broken links. This mesh supports edges from {} to {}".format(
                        0, num_edges))
                raise ValueError(
                    "invalid edges for broken links. This mesh supports edges from {} to {}".format(
                        0, num_edges))

            rdd = self._spark_context.parallelize(
                self._edges
            )

        rdd = rdd.map(
            lambda m: (m, True)
        )

        generation_mode = util.get_conf(
            self._spark_context,
            'sparkquantum.dtqw.mesh.brokenLinks.generationMode')

        if generation_mode == util.BrokenLinksGenerationModeRDD:
            return rdd
        elif generation_mode == util.BrokenLinksGenerationModeBroadcast:
            return util.broadcast(self._spark_context, rdd.collectAsMap())
        else:
            self._logger.error("invalid broken links generation mode")
            raise ValueError("invalid broken links generation mode")
