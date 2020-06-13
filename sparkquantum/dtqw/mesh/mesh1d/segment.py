from datetime import datetime

from pyspark import StorageLevel

from sparkquantum.dtqw.mesh.mesh1d.mesh1d import Mesh1D
from sparkquantum.dtqw.operator import Operator
from sparkquantum.utils.utils import Utils

__all__ = ['Segment']


class Segment(Mesh1D):
    """Class for Segment mesh."""

    def __init__(self, size, broken_links=None):
        """Build a Segment mesh object.

        Parameters
        ----------
        size : int
            Size of the mesh.
        broken_links : :py:class:`sparkquantum.dtqw.mesh.broken_links.BrokenLinks`, optional
            A :py:class:`sparkquantum.dtqw.mesh.broken_links.BrokenLinks` object.

        """
        super().__init__(size, broken_links=broken_links)

    def __str__(self):
        """Build a string representing this mesh.

        Returns
        -------
        str
            The string representation of this mesh.

        """
        return 'Segment {}'.format(self.__strcomp__())

    def check_steps(self, steps):
        """Check if the number of steps is valid for the size of the mesh.

        Parameters
        ----------
        steps : int
            Number of steps of the walk.

        Returns
        -------
        bool
            True if this number of steps is valid for the size of the mesh, False otherwise.

        """
        return True

    def create_operator(self):
        """Build the shift operator for the walk.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.operator.Operator`
            The created operator using this mesh.

        Raises
        ------
        ValueError
            If the chosen 'quantum.dtqw.state.representationFormat' configuration is not valid or
            if the chosen 'quantum.dtqw.mesh.brokenLinks.generationMode' configuration is not valid.

        """
        coin_size = self._coin_size
        size_per_coin = int(coin_size / self._dimension)
        size = self._size
        num_edges = self._num_edges
        shape = (coin_size * size, coin_size * size)
        broken_links = None

        num_elements = shape[0]

        repr_format = int(
            Utils.get_conf(
                self._spark_context,
                'quantum.dtqw.state.representationFormat'))

        if self._broken_links:
            broken_links = self._broken_links.generate(num_edges)

            generation_mode = Utils.get_conf(
                self._spark_context,
                'quantum.dtqw.mesh.brokenLinks.generationMode')

            if generation_mode == Utils.BrokenLinksGenerationModeRDD:
                if repr_format == Utils.StateRepresentationFormatCoinPosition:
                    def __map(e):
                        """e = (edge, (edge, broken or not))"""
                        for i in range(size_per_coin):
                            l = (-1) ** i

                            # Finding the correspondent x coordinate of the
                            # vertex from the edge number
                            x = (e[1][0] - i - l) % size

                            if e[1][1]:
                                bl = 0
                            else:
                                if x + l >= size or x + l < 0:
                                    bl = 0
                                else:
                                    bl = l

                            yield (i + bl) * size + x + bl, (1 - i) * size + x, 1
                elif repr_format == Utils.StateRepresentationFormatPositionCoin:
                    def __map(e):
                        """e = (edge, (edge, broken or not))"""
                        for i in range(size_per_coin):
                            l = (-1) ** i

                            # Finding the correspondent x coordinate of the
                            # vertex from the edge number
                            x = (e[1][0] - i - l) % size

                            if e[1][1]:
                                bl = 0
                            else:
                                if x + l >= size or x + l < 0:
                                    bl = 0
                                else:
                                    bl = l

                            yield (x + bl) * coin_size + i + bl, x * coin_size + 1 - i, 1
                else:
                    self._logger.error("invalid representation format")
                    raise ValueError("invalid representation format")

                rdd = self._spark_context.range(
                    num_edges
                ).map(
                    lambda m: (m, m)
                ).leftOuterJoin(
                    broken_links
                ).flatMap(
                    __map
                )
            elif generation_mode == Utils.BrokenLinksGenerationModeBroadcast:
                if repr_format == Utils.StateRepresentationFormatCoinPosition:
                    def __map(e):
                        for i in range(size_per_coin):
                            l = (-1) ** i

                            # Finding the correspondent x coordinate of the
                            # vertex from the edge number
                            x = (e - i - l) % size

                            if e in broken_links.value:
                                bl = 0
                            else:
                                if x + l >= size or x + l < 0:
                                    bl = 0
                                else:
                                    bl = l

                            yield (i + bl) * size + x + bl, (1 - i) * size + x, 1
                elif repr_format == Utils.StateRepresentationFormatPositionCoin:
                    def __map(e):
                        for i in range(size_per_coin):
                            l = (-1) ** i

                            # Finding the correspondent x coordinate of the
                            # vertex from the edge number
                            x = (e - i - l) % size

                            if e in broken_links.value:
                                bl = 0
                            else:
                                if x + l >= size or x + l < 0:
                                    bl = 0
                                else:
                                    bl = l

                            yield (x + bl) * coin_size + i + bl, x * coin_size + 1 - i, 1
                else:
                    self._logger.error("invalid representation format")
                    raise ValueError("invalid representation format")

                rdd = self._spark_context.range(
                    num_edges
                ).flatMap(
                    __map
                )
            else:
                self._logger.error("invalid broken links generation mode")
                raise ValueError("invalid broken links generation mode")
        else:
            if repr_format == Utils.StateRepresentationFormatCoinPosition:
                def __map(x):
                    for i in range(size_per_coin):
                        l = (-1) ** i

                        if x + l >= size or x + l < 0:
                            bl = 0
                        else:
                            bl = l

                        yield (i + bl) * size + x + bl, (1 - i) * size + x, 1
            elif repr_format == Utils.StateRepresentationFormatPositionCoin:
                def __map(x):
                    for i in range(size_per_coin):
                        l = (-1) ** i

                        if x + l >= size or x + l < 0:
                            bl = 0
                        else:
                            bl = l

                        yield (x + bl) * coin_size + i + bl, x * coin_size + (1 - i), 1
            else:
                self._logger.error("invalid representation format")
                raise ValueError("invalid representation format")

            rdd = self._spark_context.range(
                size
            ).flatMap(
                __map
            )

        return Operator(rdd, shape, data_type=int, num_elements=num_elements)
