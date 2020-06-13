from datetime import datetime

from pyspark import StorageLevel

from sparkquantum.dtqw.mesh.mesh2d.diagonal.diagonal import Diagonal
from sparkquantum.dtqw.operator import Operator
from sparkquantum.utils.utils import Utils

__all__ = ['Box']


class Box(Diagonal):
    """Class for Diagonal Box mesh."""

    def __init__(self, size, broken_links=None):
        """
        Build a Diagonal Box mesh object.

        Parameters
        ----------
        size : tuple
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
        return 'Diagonal Box {}'.format(self.__strcomp__())

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
        size_xy = size[0] * size[1]
        shape = (coin_size * size_xy, coin_size * size_xy)
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
                            l1 = (-1) ** i
                            for j in range(size_per_coin):
                                l2 = (-1) ** j

                                # Finding the correspondent x,y coordinates of
                                # the vertex from the edge number
                                x = (e[1][0] % size[0] - i - l1) % size[0]
                                y = (int(e[1][0] / size[0]) - j - l2) % size[1]

                                if e[1][1]:
                                    bl1, bl2 = 0, 0
                                else:
                                    # The border edges are considered broken so
                                    # that they become reflexive
                                    if x + \
                                            l1 >= size[0] or x + l1 < 0 or y + l2 >= size[1] or y + l2 < 0:
                                        bl1, bl2 = 0, 0
                                    else:
                                        bl1, bl2 = l1, l2

                                m = ((i + bl1) * size_per_coin + (j + bl2)) * \
                                    size_xy + (x + bl1) * size[1] + (y + bl2)
                                n = ((1 - i) * size_per_coin + (1 - j)) * \
                                    size_xy + x * size[1] + y

                                yield m, n, 1
                elif repr_format == Utils.StateRepresentationFormatPositionCoin:
                    def __map(e):
                        """e = (edge, (edge, broken or not))"""
                        for i in range(size_per_coin):
                            l1 = (-1) ** i
                            for j in range(size_per_coin):
                                l2 = (-1) ** j

                                # Finding the correspondent x,y coordinates of
                                # the vertex from the edge number
                                x = (e[1][0] % size[0] - i - l1) % size[0]
                                y = (int(e[1][0] / size[0]) - j - l2) % size[1]

                                if e[1][1]:
                                    bl1, bl2 = 0, 0
                                else:
                                    # The border edges are considered broken so
                                    # that they become reflexive
                                    if x + \
                                            l1 >= size[0] or x + l1 < 0 or y + l2 >= size[1] or y + l2 < 0:
                                        bl1, bl2 = 0, 0
                                    else:
                                        bl1, bl2 = l1, l2

                                m = (
                                    (x + bl1) * size[1] + (y + bl2)) * coin_size + (i + bl1) * size_per_coin + (j + bl2)
                                n = (x * size[1] + y) * coin_size + \
                                    (1 - i) * size_per_coin + (1 - j)

                                yield m, n, 1
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
                            l1 = (-1) ** i
                            for j in range(size_per_coin):
                                l2 = (-1) ** j

                                # Finding the correspondent x,y coordinates of
                                # the vertex from the edge number
                                x = (e % size[0] - i - l1) % size[0]
                                y = (int(e / size[0]) - j - l2) % size[1]

                                if e in broken_links.value:
                                    bl1, bl2 = 0, 0
                                else:
                                    # The border edges are considered broken so
                                    # that they become reflexive
                                    if x + \
                                            l1 >= size[0] or x + l1 < 0 or y + l2 >= size[1] or y + l2 < 0:
                                        bl1, bl2 = 0, 0
                                    else:
                                        bl1, bl2 = l1, l2

                                m = ((i + bl1) * size_per_coin + (j + bl2)) * \
                                    size_xy + (x + bl1) * size[1] + (y + bl2)
                                n = ((1 - i) * size_per_coin + (1 - j)) * \
                                    size_xy + x * size[1] + y

                                yield m, n, 1
                elif repr_format == Utils.StateRepresentationFormatPositionCoin:
                    def __map(e):
                        for i in range(size_per_coin):
                            l1 = (-1) ** i
                            for j in range(size_per_coin):
                                l2 = (-1) ** j

                                # Finding the correspondent x,y coordinates of
                                # the vertex from the edge number
                                x = (e % size[0] - i - l1) % size[0]
                                y = (int(e / size[0]) - j - l2) % size[1]

                                if e in broken_links.value:
                                    bl1, bl2 = 0, 0
                                else:
                                    # The border edges are considered broken so
                                    # that they become reflexive
                                    if x + \
                                            l1 >= size[0] or x + l1 < 0 or y + l2 >= size[1] or y + l2 < 0:
                                        bl1, bl2 = 0, 0
                                    else:
                                        bl1, bl2 = l1, l2

                                m = (
                                    (x + bl1) * size[1] + (y + bl2)) * coin_size + (i + bl1) * size_per_coin + (j + bl2)
                                n = (x * size[1] + y) * coin_size + \
                                    (1 - i) * size_per_coin + (1 - j)

                                yield m, n, 1
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
                def __map(xy):
                    x = xy % size[0]
                    y = int(xy / size[0])

                    for i in range(size_per_coin):
                        l1 = (-1) ** i
                        for j in range(size_per_coin):
                            l2 = (-1) ** j

                            # The border edges are considered broken so that
                            # they become reflexive
                            if x + l1 >= size[0] or x + l1 < 0 or y + \
                                    l2 >= size[1] or y + l2 < 0:
                                bl1, bl2 = 0, 0
                            else:
                                bl1, bl2 = l1, l2

                            m = ((i + bl1) * size_per_coin + (j + bl2)) * \
                                size_xy + (x + bl1) * size[1] + y + bl2
                            n = ((1 - i) * size_per_coin + (1 - j)) * \
                                size_xy + x * size[1] + y

                            yield m, n, 1
            elif repr_format == Utils.StateRepresentationFormatPositionCoin:
                def __map(xy):
                    x = xy % size[0]
                    y = int(xy / size[0])

                    for i in range(size_per_coin):
                        l1 = (-1) ** i
                        for j in range(size_per_coin):
                            l2 = (-1) ** j

                            # The border edges are considered broken so that
                            # they become reflexive
                            if x + l1 >= size[0] or x + l1 < 0 or y + \
                                    l2 >= size[1] or y + l2 < 0:
                                bl1, bl2 = 0, 0
                            else:
                                bl1, bl2 = l1, l2

                            m = ((x + bl1) * size[1] + (y + bl2)) * \
                                coin_size + (i + bl1) * \
                                size_per_coin + (j + bl2)
                            n = (x * size[1] + y) * coin_size + \
                                (1 - i) * size_per_coin + (1 - j)

                            yield m, n, 1
            else:
                self._logger.error("invalid representation format")
                raise ValueError("invalid representation format")

            rdd = self._spark_context.range(
                size_xy
            ).flatMap(
                __map
            )

        return Operator(rdd, shape, data_type=int, num_elements=num_elements)
