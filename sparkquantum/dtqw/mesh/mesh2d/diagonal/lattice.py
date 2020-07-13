from datetime import datetime
import numpy as np

from pyspark import StorageLevel

from sparkquantum import util
from sparkquantum.dtqw.mesh.mesh2d.diagonal.diagonal import Diagonal
from sparkquantum.dtqw.operator import Operator

__all__ = ['Lattice']


class Lattice(Diagonal):
    """Class for Diagonal Lattice mesh."""

    def __init__(self, size, broken_links=None):
        """Build a Diagonal Lattice mesh object.

        Parameters
        ----------
        size : tuple
            Size of the mesh.
        broken_links : :py:class:`sparkquantum.dtqw.mesh.broken_links.BrokenLinks`, optional
            A :py:class:`sparkquantum.dtqw.mesh.broken_links.BrokenLinks` object.

        """
        super().__init__(size, broken_links=broken_links)

    def _define_size(self, size):
        self._validate(size)
        return 2 * size[0] + 1, 2 * size[1] + 1

    def __str__(self):
        """Build a string representing this mesh.

        Returns
        -------
        str
            The string representation of this mesh.

        """
        return 'Diagonal Lattice {}'.format(self.__strcomp__())

    def axis(self):
        """Build a meshgrid with the sizes of this mesh.

        Returns
        -------
        :py:class:`numpy.ndarray`
            The meshgrid with the sizes of this mesh.

        """
        center_x, center_y = self.center_coordinates()

        return np.meshgrid(
            range(- center_x, center_x + 1),
            range(- center_y, center_y + 1),
            indexing='ij'
        )

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
        center_x, center_y = self.center_coordinates()

        return super().check_steps(steps) and steps <= center_x and steps <= center_y

    def create_operator(self):
        """Build the shift operator for the walk.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.operator.Operator`
            The created operator using this mesh.

        Raises
        ------
        ValueError
            If the chosen 'sparkquantum.dtqw.state.representationFormat' configuration is not valid or
            if the chosen 'sparkquantum.dtqw.mesh.brokenLinks.generationMode' configuration is not valid.

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
            util.get_conf(
                self._spark_context,
                'sparkquantum.dtqw.state.representationFormat'))

        if self._broken_links:
            broken_links = self._broken_links.generate(num_edges)

            generation_mode = util.get_conf(
                self._spark_context,
                'sparkquantum.dtqw.mesh.brokenLinks.generationMode')

            if generation_mode == util.BrokenLinksGenerationModeRDD:
                if repr_format == util.StateRepresentationFormatCoinPosition:
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
                                    bl1, bl2 = l1, l2

                                m = ((i + bl1) * size_per_coin + (j + bl2)) * size_xy + \
                                    ((x + bl1) %
                                     size[0]) * size[1] + ((y + bl2) %
                                                           size[1])
                                n = ((1 - i) * size_per_coin + (1 - j)) * \
                                    size_xy + x * size[1] + y

                                yield m, n, 1
                elif repr_format == util.StateRepresentationFormatPositionCoin:
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
                                    bl1, bl2 = l1, l2

                                m = (((x + bl1) % size[0]) * size[1] + ((y + bl2) % size[1])) * coin_size + \
                                    (i + bl1) * size_per_coin + (j + bl2)
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
            elif generation_mode == util.BrokenLinksGenerationModeBroadcast:
                if repr_format == util.StateRepresentationFormatCoinPosition:
                    def __map(e):
                        """e = (edge, (edge, broken or not))"""
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
                                    bl1, bl2 = l1, l2

                                m = ((i + bl1) * size_per_coin + (j + bl2)) * size_xy + \
                                    ((x + bl1) %
                                     size[0]) * size[1] + ((y + bl2) %
                                                           size[1])
                                n = ((1 - i) * size_per_coin + (1 - j)) * \
                                    size_xy + x * size[1] + y

                                yield m, n, 1
                elif repr_format == util.StateRepresentationFormatPositionCoin:
                    def __map(e):
                        """e = (edge, (edge, broken or not))"""
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
                                    bl1, bl2 = l1, l2

                                m = (((x + bl1) % size[0]) * size[1] + ((y + bl2) % size[1])) * coin_size + \
                                    (i + bl1) * size_per_coin + (j + bl2)
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
            if repr_format == util.StateRepresentationFormatCoinPosition:
                def __map(xy):
                    x = xy % size[0]
                    y = int(xy / size[0])

                    for i in range(size_per_coin):
                        l1 = (-1) ** i
                        for j in range(size_per_coin):
                            l2 = (-1) ** j

                            m = (i * size_per_coin + j) * size_xy + ((x + l1) %
                                                                     size[0]) * size[1] + ((y + l2) % size[1])
                            n = (i * size_per_coin + j) * \
                                size_xy + x * size[1] + y

                            yield m, n, 1
            elif repr_format == util.StateRepresentationFormatPositionCoin:
                def __map(xy):
                    x = xy % size[0]
                    y = int(xy / size[0])

                    for i in range(size_per_coin):
                        l1 = (-1) ** i
                        for j in range(size_per_coin):
                            l2 = (-1) ** j

                            m = (((x + l1) %
                                  size[0]) * size[1] + ((y + l2) %
                                                        size[1])) * coin_size + i * size_per_coin + j
                            n = (x * size[1] + y) * coin_size + \
                                i * size_per_coin + j

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
