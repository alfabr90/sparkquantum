from datetime import datetime
import numpy as np

from pyspark import StorageLevel

from sparkquantum import conf, constants, util
from sparkquantum.dtqw.mesh.mesh2d.natural.natural import Natural
from sparkquantum.dtqw.operator import Operator

__all__ = ['Lattice']


class Lattice(Natural):
    """Class for Natural Lattice mesh."""

    def __init__(self, size, broken_links=None):
        """Build a Natural Lattice mesh object.

        Parameters
        ----------
        size : tuple or list of int
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
        return 'Natural Lattice {}'.format(self.__strcomp__())

    def center_coordinates(self):
        """Return the coordinates of the center site of this mesh.

        Returns
        -------
        tuple or list
            The coordinates of the center site.

        """
        return (0, 0)

    def axis(self):
        """Build a meshgrid with the sizes of this mesh.

        Returns
        -------
        :py:class:`numpy.ndarray`
            The meshgrid with the sizes of this mesh.

        """
        center_x = int((self._size[0] - 1) / 2)
        center_y = int((self._size[1] - 1) / 2)

        return np.meshgrid(
            range(- center_x, center_x + 1),
            range(- center_y, center_y + 1),
            indexing='ij'
        )

    def has_coordinates(self, coordinate):
        """Indicate whether the coordinates are inside this mesh.

        Parameters
        ----------
        coordinate : tuple or list
            The coordinates.

        Returns
        -------
        bool
            True if this mesh comprises the coordinates, False otherwise.

        """
        center_x = int((self._size[0] - 1) / 2)
        center_y = int((self._size[1] - 1) / 2)

        return (coordinate[0] >= -center_x and coordinate[0] <= center_x and
                coordinate[1] >= -center_y and coordinate[1] <= center_y)

    def to_site(self, coordinate):
        """Get the site number from the correspondent coordinates.

        Parameters
        ----------
        coordinate : tuple or list
            The coordinates.

        Returns
        -------
        int
            The site number.

        Raises
        ------
        ValueError
            If the coordinates are out of the mesh boundaries.

        """
        center_x = int((self._size[0] - 1) / 2)
        center_y = int((self._size[1] - 1) / 2)

        size_x = coordinate[0] + center_x
        size_y = coordinate[1] + center_y

        if (size_x < 0 or size_x > self._size[0]
                or size_y < 0 or size_y > self._size[1]):
            self._logger.error("coordinates out of mesh boundaries")
            raise ValueError("coordinates out of mesh boundaries")

        return size_x * self._size[1] + size_y

    def to_coordinates(self, site):
        """Get the coordinates from the correspondent site.

        Parameters
        ----------
        site : int
            Site number.

        Raises
        -------
        tuple or list
            The coordinates.

        Raises
        ------
        ValueError
            If the site number is out of the mesh boundaries.

        """
        if site < 0 or site >= self._size[0] * self._size[1]:
            self._logger.error("site number out of mesh boundaries")
            raise ValueError("site number out of mesh boundaries")

        center_x = int((self._size[0] - 1) / 2)
        center_y = int((self._size[1] - 1) / 2)

        return (int(site / self._size[1]) - center_x,
                site % self._size[1] - center_y)

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
        center_x = int((self._size[0] - 1) / 2)
        center_y = int((self._size[1] - 1) / 2)

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
            conf.get_conf(
                self._spark_context,
                'sparkquantum.dtqw.state.representationFormat'))

        if self._broken_links:
            broken_links = self._broken_links.generate(num_edges)

            generation_mode = conf.get_conf(
                self._spark_context,
                'sparkquantum.dtqw.mesh.brokenLinks.generationMode')

            if generation_mode == constants.BrokenLinksGenerationModeRDD:
                if repr_format == constants.StateRepresentationFormatCoinPosition:
                    def __map(e):
                        """e = (edge, (edge, broken or not))"""
                        for i in range(size_per_coin):
                            l = (-1) ** i

                            # Finding the correspondent x,y coordinates of the
                            # vertex from the edge number
                            if e[1][0] >= size[0] * size[1]:
                                j = i
                                x = int(
                                    (e[1][0] - size[0] * size[1]) / size[0])
                                y = ((e[1][0] - size[0] * size[1]) %
                                     size[1] - i - l) % size[1]
                            else:
                                j = int(not i)
                                x = (e[1][0] % size[0] - i - l) % size[0]
                                y = int(e[1][0] / size[0])

                            delta = int(not (i ^ j))

                            if e[1][1]:
                                l = 0

                            m = ((i + l) * size_per_coin + (abs(j + l) % size_per_coin)) * size_xy + \
                                ((x + l * (1 - delta)) %
                                 size[0]) * size[1] + (y + l * delta) % size[1]
                            n = ((1 - i) * size_per_coin + (1 - j)) * \
                                size_xy + x * size[1] + y

                            yield m, n, 1
                elif repr_format == constants.StateRepresentationFormatPositionCoin:
                    def __map(e):
                        """e = (edge, (edge, broken or not))"""
                        for i in range(size_per_coin):
                            l = (-1) ** i

                            # Finding the correspondent x,y coordinates of the
                            # vertex from the edge number
                            if e[1][0] >= size[0] * size[1]:
                                j = i
                                x = int(
                                    (e[1][0] - size[0] * size[1]) / size[0])
                                y = ((e[1][0] - size[0] * size[1]) %
                                     size[1] - i - l) % size[1]
                            else:
                                j = int(not i)
                                x = (e[1][0] % size[0] - i - l) % size[0]
                                y = int(e[1][0] / size[0])

                            delta = int(not (i ^ j))

                            if e[1][1]:
                                l = 0

                            m = (((x + l * (1 - delta)) % size[0]) * size[1] + (y + l * delta) % size[1]) * coin_size + \
                                (i + l) * size_per_coin + \
                                (abs(j + l) % size_per_coin)
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
            elif generation_mode == constants.BrokenLinksGenerationModeBroadcast:
                if repr_format == constants.StateRepresentationFormatCoinPosition:
                    def __map(e):
                        """e = (edge, (edge, broken or not))"""
                        for i in range(size_per_coin):
                            l = (-1) ** i

                            # Finding the correspondent x,y coordinates of the
                            # vertex from the edge number
                            if e >= size[0] * size[1]:
                                j = i
                                delta = int(not (i ^ j))
                                x = int((e - size[0] * size[1]) / size[0])
                                y = ((e - size[0] * size[1]) %
                                     size[1] - i - l) % size[1]
                            else:
                                j = int(not i)
                                delta = int(not (i ^ j))
                                x = (e % size[0] - i - l) % size[0]
                                y = int(e / size[0])

                            if e in broken_links.value:
                                bl = 0
                            else:
                                bl = l

                            m = ((i + bl) * size_per_coin + (abs(j + bl) % size_per_coin)) * size_xy + \
                                ((x + bl * (1 - delta)) %
                                 size[0]) * size[1] + (y + bl * delta) % size[1]
                            n = ((1 - i) * size_per_coin + (1 - j)) * \
                                size_xy + x * size[1] + y

                            yield m, n, 1
                elif repr_format == constants.StateRepresentationFormatPositionCoin:
                    def __map(e):
                        """e = (edge, (edge, broken or not))"""
                        for i in range(size_per_coin):
                            l = (-1) ** i

                            # Finding the correspondent x,y coordinates of the
                            # vertex from the edge number
                            if e >= size[0] * size[1]:
                                j = i
                                delta = int(not (i ^ j))
                                x = int((e - size[0] * size[1]) / size[0])
                                y = ((e - size[0] * size[1]) %
                                     size[1] - i - l) % size[1]
                            else:
                                j = int(not i)
                                delta = int(not (i ^ j))
                                x = (e % size[0] - i - l) % size[0]
                                y = int(e / size[0])

                            if e in broken_links.value:
                                bl = 0
                            else:
                                bl = l

                            m = (((x + bl * (1 - delta)) % size[0]) * size[1] + (y + bl * delta) % size[1]) * coin_size + \
                                (i + bl) * size_per_coin + \
                                (abs(j + bl) % size_per_coin)
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
            if repr_format == constants.StateRepresentationFormatCoinPosition:
                def __map(xy):
                    x = xy % size[0]
                    y = int(xy / size[0])

                    for i in range(size_per_coin):
                        l = (-1) ** i
                        for j in range(size_per_coin):
                            delta = int(not (i ^ j))

                            m = (i * size_per_coin + j) * size_xy + \
                                ((x + l * (1 - delta)) %
                                 size[0]) * size[1] + (y + l * delta) % size[1]
                            n = (i * size_per_coin + j) * \
                                size_xy + x * size[1] + y

                            yield m, n, 1
            elif repr_format == constants.StateRepresentationFormatPositionCoin:
                def __map(xy):
                    x = xy % size[0]
                    y = int(xy / size[0])

                    for i in range(size_per_coin):
                        l = (-1) ** i
                        for j in range(size_per_coin):
                            delta = int(not (i ^ j))

                            m = (((x + l * (1 - delta)) % size[0]) * size[1] + (y + l * delta) % size[1]) * \
                                coin_size + i * size_per_coin + j
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
