from datetime import datetime

from pyspark import StorageLevel

from sparkquantum.dtqw.mesh.mesh2d.natural.natural import Natural
from sparkquantum.dtqw.operator import Operator
from sparkquantum.utils.utils import Utils

__all__ = ['BoxNatural']


class BoxNatural(Natural):
    """Class for Natural Box mesh."""

    def __init__(self, size, broken_links=None):
        """Build a Natural Box mesh object.

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
        return 'Natural Box {}'.format(self.__strcomp__())

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

    def create_operator(self, coord_format=Utils.MatrixCoordinateDefault):
        """Build the shift operator for the walk.

        Parameters
        ----------
        coord_format : bool, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is :py:const:`sparkquantum.utils.Utils.MatrixCoordinateDefault`.

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
                            pos1 = x + l * (1 - delta)
                            pos2 = y + l * delta

                            if e[1][1]:
                                l = 0
                            else:
                                # The border edges are considered broken so
                                # that they become reflexive
                                if pos1 >= size[0] or pos1 < 0 or pos2 >= size[1] or pos2 < 0:
                                    l = 0

                            m = ((i + l) * size_per_coin + (abs(j + l) % size_per_coin)) * size_xy + \
                                (x + l * (1 - delta)) * \
                                size[1] + (y + l * delta)
                            n = ((1 - i) * size_per_coin + (1 - j)) * \
                                size_xy + x * size[1] + y

                            yield m, n, 1
                elif repr_format == Utils.StateRepresentationFormatPositionCoin:
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
                            pos1 = x + l * (1 - delta)
                            pos2 = y + l * delta

                            if e[1][1]:
                                l = 0
                            else:
                                # The border edges are considered broken so
                                # that they become reflexive
                                if pos1 >= size[0] or pos1 < 0 or pos2 >= size[1] or pos2 < 0:
                                    l = 0

                            m = ((x + l * (1 - delta)) * size[1] + (y + l * delta)) * coin_size + \
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
            elif generation_mode == Utils.BrokenLinksGenerationModeBroadcast:
                if repr_format == Utils.StateRepresentationFormatCoinPosition:
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

                            pos1 = x + l * (1 - delta)
                            pos2 = y + l * delta

                            if e in broken_links.value:
                                bl = 0
                            else:
                                # The border edges are considered broken so
                                # that they become reflexive
                                if pos1 >= size[0] or pos1 < 0 or pos2 >= size[1] or pos2 < 0:
                                    bl = 0
                                else:
                                    bl = l

                            m = ((i + bl) * size_per_coin + (abs(j + bl) % size_per_coin)) * size_xy + \
                                (x + bl * (1 - delta)) * \
                                size[1] + (y + bl * delta)
                            n = ((1 - i) * size_per_coin + (1 - j)) * \
                                size_xy + x * size[1] + y

                            yield m, n, 1
                elif repr_format == Utils.StateRepresentationFormatPositionCoin:
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

                            pos1 = x + l * (1 - delta)
                            pos2 = y + l * delta

                            if e in broken_links.value:
                                bl = 0
                            else:
                                # The border edges are considered broken so
                                # that they become reflexive
                                if pos1 >= size[0] or pos1 < 0 or pos2 >= size[1] or pos2 < 0:
                                    bl = 0
                                else:
                                    bl = l

                            m = ((x + bl * (1 - delta)) * size[1] + (y + bl * delta)) * coin_size + \
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
            if repr_format == Utils.StateRepresentationFormatCoinPosition:
                def __map(xy):
                    x = xy % size[0]
                    y = int(xy / size[0])

                    for i in range(size_per_coin):
                        l = (-1) ** i
                        for j in range(size_per_coin):
                            delta = int(not (i ^ j))

                            pos1 = x + l * (1 - delta)
                            pos2 = y + l * delta

                            # The border edges are considered broken so that
                            # they become reflexive
                            if pos1 >= size[0] or pos1 < 0 or pos2 >= size[1] or pos2 < 0:
                                bl = 0
                            else:
                                bl = l

                            m = ((i + bl) * size_per_coin + (abs(j + bl) % size_per_coin)) * size_xy + \
                                (x + bl * (1 - delta)) * \
                                size[1] + (y + bl * delta)
                            n = ((1 - i) * size_per_coin + (1 - j)) * \
                                size_xy + x * size[1] + y

                            yield m, n, 1
            elif repr_format == Utils.StateRepresentationFormatPositionCoin:
                def __map(xy):
                    x = xy % size[0]
                    y = int(xy / size[0])

                    for i in range(size_per_coin):
                        l = (-1) ** i
                        for j in range(size_per_coin):
                            delta = int(not (i ^ j))

                            pos1 = x + l * (1 - delta)
                            pos2 = y + l * delta

                            # The border edges are considered broken so that
                            # they become reflexive
                            if pos1 >= size[0] or pos1 < 0 or pos2 >= size[1] or pos2 < 0:
                                bl = 0
                            else:
                                bl = l

                            m = ((x + bl * (1 - delta)) * size[1] + (y + bl * delta)) * coin_size + \
                                (i + bl) * size_per_coin + \
                                (abs(j + bl) % size_per_coin)
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

        if coord_format == Utils.MatrixCoordinateMultiplier or coord_format == Utils.MatrixCoordinateMultiplicand:
            rdd = Utils.change_coordinate(
                rdd, Utils.MatrixCoordinateDefault, new_coord=coord_format
            )

            expected_elems = coin_size * size_xy
            expected_size = Utils.get_size_of_type(int) * expected_elems
            num_partitions = Utils.get_num_partitions(
                self._spark_context, expected_size)

            if num_partitions:
                rdd = rdd.partitionBy(
                    numPartitions=num_partitions
                )

        return Operator(rdd, shape, int, coord_format)
