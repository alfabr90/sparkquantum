import numpy as np
from datetime import datetime

from pyspark import StorageLevel

from sparkquantum.dtqw.mesh.mesh2d.diagonal.diagonal import Diagonal
from sparkquantum.dtqw.operator import Operator
from sparkquantum.utils.utils import Utils

__all__ = ['LatticeDiagonal']


class LatticeDiagonal(Diagonal):
    """Class for Diagonal Lattice mesh."""

    def __init__(self, spark_context, size, broken_links=None):
        """Build a Diagonal Lattice `Mesh` object.

        Parameters
        ----------
        spark_context : `SparkContext`
            The `SparkContext` object.
        size : tuple
            Size of the mesh.
        broken_links : `BrokenLinks`, optional
            A `BrokenLinks` object.

        """
        super().__init__(spark_context, size, broken_links=broken_links)

    def _define_size(self, size):
        if not self._validate(size):
            if self._logger:
                self._logger.error("invalid size")
            raise ValueError("invalid size")

        return 2 * size[0] + 1, 2 * size[1] + 1

    def title(self):
        return 'Diagonal Lattice'

    def axis(self):
        return np.meshgrid(
            range(- int((self._size[0] - 1) / 2), int((self._size[0] - 1) / 2) + 1),
            range(- int((self._size[1] - 1) / 2), int((self._size[1] - 1) / 2) + 1),
            indexing='ij'
        )

    def check_steps(self, steps):
        """Check if the number of steps is valid for the size of the mesh.

        Parameters
        ----------
        steps : int

        Returns
        -------
        bool

        """
        return steps <= int((self._size[0] - 1) / 2) and steps <= int((self._size[1] - 1) / 2)

    def _create_rdd(self, coord_format, storage_level):
        coin_size = self._coin_size
        size_per_coin = int(coin_size / self._dimension)
        size = self._size
        num_edges = self._num_edges
        size_xy = size[0] * size[1]
        shape = (coin_size * size_xy, coin_size * size_xy)
        broken_links = None

        repr_format = int(Utils.get_conf(self._spark_context, 'quantum.dtqw.state.representationFormat'))

        if self._broken_links:
            broken_links = self._broken_links.generate(num_edges)

            generation_mode = Utils.get_conf(self._spark_context, 'quantum.dtqw.mesh.brokenLinks.generationMode')

            if generation_mode == Utils.BrokenLinksGenerationModeRDD:
                if repr_format == Utils.StateRepresentationFormatCoinPosition:
                    def __map(e):
                        """e = (edge, (edge, broken or not))"""
                        for i in range(size_per_coin):
                            l1 = (-1) ** i
                            for j in range(size_per_coin):
                                l2 = (-1) ** j

                                # Finding the correspondent x,y coordinates of the vertex from the edge number
                                x = (e[1][0] % size[0] - i - l1) % size[0]
                                y = (int(e[1][0] / size[0]) - j - l2) % size[1]

                                if e[1][1]:
                                    bl1, bl2 = 0, 0
                                else:
                                    bl1, bl2 = l1, l2

                                m = ((i + bl1) * size_per_coin + (j + bl2)) * size_xy + \
                                    ((x + bl1) % size[0]) * size[1] + ((y + bl2) % size[1])
                                n = ((1 - i) * size_per_coin + (1 - j)) * size_xy + x * size[1] + y

                                yield m, n, 1
                elif repr_format == Utils.StateRepresentationFormatPositionCoin:
                    def __map(e):
                        """e = (edge, (edge, broken or not))"""
                        for i in range(size_per_coin):
                            l1 = (-1) ** i
                            for j in range(size_per_coin):
                                l2 = (-1) ** j

                                # Finding the correspondent x,y coordinates of the vertex from the edge number
                                x = (e[1][0] % size[0] - i - l1) % size[0]
                                y = (int(e[1][0] / size[0]) - j - l2) % size[1]

                                if e[1][1]:
                                    bl1, bl2 = 0, 0
                                else:
                                    bl1, bl2 = l1, l2

                                m = (((x + bl1) % size[0]) * size[1] + ((y + bl2) % size[1])) * coin_size + \
                                    (i + bl1) * size_per_coin + (j + bl2)
                                n = (x * size[1] + y) * coin_size + (1 - i) * size_per_coin + (1 - j)

                                yield m, n, 1
                else:
                    if self._logger:
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
                            l1 = (-1) ** i
                            for j in range(size_per_coin):
                                l2 = (-1) ** j

                                # Finding the correspondent x,y coordinates of the vertex from the edge number
                                x = (e % size[0] - i - l1) % size[0]
                                y = (int(e / size[0]) - j - l2) % size[1]

                                if e in broken_links.value:
                                    bl1, bl2 = 0, 0
                                else:
                                    bl1, bl2 = l1, l2

                                m = ((i + bl1) * size_per_coin + (j + bl2)) * size_xy + \
                                    ((x + bl1) % size[0]) * size[1] + ((y + bl2) % size[1])
                                n = ((1 - i) * size_per_coin + (1 - j)) * size_xy + x * size[1] + y

                                yield m, n, 1
                elif repr_format == Utils.StateRepresentationFormatPositionCoin:
                    def __map(e):
                        """e = (edge, (edge, broken or not))"""
                        for i in range(size_per_coin):
                            l1 = (-1) ** i
                            for j in range(size_per_coin):
                                l2 = (-1) ** j

                                # Finding the correspondent x,y coordinates of the vertex from the edge number
                                x = (e % size[0] - i - l1) % size[0]
                                y = (int(e / size[0]) - j - l2) % size[1]

                                if e in broken_links.value:
                                    bl1, bl2 = 0, 0
                                else:
                                    bl1, bl2 = l1, l2

                                m = (((x + bl1) % size[0]) * size[1] + ((y + bl2) % size[1])) * coin_size + \
                                    (i + bl1) * size_per_coin + (j + bl2)
                                n = (x * size[1] + y) * coin_size + (1 - i) * size_per_coin + (1 - j)

                                yield m, n, 1
                else:
                    if self._logger:
                        self._logger.error("invalid representation format")
                    raise ValueError("invalid representation format")

                rdd = self._spark_context.range(
                    num_edges
                ).flatMap(
                    __map
                )
            else:
                if self._logger:
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

                            m = (i * size_per_coin + j) * size_xy + ((x + l1) % size[0]) * size[1] + ((y + l2) % size[1])
                            n = (i * size_per_coin + j) * size_xy + x * size[1] + y

                            yield m, n, 1
            elif repr_format == Utils.StateRepresentationFormatPositionCoin:
                def __map(xy):
                    x = xy % size[0]
                    y = int(xy / size[0])

                    for i in range(size_per_coin):
                        l1 = (-1) ** i
                        for j in range(size_per_coin):
                            l2 = (-1) ** j

                            m = (((x + l1) % size[0]) * size[1] + ((y + l2) % size[1])) * coin_size + i * size_per_coin + j
                            n = (x * size[1] + y) * coin_size + i * size_per_coin + j

                            yield m, n, 1
            else:
                if self._logger:
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
            num_partitions = Utils.get_num_partitions(self._spark_context, expected_size)

            if num_partitions:
                rdd = rdd.partitionBy(
                    numPartitions=num_partitions
                )

        return (rdd, shape, broken_links)

    def create_operator(self, coord_format=Utils.MatrixCoordinateDefault, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Build the shift operator for the walk.

        Parameters
        ----------
        coord_format : bool, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is `Utils.MatrixCoordinateDefault`.
        storage_level : `StorageLevel`, optional
            The desired storage level when materializing the RDD. Default value is `StorageLevel.MEMORY_AND_DISK`.

        Returns
        -------
        `Operator`

        Raises
        ------
        `ValueError`

        """
        if self._logger:
            self._logger.info("building shift operator...")

        initial_time = datetime.now()

        rdd, shape, broken_links = self._create_rdd(coord_format, storage_level)

        operator = Operator(rdd, shape, data_type=int, coord_format=coord_format).materialize(storage_level)

        if broken_links:
            broken_links.unpersist()

        self._profile(operator, initial_time)

        return operator
