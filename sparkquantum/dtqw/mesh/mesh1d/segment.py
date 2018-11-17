from datetime import datetime

from pyspark import StorageLevel

from sparkquantum.dtqw.mesh.mesh1d.mesh1d import Mesh1D
from sparkquantum.dtqw.operator import Operator
from sparkquantum.utils.utils import Utils

__all__ = ['Segment']


class Segment(Mesh1D):
    """Class for Segment mesh."""

    def __init__(self, spark_context, size, broken_links=None):
        """Build a Segment `Mesh` object.

        Parameters
        ----------
        spark_context : `SparkContext`
            The `SparkContext` object.
        size : int
            Size of the mesh.
        broken_links : `BrokenLinks`, optional
            A `BrokenLinks` object.

        """
        super().__init__(spark_context, size, broken_links=broken_links)

    def check_steps(self, steps):
        """Check if the number of steps is valid for the size of the mesh.

        Parameters
        ----------
        steps : int

        Returns
        -------
        bool

        """
        return True

    def _create_rdd(self, coord_format, storage_level):
        coin_size = self._coin_size
        size_per_coin = int(coin_size / self._dimension)
        size = self._size
        num_edges = self._num_edges
        shape = (coin_size * size, coin_size * size)
        broken_links = None

        repr_format = int(Utils.get_conf(self._spark_context, 'quantum.representationFormat', default=Utils.RepresentationFormatCoinPosition))

        if self._broken_links:
            broken_links = self._broken_links.generate(num_edges)

            generation_mode = Utils.get_conf(self._spark_context, 'quantum.dtqw.mesh.brokenLinks.generationMode', default='broadcast')

            if generation_mode == 'rdd':
                if repr_format == Utils.RepresentationFormatCoinPosition:
                    def __map(e):
                        """e = (edge, (edge, broken or not))"""
                        for i in range(size_per_coin):
                            l = (-1) ** i

                            # Finding the correspondent x coordinate of the vertex from the edge number
                            x = (e[1][0] - i - l) % size

                            if e[1][1]:
                                bl = 0
                            else:
                                if x + l >= size or x + l < 0:
                                    bl = 0
                                else:
                                    bl = l

                            yield (i + bl) * size + x + bl, (1 - i) * size + x, 1
                elif repr_format == Utils.RepresentationFormatPositionCoin:
                    def __map(e):
                        """e = (edge, (edge, broken or not))"""
                        for i in range(size_per_coin):
                            l = (-1) ** i

                            # Finding the correspondent x coordinate of the vertex from the edge number
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
            elif generation_mode == 'broadcast':
                if repr_format == Utils.RepresentationFormatCoinPosition:
                    def __map(e):
                        for i in range(size_per_coin):
                            l = (-1) ** i

                            # Finding the correspondent x coordinate of the vertex from the edge number
                            x = (e - i - l) % size

                            if e in broken_links.value:
                                bl = 0
                            else:
                                if x + l >= size or x + l < 0:
                                    bl = 0
                                else:
                                    bl = l

                            yield (i + bl) * size + x + bl, (1 - i) * size + x, 1
                elif repr_format == Utils.RepresentationFormatPositionCoin:
                    def __map(e):
                        for i in range(size_per_coin):
                            l = (-1) ** i

                            # Finding the correspondent x coordinate of the vertex from the edge number
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
            if repr_format == Utils.RepresentationFormatCoinPosition:
                def __map(x):
                    for i in range(size_per_coin):
                        l = (-1) ** i

                        if x + l >= size or x + l < 0:
                            bl = 0
                        else:
                            bl = l

                        yield (i + bl) * size + x + bl, (1 - i) * size + x, 1
            elif repr_format == Utils.RepresentationFormatPositionCoin:
                def __map(x):
                    for i in range(size_per_coin):
                        l = (-1) ** i

                        if x + l >= size or x + l < 0:
                            bl = 0
                        else:
                            bl = l

                        yield (x + bl) * coin_size + i + bl, x * coin_size + (1 - i), 1
            else:
                if self._logger:
                    self._logger.error("invalid representation format")
                raise ValueError("invalid representation format")

            rdd = self._spark_context.range(
                size
            ).flatMap(
                __map
            )

        if coord_format == Utils.CoordinateMultiplier or coord_format == Utils.CoordinateMultiplicand:
            rdd = Utils.change_coordinate(
                rdd, Utils.CoordinateDefault, new_coord=coord_format
            )

            expected_elems = coin_size * size
            expected_size = Utils.get_size_of_type(int) * expected_elems
            num_partitions = Utils.get_num_partitions(self._spark_context, expected_size)

            if num_partitions:
                rdd = rdd.partitionBy(
                    numPartitions=num_partitions
                )

        return (rdd, shape, broken_links)

    def create_operator(self, coord_format=Utils.CoordinateDefault, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Build the shift operator for the walk.

        Parameters
        ----------
        coord_format : bool, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is `Utils.CoordinateDefault`.
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
