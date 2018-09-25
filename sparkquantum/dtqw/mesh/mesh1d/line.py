from datetime import datetime

from pyspark import StorageLevel

from sparkquantum.dtqw.mesh.mesh1d.mesh1d import Mesh1D
from sparkquantum.dtqw.operator import Operator
from sparkquantum.utils.utils import Utils

__all__ = ['Line']


class Line(Mesh1D):
    """Class for Line mesh."""

    def __init__(self, spark_context, size, broken_links=None):
        """
        Build a Line mesh object.

        Parameters
        ----------
        spark_context : SparkContext
            The SparkContext object.
        size : int
            Size of the mesh.
        broken_links : BrokenLinks, optional
            A BrokenLinks object.
        """
        super().__init__(spark_context, size, broken_links=broken_links)

    def _define_size(self, size):
        if not self._validate(size):
            if self._logger:
                self._logger.error("invalid size")
            raise ValueError("invalid size")

        return 2 * size + 1

    def axis(self):
        return range(- int((self._size - 1) / 2), int((self._size - 1) / 2) + 1)

    def check_steps(self, steps):
        """
        Check if the number of steps is valid for the size of the mesh.

        Parameters
        ----------
        steps : int

        Returns
        -------
        bool

        """
        return steps <= int((self._size - 1) / 2)

    def create_operator(self, coord_format=Utils.CoordinateDefault, storage_level=StorageLevel.MEMORY_AND_DISK):
        """
        Build the shift operator for the walk.

        Parameters
        ----------
        coord_format : bool, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is Utils.CoordinateDefault.
        storage_level : StorageLevel, optional
            The desired storage level when materializing the RDD. Default value is StorageLevel.MEMORY_AND_DISK.

        Returns
        -------
        Operator

        Raises
        ------
        ValueError

        """
        if self._logger:
            self._logger.info("building shift operator...")

        initial_time = datetime.now()

        coin_size = 2
        size = self._size
        num_edges = self._num_edges
        shape = (coin_size * size, coin_size * size)

        if self._broken_links:
            broken_links = self._broken_links.generate(num_edges)

            generation_mode = Utils.get_conf(self._spark_context, 'quantum.dtqw.mesh.brokenLinks.generationMode', default='broadcast')

            if generation_mode == 'rdd':
                def __map(e):
                    """e = (edge, (edge, broken or not))"""
                    for i in range(coin_size):
                        l = (-1) ** i

                        # Finding the correspondent x coordinate of the vertex from the edge number
                        x = (e[1][0] - i - l) % size

                        if e[1][1]:
                            l = 0

                        yield (i + l) * size + (x + l) % size, (1 - i) * size + x, 1

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
                def __map(e):
                    for i in range(coin_size):
                        l = (-1) ** i

                        # Finding the correspondent x coordinate of the vertex from the edge number
                        x = (e - i - l) % size

                        if e in broken_links.value:
                            l = 0

                        yield (i + l) * size + (x + l) % size, (1 - i) * size + x, 1

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
            def __map(x):
                for i in range(coin_size):
                    l = (-1) ** i
                    yield i * size + (x + l) % size, i * size + x, 1

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

        operator = Operator(rdd, shape, data_type=int, coord_format=coord_format).materialize(storage_level)

        if self._broken_links:
            broken_links.unpersist()

        self._profile(operator, initial_time)

        return operator
