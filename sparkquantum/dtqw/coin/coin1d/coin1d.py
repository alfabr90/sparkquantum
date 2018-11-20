from datetime import datetime

from pyspark import StorageLevel

from sparkquantum.dtqw.coin.coin import Coin
from sparkquantum.dtqw.operator import Operator
from sparkquantum.dtqw.mesh.mesh import is_mesh
from sparkquantum.utils.utils import Utils

__all__ = ['Coin1D']


class Coin1D(Coin):
    """Top-level class for 1-dimensional coins."""

    def __init__(self, spark_context):
        """Build a top-level 1-dimensional `Coin` object.

        Parameters
        ----------
        spark_context : `SparkContext`
            The `SparkContext` object.

        """
        super().__init__(spark_context)

        self._size = 2

    def is_1d(self):
        """Check if this is a coin for 1-dimensional meshes.

        Returns
        -------
        bool

        """
        return True

    def _create_rdd(self, mesh, coord_format, storage_level):
        coin_size = self._size
        mesh_size = mesh.size
        shape = (self._data.shape[0] * mesh_size, self._data.shape[1] * mesh_size)
        data = Utils.broadcast(self._spark_context, self._data)

        repr_format = int(Utils.get_conf(self._spark_context, 'quantum.dtqw.state.representationFormat'))

        if repr_format == Utils.StateRepresentationFormatCoinPosition:
            # The coin operator is built by applying a tensor product between the chosen coin and
            # an identity matrix with the dimensions of the chosen mesh.
            def __map(x):
                for i in range(data.value.shape[0]):
                    for j in range(data.value.shape[1]):
                        yield (i * mesh_size + x, j * mesh_size + x, data.value[i][j])
        elif repr_format == Utils.StateRepresentationFormatPositionCoin:
            # The coin operator is built by applying a tensor product between
            # an identity matrix with the dimensions of the chosen mesh and the chosen coin.
            def __map(x):
                for i in range(data.value.shape[0]):
                    for j in range(data.value.shape[1]):
                        yield (x * coin_size + i, x * coin_size + j, data.value[i][j])
        else:
            if self._logger:
                self._logger.error("invalid representation format")
            raise ValueError("invalid representation format")

        rdd = self._spark_context.range(
            mesh_size
        ).flatMap(
            __map
        )

        if coord_format == Utils.MatrixCoordinateMultiplier or coord_format == Utils.MatrixCoordinateMultiplicand:
            rdd = Utils.change_coordinate(
                rdd, Utils.MatrixCoordinateDefault, new_coord=coord_format
            )

            expected_elems = len(self._data) * mesh_size
            expected_size = Utils.get_size_of_type(complex) * expected_elems
            num_partitions = Utils.get_num_partitions(self._spark_context, expected_size)

            if num_partitions:
                rdd = rdd.partitionBy(
                    numPartitions=num_partitions
                )

        return rdd, shape

    def create_operator(self, mesh, coord_format=Utils.MatrixCoordinateDefault, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Build the coin operator for the walk.

        Parameters
        ----------
        mesh : `Mesh`
            A `Mesh` instance.
        coord_format : int, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is `Utils.MatrixCoordinateDefault`.
        storage_level : `StorageLevel`, optional
            The desired storage level when materializing the RDD. Default value is `StorageLevel.MEMORY_AND_DISK`.

        Returns
        -------
        `Operator`

        """
        if self._logger:
            self._logger.info("building coin operator...")

        initial_time = datetime.now()

        if not is_mesh(mesh):
            if self._logger:
                self._logger.error("expected 'Mesh' instance, not '{}'".format(type(mesh)))
            raise TypeError("expected 'Mesh' instance, not '{}'".format(type(mesh)))

        if not mesh.is_1d():
            if self._logger:
                self._logger.error("non correspondent coin and mesh dimensions")
            raise ValueError("non correspondent coin and mesh dimensions")

        rdd, shape = self._create_rdd(mesh, coord_format, storage_level)

        operator = Operator(rdd, shape, coord_format=coord_format).materialize(storage_level)

        self._profile(operator, initial_time)

        return operator
