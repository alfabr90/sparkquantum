from datetime import datetime

from pyspark import StorageLevel

from sparkquantum.dtqw.coin.coin import Coin
from sparkquantum.dtqw.operator import Operator
from sparkquantum.utils.utils import Utils
from sparkquantum.dtqw.mesh.mesh import is_mesh

__all__ = ['Coin2D']


class Coin2D(Coin):
    """Top-level class for two-dimensional coins."""

    def __init__(self):
        """Build a top-level two-dimensional :py:class:`sparkquantum.dtqw.coin.coin.Coin` object."""
        super().__init__()

        self._size = 4

    def is_2d(self):
        """
        Check if this is a coin for two-dimensional meshes.

        Returns
        -------
        bool
            True if this coin is for two-dimensional meshes, False otherwise.

        """
        return True

    def __str__(self):
        return 'Two-dimensional Coin'

    def create_operator(
            self, mesh, coord_format=Utils.MatrixCoordinateDefault):
        """Build the coin operator for the walk.

        Parameters
        ----------
        mesh : :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh`
            A :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh` instance.
        coord_format : int, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is :py:const:`sparkquantum.utils.Utils.MatrixCoordinateDefault`.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.operator.Operator`
            The created operator using this coin.

        Raises
        ------
        TypeError
            If `mesh` is not valid.

        ValueError
            If `mesh` is not two-dimensional or if the chosen 'quantum.dtqw.state.representationFormat' configuration is not valid.

        """
        if not is_mesh(mesh):
            self._logger.error(
                "expected 'Mesh' instance, not '{}'".format(
                    type(mesh)))
            raise TypeError(
                "expected 'Mesh' instance, not '{}'".format(
                    type(mesh)))

        if not mesh.is_2d():
            self._logger.error(
                "non correspondent coin and mesh dimensions")
            raise ValueError("non correspondent coin and mesh dimensions")

        coin_size = self._size
        mesh_size = mesh.size[0] * mesh.size[1]
        shape = (
            self._data.shape[0] *
            mesh_size,
            self._data.shape[1] *
            mesh_size)
        data = Utils.broadcast(self._spark_context, self._data)

        repr_format = int(
            Utils.get_conf(
                self._spark_context,
                'quantum.dtqw.state.representationFormat'))

        if repr_format == Utils.StateRepresentationFormatCoinPosition:
            # The coin operator is built by applying a tensor product between the chosen coin and
            # an identity matrix with the dimensions of the chosen mesh.
            def __map(xy):
                for i in range(data.value.shape[0]):
                    for j in range(data.value.shape[1]):
                        yield (i * mesh_size + xy, j * mesh_size + xy, data.value[i][j])
        elif repr_format == Utils.StateRepresentationFormatPositionCoin:
            # The coin operator is built by applying a tensor product between
            # an identity matrix with the dimensions of the chosen mesh and the
            # chosen coin.
            def __map(xy):
                for i in range(data.value.shape[0]):
                    for j in range(data.value.shape[1]):
                        yield (xy * coin_size + i, xy * coin_size + j, data.value[i][j])
        else:
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
            num_partitions = Utils.get_num_partitions(
                self._spark_context, expected_size)

            if num_partitions:
                rdd = rdd.partitionBy(
                    numPartitions=num_partitions
                )

        return Operator(rdd, shape, coord_format=coord_format)
