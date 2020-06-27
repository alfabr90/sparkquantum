from datetime import datetime

from pyspark import StorageLevel

from sparkquantum.dtqw.coin.coin import Coin
from sparkquantum.dtqw.mesh.mesh import is_mesh
from sparkquantum.dtqw.operator import Operator
from sparkquantum.utils.utils import Utils

__all__ = ['Coin1D']


class Coin1D(Coin):
    """Top-level class for one-dimensional coins."""

    def __init__(self):
        """Build a top-level one-dimensional :py:class:`sparkquantum.dtqw.coin.coin.Coin` object."""
        super().__init__()

        self._size = 2

    def __str__(self):
        return 'One-dimensional Coin'

    def create_operator(self, mesh):
        """Build the coin operator for the walk.

        Parameters
        ----------
        mesh : :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh`
            A :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh` instance.

        Returns
        -------
        :py:class:`sparkquantum.dtqw.operator.Operator`
            The created operator using this coin.

        Raises
        ------
        TypeError
            If `mesh` is not valid.

        ValueError
            If `mesh` is not one-dimensional or if the chosen 'quantum.dtqw.state.representationFormat' configuration is not valid.

        """
        if not is_mesh(mesh):
            self._logger.error(
                "expected 'Mesh' instance, not '{}'".format(
                    type(mesh)))
            raise TypeError(
                "expected 'Mesh' instance, not '{}'".format(
                    type(mesh)))

        if mesh.dimension != 1:
            self._logger.error(
                "non correspondent coin and mesh dimensions")
            raise ValueError("non correspondent coin and mesh dimensions")

        coin_size = self._size
        mesh_size = mesh.size
        shape = (
            self._data.shape[0] *
            mesh_size,
            self._data.shape[1] *
            mesh_size)
        data = Utils.broadcast(self._spark_context, self._data)

        num_elements = self._size ** 2 * mesh_size

        repr_format = int(
            Utils.get_conf(
                self._spark_context,
                'quantum.dtqw.state.representationFormat'))

        if repr_format == Utils.StateRepresentationFormatCoinPosition:
            # The coin operator is built by applying a tensor product between the chosen coin and
            # an identity matrix with the dimensions of the chosen mesh.
            def __map(x):
                for i in range(data.value.shape[0]):
                    for j in range(data.value.shape[1]):
                        yield (i * mesh_size + x, j * mesh_size + x, data.value[i][j])
        elif repr_format == Utils.StateRepresentationFormatPositionCoin:
            # The coin operator is built by applying a tensor product between
            # an identity matrix with the dimensions of the chosen mesh and the
            # chosen coin.
            def __map(x):
                for i in range(data.value.shape[0]):
                    for j in range(data.value.shape[1]):
                        yield (x * coin_size + i, x * coin_size + j, data.value[i][j])
        else:
            self._logger.error("invalid representation format")
            raise ValueError("invalid representation format")

        rdd = self._spark_context.range(
            mesh_size
        ).flatMap(
            __map
        )

        return Operator(rdd, shape, num_elements=num_elements)
