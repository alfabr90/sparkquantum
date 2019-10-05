import numpy as np

from pyspark import StorageLevel

from sparkquantum.dtqw.mesh.mesh import Mesh
from sparkquantum.utils.utils import Utils

__all__ = ['Mesh2D']


class Mesh2D(Mesh):
    """Top-level class for 2-dimensional meshes."""

    def __init__(self, size, broken_links=None):
        """
        Build a top-level 2-dimensional `Mesh` object.

        Parameters
        ----------
        size : tuple
            Size of the mesh.
        broken_links : `BrokenLinks`, optional
            A `BrokenLinks` object.

        """
        super().__init__(size, broken_links=broken_links)

        self._coin_size = 4
        self._dimension = 2

    def _validate(self, size):
        if isinstance(size, (list, tuple)):
            if len(size) != 2:
                return False
        else:
            return False

        return True

    def _define_size(self, size):
        if not self._validate(size):
            if self._logger:
                self._logger.error("invalid size")
            raise ValueError("invalid size")

        return size

    def _define_num_edges(self, size):
        raise NotImplementedError

    def filename(self):
        if self._broken_links:
            probability = self._broken_links.probability
        else:
            probability = 0.0

        return "{}_{}-{}_{}".format(
            self.to_string(), self._size[0], self._size[1], probability
        )

    def axis(self):
        return np.meshgrid(
            range(self._size[0]),
            range(self._size[1]),
            indexing='ij'
        )

    def is_2d(self):
        """Check if this is a 2-dimensional mesh.

        Returns
        -------
        bool

        """
        return True

    def check_steps(self, steps):
        """Check if the number of steps is valid for the size of the mesh.

        Parameters
        ----------
        steps : int

        Raises
        -------
        `NotImplementedError`

        """
        raise NotImplementedError

    def create_operator(self, coord_format=Utils.MatrixCoordinateDefault, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Build the mesh operator.

        Parameters
        ----------
        coord_format : int, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is `Utils.MatrixCoordinateDefault`.
        storage_level : `StorageLevel`, optional
            The desired storage level when materializing the RDD. Default value is `StorageLevel.MEMORY_AND_DISK`.

        Raises
        -------
        `NotImplementedError`

        """
        raise NotImplementedError
