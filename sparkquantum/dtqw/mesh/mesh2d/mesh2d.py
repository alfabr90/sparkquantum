import numpy as np

from pyspark import StorageLevel

from sparkquantum.dtqw.mesh.mesh import Mesh
from sparkquantum.utils.utils import Utils

__all__ = ['Mesh2D']


class Mesh2D(Mesh):
    """Top-level class for two-dimensional meshes."""

    def __init__(self, size, broken_links=None):
        """
        Build a top-level two-dimensional mesh object.

        Parameters
        ----------
        size : tuple
            Size of the mesh.
        broken_links : :py:class:`sparkquantum.dtqw.mesh.broken_links.BrokenLinks`, optional
            A :py:class:`sparkquantum.dtqw.mesh.broken_links.BrokenLinks` object.

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
            self._logger.error("invalid size")
            raise ValueError("invalid size")

        return size

    def _define_num_edges(self, size):
        raise NotImplementedError

    def __str__(self):
        """Build a string representing this mesh.

        Returns
        -------
        str
            The string representation of this mesh.

        """
        return 'Two-dimensional Mesh {}'.format(self.__strcomp__())

    def axis(self):
        """Build a meshgrid with the sizes of this mesh.

        Returns
        -------
        :py:class:`numpy.ndarray`
            The meshgrid with the sizes of this mesh.

        """
        return np.meshgrid(
            range(self._size[0]),
            range(self._size[1]),
            indexing='ij'
        )

    def check_steps(self, steps):
        """Check if the number of steps is valid for the size of the mesh.

        Parameters
        ----------
        steps : int
            Number of steps of the walk.

        Raises
        -------
        NotImplementedError
            This method must not be called from this class, because the successor classes should implement it.

        """
        raise NotImplementedError

    def create_operator(self, coord_format=Utils.MatrixCoordinateDefault,
                        storage_level=StorageLevel.MEMORY_AND_DISK):
        """Build the mesh operator.

        Parameters
        ----------
        coord_format : int, optional
            Indicate if the operator must be returned in an apropriate format for multiplications.
            Default value is :py:const:`sparkquantum.utils.Utils.MatrixCoordinateDefault`.
        storage_level : :py:class:`pyspark.StorageLevel`, optional
            The desired storage level when materializing the RDD. Default value is :py:const:`pyspark.StorageLevel.MEMORY_AND_DISK`.

        Raises
        -------
        NotImplementedError
            This method must not be called from this class, because the successor classes should implement it.

        """
        raise NotImplementedError
