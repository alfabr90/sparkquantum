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
                self._logger.error("invalid size")
                raise ValueError("invalid size")
        else:
            self._logger.error("invalid size")
            raise ValueError("invalid size")

    def _define_size(self, size):
        self._validate(size)
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

    def center_x(self):
        """Return the site number of the center of this mesh for the `x` coordinate.

        Returns
        -------
        int
            The center site number for the `x` coordinate.

        """
        return int((self._size[0] - 1) / 2)

    def center_y(self):
        """Return the site number of the center of this mesh for the `y` coordinate.

        Returns
        -------
        int
            The center site number for the `y` coordinate.

        """
        return int((self._size[1] - 1) / 2)

    def center(self):
        """Return the site number of the center of this mesh.

        Returns
        -------
        int
            The center site number.

        """
        return self.center_x() * self._size[1] + self.center_y()

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

    def create_operator(self, storage_level=StorageLevel.MEMORY_AND_DISK):
        """Build the mesh operator.

        Parameters
        ----------
        storage_level : :py:class:`pyspark.StorageLevel`, optional
            The desired storage level when materializing the RDD. Default value is :py:const:`pyspark.StorageLevel.MEMORY_AND_DISK`.

        Raises
        -------
        NotImplementedError
            This method must not be called from this class, because the successor classes should implement it.

        """
        raise NotImplementedError
