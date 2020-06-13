from pyspark import StorageLevel

from sparkquantum.dtqw.mesh.mesh2d.mesh2d import Mesh2D
from sparkquantum.utils.utils import Utils

__all__ = ['Diagonal']


class Diagonal(Mesh2D):
    """Top-level class for Diagonal meshes."""

    def __init__(self, size, broken_links=None):
        """Build a top-level Diagonal :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh` object.

        Parameters
        ----------
        size : tuple
            Size of the mesh.
        broken_links : :py:class:`sparkquantum.dtqw.mesh.broken_links.BrokenLinks`, optional
            A :py:class:`sparkquantum.dtqw.mesh.broken_links.BrokenLinks` object.

        """
        super().__init__(size, broken_links=broken_links)

    def _define_num_edges(self, size):
        # The number of edges is based on the size of the mesh.
        # For the already implemented types of mesh, the border edges are the same.
        #
        # An example of a 5x5 diagonal mesh:
        #
        # 00 01 01 03 04 00 |
        #   O     O     O   |
        # 20 21 22 23 24 20 |
        #      O     O      |
        # 15 16 17 18 19 20 |
        #   O     O     O   | y
        # 10 11 12 13 14 10 |
        #      O     O      |
        # 05 06 07 08 09 05 |
        #   O     O     O   |
        # 00 01 02 03 04 00 |
        # -----------------
        #         x
        size = self._define_size(size)
        return size[0] * size[1]

    def __str__(self):
        return 'Two-dimensional Diagonal Mesh with dimension {}'.format(
            self._size)

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
