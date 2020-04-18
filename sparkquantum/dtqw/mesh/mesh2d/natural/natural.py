from pyspark import StorageLevel

from sparkquantum.dtqw.mesh.mesh2d.mesh2d import Mesh2D
from sparkquantum.utils.utils import Utils

__all__ = ['Natural']


class Natural(Mesh2D):
    """Top-level class for Natural meshes."""

    def __init__(self, size, broken_links=None):
        """Build a top-level Natural :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh` object.

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
        #   25   30   35   40   45    |
        # 20 O 21 O 22 O 23 O 24 O 20 |
        #   29   34   39   44   49    |
        # 15 O 16 O 17 O 18 O 19 O 15 |
        #   28   33   38   43   48    |
        # 10 O 11 O 12 O 13 O 14 O 10 | y
        #   27   32   37   42   47    |
        # 05 O 06 O 07 O 08 O 09 O 05 |
        #   26   31   36   41   46    |
        # 00 O 01 O 02 O 03 O 04 O 00 |
        #   25   30   35   40   45    |
        # ---------------------------
        #              x
        size = self._define_size(size)
        return size[0] * size[1] + size[0] * size[1]

    def __str__(self):
        return 'Two-dimensional Natural Mesh with dimension {}'.format(
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
