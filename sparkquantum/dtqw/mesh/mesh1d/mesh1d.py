from pyspark import StorageLevel

from sparkquantum.dtqw.mesh.mesh import Mesh
from sparkquantum.utils.utils import Utils

__all__ = ['Mesh1D']


class Mesh1D(Mesh):
    """Top-level class for one-dimensional meshes."""

    def __init__(self, size, broken_links=None):
        """Build a top-level one-dimensional :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh` object.

        Parameters
        ----------
        size : int
            Size of the mesh.
        broken_links : :py:class:`sparkquantum.dtqw.mesh.broken_links.BrokenLinks`, optional
            A :py:class:`sparkquantum.dtqw.mesh.broken_links.BrokenLinks` object.

        """
        super().__init__(size, broken_links=broken_links)

        self._coin_size = 2
        self._dimension = 1

    def _validate(self, size):
        if not isinstance(size, int):
            return False
        elif size <= 0:
            return False

        return True

    def _define_size(self, size):
        if not self._validate(size):
            self._logger.error("invalid size")
            raise ValueError("invalid size")

        return size

    def _define_num_edges(self, size):
        # The number of edges is the same of the size of the mesh.
        # For the already implemented types of mesh, the border edges are the same.
        #
        # An example of a 5x1 mesh:
        #
        # 00 O 01 O 02 O 03 O 04 O 00
        # ---------------------------
        #              x
        return self._define_size(size)

    def __str__(self):
        """Build a string representing this mesh.

        Returns
        -------
        str
            The string representation of this mesh.

        """
        return 'One-dimensional Mesh {}'.format(self.__strcomp__())

    def axis(self):
        """Build a generator with the size of this mesh.

        Returns
        -------
        range
            The range with the size of this mesh.

        """
        return range(self._size)

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
