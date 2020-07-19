import numpy as np

from pyspark import StorageLevel

from sparkquantum import util
from sparkquantum.dtqw.mesh.mesh import Mesh

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

    def __str__(self):
        """Build a string representing this mesh.

        Returns
        -------
        str
            The string representation of this mesh.

        """
        return 'Two-dimensional Mesh {}'.format(self.__strcomp__())

    def center(self):
        """Return the site number of the center of this mesh.

        Returns
        -------
        int
            The center site number.

        """
        return int((self._size[0] - 1) / 2) * \
            self._size[1] + int((self._size[1] - 1) / 2)

    def center_coordinates(self):
        """Return the coordinates of the center site of this mesh.

        Returns
        -------
        tuple or list
            The coordinates of the center site.

        """
        return (int((self._size[0] - 1) / 2), int((self._size[1] - 1) / 2))

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
