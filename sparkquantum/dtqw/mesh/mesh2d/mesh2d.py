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
        size : tuple or list of int
            Size of the mesh.
        broken_links : :py:class:`sparkquantum.dtqw.mesh.broken_links.BrokenLinks`, optional
            A :py:class:`sparkquantum.dtqw.mesh.broken_links.BrokenLinks` object.

        """
        super().__init__(size, broken_links=broken_links)

        self._coin_size = 4
        self._dimension = 2

    def _validate(self, size):
        if (not isinstance(size, (list, tuple)) or len(size) != 2
                or size[0] <= 0 or size[1] <= 0):
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

    def has_site(self, site):
        """Indicate whether this mesh comprises a site.

        Parameters
        ----------
        site : int
            Site number.

        Returns
        -------
        bool
            True if this mesh comprises the site, False otherwise.

        Raises
        ------
        ValueError
            If `site` is invalid, i.e., has a negative value.

        """
        if site < 0:
            self._logger.error("invalid site number")
            raise ValueError("invalid site number")

        return site < (self._size[0] * self._size[1])

    def has_coordinates(self, coordinate):
        """Indicate whether the coordinates are inside this mesh.

        Parameters
        ----------
        coordinate : tuple or list
            The coordinates.

        Returns
        -------
        bool
            True if this mesh comprises the coordinates, False otherwise.

        """
        return (coordinate[0] >= 0 and coordinate[0] < self._size[0] and
                coordinate[1] >= 0 and coordinate[1] < self._size[1])

    def to_site(self, coordinate):
        """Get the site number from the correspondent coordinates.

        Parameters
        ----------
        coordinate : tuple or list
            The coordinates.

        Returns
        -------
        int
            The site number.

        Raises
        ------
        ValueError
            If the coordinates are out of the mesh boundaries.

        """
        if (coordinate[0] < 0 or coordinate[0] >= self._size[0]
                or coordinate[1] < 0 or coordinate[1] >= self._size[1]):
            self._logger.error("coordinates out of mesh boundaries")
            raise ValueError("coordinates out of mesh boundaries")

        return coordinate[0] * self._size[1] + coordinate[1]

    def to_coordinates(self, site):
        """Get the coordinates from the correspondent site.

        Parameters
        ----------
        site : int
            Site number.

        Raises
        -------
        tuple or list
            The coordinates.

        Raises
        ------
        ValueError
            If the site number is out of the mesh boundaries.

        """
        if site < 0 or site >= self._size[0] * self._size[1]:
            self._logger.error("site number out of mesh boundaries")
            raise ValueError("site number out of mesh boundaries")

        return (int(site / self._size[1]), site % self._size[1])
