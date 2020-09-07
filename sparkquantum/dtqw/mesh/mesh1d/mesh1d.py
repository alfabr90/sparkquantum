from pyspark import StorageLevel

from sparkquantum import util
from sparkquantum.dtqw.mesh.mesh import Mesh

__all__ = ['Mesh1D']


class Mesh1D(Mesh):
    """Top-level class for one-dimensional meshes."""

    def __init__(self, size, broken_links=None):
        """Build a top-level one-dimensional :py:class:`sparkquantum.dtqw.mesh.mesh.Mesh` object.

        Parameters
        ----------
        size : tuple or list of int
            Size of the mesh.
        broken_links : :py:class:`sparkquantum.dtqw.mesh.broken_links.BrokenLinks`, optional
            A :py:class:`sparkquantum.dtqw.mesh.broken_links.BrokenLinks` object.

        """
        super().__init__(size, broken_links=broken_links)

        self._coin_size = 2
        self._dimension = 1

    def _validate(self, size):
        if (not isinstance(size, (tuple, list))
                or len(size) != 1 or size[0] <= 0):
            self._logger.error("invalid size")
            raise ValueError("invalid size")

    def _define_size(self, size):
        self._validate(size)
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
        return self._size[0]

    def __str__(self):
        """Build a string representing this mesh.

        Returns
        -------
        str
            The string representation of this mesh.

        """
        return 'One-dimensional Mesh {}'.format(self.__strcomp__())

    def center(self):
        """Return the site number of the center of this mesh.

        Returns
        -------
        int
            The center site number.

        """
        return int((self._size[0] - 1) / 2)

    def center_coordinates(self):
        """Return the coordinates of the center site of this mesh.

        Returns
        -------
        tuple or list
            The coordinates of the center site.

        """
        return (self.center(), )

    def axis(self):
        """Build a generator with the size of this mesh.

        Returns
        -------
        range
            The range with the size of this mesh.

        """
        return range(self._size[0])

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

        """
        if site < 0:
            self._logger.error("invalid site number")
            raise ValueError("invalid site number")

        return site < self._size[0]

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
        return coordinate[0] >= 0 and coordinate[0] < self._size[0]

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
        if coordinate[0] < 0 or coordinate[0] >= self._size[0]:
            self._logger.error("coordinates out of mesh boundaries")
            raise ValueError("coordinates out of mesh boundaries")

        return coordinate[0]

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
        if site < 0 or site >= self._size[0]:
            self._logger.error("site number out of mesh boundaries")
            raise ValueError("site number out of mesh boundaries")

        return (site, )
