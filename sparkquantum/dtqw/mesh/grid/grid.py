from sparkquantum.dtqw.mesh.mesh import Mesh
from sparkquantum.math.util import is_shape

__all__ = ['Grid']


class Grid(Mesh):
    """Top-level class for grids."""

    def __init__(self, shape, percolation=None):
        """Build a top-level grid object.

        Parameters
        ----------
        shape : tuple of int
            Shape of this grid. Can be n-dimensional. n must be positive.
        percolation : :py:class:`sparkquantum.dtqw.mesh.percolation.Percolation`, optional
            A percolation object.

        """
        super().__init__(percolation=percolation)

        if not is_shape(shape):
            raise ValueError("invalid shape")

        self._shape = shape

        self._ndim = len(shape)

        sites = 1

        for d in range(self._ndim):
            sites *= self._shape[d]

        self._sites = sites

        self._edges = None

    @property
    def shape(self):
        """tuple of int"""
        return self._shape

    @property
    def ndim(self):
        """int"""
        return self._ndim

    @property
    def sites(self):
        """int"""
        return self._sites

    @property
    def edges(self):
        """int"""
        return self._edges

    def __str__(self):
        """Build a string representing this grid.

        Returns
        -------
        str
            The string representation of this grid.

        """
        percolation = ''

        if self._percolation is not None:
            percolation = ' and {}'.format(self._percolation)

        return '{}d grid with shape {}{}'.format(
            self._ndim, self._shape, percolation)

    def center(self, dim=None, coord=False):
        """Get the center site number or coordinate of a dimension or of the entire grid.

        Parameters
        ----------
        dim : int, optional
            The chosen dimension to get the center site. Default value is None.
        coord : bool, optional
            Indicate to return the center site in coordinates. Default value is False.

        Returns
        -------
        int or tuple of int
            The center site of a dimension of this grid or of the entire grid, whether in coordinates or not.

        """
        if dim is not None:
            if dim < 0 or dim >= self._ndim:
                self._logger.error("invalid dimension")
                raise ValueError("invalid dimension")

            return int((self._shape[dim] - 1) / 2)

        if coord:
            return tuple([self.center(d) for d in range(self._ndim)])
        else:
            def __center(ndim):
                if ndim == 1:
                    return self.center(dim=ndim - 1)
                else:
                    accsites = 1

                    for d in range(ndim - 1):
                        accsites *= self._shape[d]

                    return accsites * \
                        self.center(dim=ndim - 1) + __center(ndim - 1)

            return __center(self._ndim)

    def has_site(self, site):
        """Indicate whether this grid comprises a site.

        Parameters
        ----------
        site : int
            Site number.

        Returns
        -------
        bool
            True if this grid comprises the site, False otherwise.

        """
        return site >= 0 and site < self._sites

    def has_coordinate(self, coord, dim=None):
        """Indicate whether a coordinate of a specific dimension or all coordinates are inside this grid.

        Parameters
        ----------
        coord : int or tuple of int
            A coordinate or all coordinates to be checked.
        dim : int, optional
            The chosen dimension. Default value is None.

        Returns
        -------
        bool
            True if this grid comprises the coordinates, False otherwise.

        """
        r = self.axis()

        if dim is not None:
            return coord >= r[dim].start and coord < r[dim].stop

        for d in range(self._ndim):
            if coord[d] < r[d].start or coord[d] >= r[d].stop:
                return False

        return True

    def to_site(self, coord):
        """Get the correspondent site number from coordinates.

        Parameters
        ----------
        coord : tuple of int
            The coordinates.

        Returns
        -------
        int
            The correspondent site number.

        Raises
        ------
        ValueError
            If the coordinates are invalid or are out of the grid boundaries.

        """
        if len(coord) != self._ndim:
            self._logger.error("invalid coordinates")
            raise ValueError("invalid coordinates")

        for d in range(self._ndim):
            if coord[d] < 0 or coord[d] >= self._shape[d]:
                self._logger.error("coordinates out of grid boundaries")
                raise ValueError("coordinates out of grid boundaries")

        def __to_site(ndim):
            if ndim == 1:
                return coord[ndim - 1]
            else:
                accsites = 1

                for d in range(ndim - 1):
                    accsites *= self._shape[d]

                return coord[ndim - 1] * accsites + __to_site(ndim - 1)

        return __to_site(self._ndim)

    def to_coordinate(self, site):
        """Get the correspondent coordinates from a site number.

        Parameters
        ----------
        site : int
            Site number.

        Returns
        -------
        tuple of int
            The correspondent coordinates.

        Raises
        ------
        ValueError
            If the site number is out of the grid boundaries.

        """
        if site < 0 or site >= self._sites:
            self._logger.error("site number out of grid boundaries")
            raise ValueError("site number out of grid boundaries")

        def __to_coordinate(ndim):
            if ndim == 1:
                return (site % self._shape[ndim - 1], )
            else:
                accsites = 1

                for d in range(ndim - 1):
                    accsites *= self._shape[d]

                return (int(site / accsites) % self._shape[ndim - 1], ) + \
                    __to_coordinate(ndim - 1)

        return tuple(reversed(__to_coordinate(self._ndim)))

    def axis(self):
        """Get the ranges corresponding to coordinates of this grid.

        Returns
        -------
        tuple of range
            The ranges corresponding to coordinates of this grid.

        """
        return tuple([range(self._shape[d]) for d in range(self._ndim)])
