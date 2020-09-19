from sparkquantum.dtqw.mesh.grid.onedim.onedim import OneDimensional

__all__ = ['Line']


class Line(OneDimensional):
    """Class for Line grids."""

    def __init__(self, shape, percolation=None):
        """Build a Line object.

        Parameters
        ----------
        shape : tuple of int
            Shape of this grid. Must be 1-dimensional.
        percolation : :py:class:`sparkquantum.dtqw.mesh.percolation.Percolation`, optional
            A percolation object.

        """
        super().__init__(shape, percolation=percolation)

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

        return 'Line grid with shape {}{}'.format(self._shape, percolation)

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
        if coord:
            return 0 if dim is not None else (0, )

        return super().center(dim=dim, coord=coord)

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

        if not self.has_coordinate(coord):
            self._logger.error("coordinates out of grid boundaries")
            raise ValueError("coordinates out of grid boundaries")

        return coord[0] + super().center()

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
        if not self.has_site(site):
            self._logger.error("site number out of grid boundaries")
            raise ValueError("site number out of grid boundaries")

        return (site - super().center(), )

    def axis(self):
        """Get the ranges corresponding to coordinates of this grid.

        Returns
        -------
        tuple of range
            The ranges corresponding to coordinates of this grid.

        """
        return (range(-super().center(), super().center() + 1), )
