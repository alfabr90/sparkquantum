from sparkquantum.dtqw.mesh.grid.grid import Grid

__all__ = ['TwoDimensional']


class TwoDimensional(Grid):
    """Top-level class for two-dimensional grids."""

    def __init__(self, shape, percolation=None):
        """Build a top-level two-dimensional grid object.

        Parameters
        ----------
        shape : tuple of int
            Shape of this grid. Must be 2-dimensional.
        percolation : :py:class:`sparkquantum.dtqw.mesh.percolation.Percolation`, optional
            A percolation object.

        """
        super().__init__(shape, percolation=percolation)

        if len(shape) != 2:
            raise ValueError("invalid shape")

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

        return '2d grid with shape {}{}'.format(
            self._shape, percolation)
