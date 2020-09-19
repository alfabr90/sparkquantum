from sparkquantum.dtqw.mesh.grid.onedim.onedim import OneDimensional

__all__ = ['Cycle']


class Cycle(OneDimensional):
    """Class for Cycle grids."""

    def __init__(self, shape, percolation=None):
        """Build a Cycle object.

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

        return 'Cycle grid with shape {}{}'.format(self._shape, percolation)
