from sparkquantum.dtqw.mesh.grid.twodim.natural.natural import Natural

__all__ = ['Torus']


class Torus(Natural):
    """Class for natural torus mesh."""

    def __init__(self, shape, percolation=None):
        """Build a natural torus object.

        Parameters
        ----------
        shape : tuple of int
            Shape of this grid. Must be 2-dimensional.
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

        return 'Natural torus with shape {}{}'.format(
            self._shape, percolation)
